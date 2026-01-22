import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, trunc_normal_
from timm.models.registry import register_model
from einops import rearrange

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_norm=None, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        if qk_norm is not None:
            self.q_norm = qk_norm(head_dim)
            self.k_norm = qk_norm(head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
            # cls to token & token 2 cls & cls to cls

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = \
                torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
            relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1

            self.register_buffer("relative_position_index", relative_position_index)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos_bias=None, return_attention=False, return_qkv=False):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple) (B, H, N, C)
        if self.q_norm is not None:
            q = self.q_norm(q).type_as(v)
        if self.k_norm is not None:
            k = self.k_norm(k).type_as(v)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.relative_position_bias_table is not None:
            relative_position_bias = \
                self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                    self.window_size[0] * self.window_size[1] + 1,
                    self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if return_attention:
            return attn

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)

        if return_qkv:
            return x, qkv
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_norm=None, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, rel_pos_bias=None, return_attention=False, return_qkv=False):
        if return_attention:
            return self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, return_attention=True)
        if return_qkv:
            y, qkv = self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, return_qkv=return_qkv)
            x = x + self.drop_path(self.gamma_1 * y)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
            return x, qkv

        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, EEG_size=2000, patch_size=200, in_chans=1, embed_dim=200):
        super().__init__()

        num_patches = 62 * (EEG_size // patch_size)
        self.patch_shape = (1, EEG_size // patch_size)
        self.EEG_size = EEG_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=(1, patch_size), stride=(1, patch_size))

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class TemporalConv(nn.Module):
    def __init__(self, in_chans=1, out_chans=8):

        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 15), stride=(1, 8), padding=(0, 7))
        self.gelu1 = nn.GELU()
        self.norm1 = nn.GroupNorm(4, out_chans)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.gelu2 = nn.GELU()
        self.norm2 = nn.GroupNorm(4, out_chans)
        self.conv3 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.norm3 = nn.GroupNorm(4, out_chans)
        self.gelu3 = nn.GELU()

    def forward(self, x, **kwargs):
        x = rearrange(x, 'B N A T -> B (N A) T')
        B, NA, T = x.shape
        x = x.unsqueeze(1)
        x = self.gelu1(self.norm1(self.conv1(x)))
        x = self.gelu2(self.norm2(self.conv2(x)))
        x = self.gelu3(self.norm3(self.conv3(x)))
        x = rearrange(x, 'B C NA T -> B NA (T C)')
        return x


class NeuralTransformer(nn.Module):
    def __init__(self, EEG_size=1600, patch_size=200, in_chans=1, out_chans=8, num_classes=3, embed_dim=200,
                 depth=3,
                 num_heads=10, mlp_ratio=4., pretrained_weights_path='', qkv_bias=False, qk_norm=None, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=0.1,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False,
                 use_mean_pooling=True, init_scale=0.001, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = TemporalConv(out_chans=out_chans) if in_chans == 1 else PatchEmbed(EEG_size=EEG_size,
                                                                                              patch_size=patch_size,
                                                                                              in_chans=in_chans,
                                                                                              embed_dim=embed_dim)
        self.time_window = EEG_size // patch_size
        self.patch_size = patch_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, 128 + 1, embed_dim), requires_grad=False)
        else:
            self.pos_embed = None
        self.time_embed = nn.Parameter(torch.zeros(1, 16, embed_dim), requires_grad=False)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.rel_pos_bias = None
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.use_rel_pos_bias = use_rel_pos_bias
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_norm=qk_norm,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=None)
            for i in range(depth)])
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        if self.time_embed is not None:
            trunc_normal_(self.time_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)

        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=.02)
        self.apply(self._init_weights)
        self.fix_init_weight()

        if isinstance(self.head, nn.Linear):
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)
        #if pretrained_weights_path != '' and pretrained_weights_path is not None:
        #    self.load_pretrained_weights(pretrained_weights_path)
        self.n_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(self.n_parameters)

    def load_pretrained_weights(self, pretrained_weights_path):

        pretrained_dict = torch.load(pretrained_weights_path)

        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)


    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, input_chans=None, return_patch_tokens=False, return_all_tokens=False, **kwargs):
        batch_size, n, a, t = x.shape
        input_time_window = a if t == self.patch_size else t
        input_chans = n + 1
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            pos_embed_used = self.pos_embed[:, 0:input_chans, :] if input_chans is not None else self.pos_embed
            pos_embed = pos_embed_used[:, 1:, :].unsqueeze(2).expand(batch_size, -1, input_time_window, -1).flatten(1,
                                                                                                                    2)
            pos_embed = torch.cat((pos_embed_used[:, 0:1, :].expand(batch_size, -1, -1), pos_embed), dim=1)
            x = x + pos_embed
        if self.time_embed is not None:
            nc = n if t == self.patch_size else a
            time_embed = self.time_embed[:, 0:input_time_window, :].unsqueeze(1).expand(batch_size, nc, -1, -1).flatten(
                1, 2)
            x[:, 1:, :] += time_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=None)
        x = self.norm(x)
        if self.fc_norm is not None:
            if return_all_tokens:
                return self.fc_norm(x)
            t = x[:, 1:, :]
            if return_patch_tokens:
                return self.fc_norm(t)
            else:
                return self.fc_norm(t.mean(1))
        else:
            if return_all_tokens:
                return x
            elif return_patch_tokens:
                return x[:, 1:]
            else:
                return x[:, 0]

    def forward(self, x, input_chans=None, return_patch_tokens=True, return_all_tokens=False, **kwargs):
        x = self.forward_features(x, input_chans=input_chans, return_patch_tokens=return_patch_tokens,
                                  return_all_tokens=return_all_tokens, **kwargs)
        return x



class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
                self.head_dim * heads == embed_size
        )

        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask=None):
        N = query.shape[0]  # Batch size
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        Q = self.queries(query)
        K = self.keys(keys)     
        V = self.values(values)

        Q = Q.view(N, query_len, self.heads, self.head_dim)
        K = K.view(N, key_len,   self.heads, self.head_dim)
        V = V.view(N, value_len, self.heads, self.head_dim)
    

        energy = torch.einsum("nqhd,nkhd->nhqk", [Q, K])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, V]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out, attention


class FeatureFusionWithAdaptivePoolingAttention(nn.Module):
    def __init__(self, embed_size, heads, pooled_size=(10, 1)):
        super(FeatureFusionWithAdaptivePoolingAttention, self).__init__()
        self.pooled_size = pooled_size
        # self.multihead_attention = MultiHeadAttention(embed_size, heads)
        self.mha = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads, batch_first=True)
        # self.adaptive_pool = nn.AdaptiveAvgPool2d(pooled_size)

    def forward(self, features, index):
        batch_size = 0
        embed_size = 0
        processed_features = []
        for feature in features:
            batch_size, embed_size, a, b = feature.shape
            # pooled_feature = self.adaptive_pool(feature)
            pooled_feature = feature.reshape(batch_size * b, embed_size, -1).permute(0, 2, 1)
            processed_features.append(pooled_feature)
            
        query = sum(index[:, i:i+1, None] * processed_features[i]
            for i in range(5))
        keys = torch.cat(processed_features, dim=1)
        values = torch.cat(processed_features, dim=1)

        #fused_output, att = self.multihead_attention(values, keys, query)
        fused_output, att = self.mha(query,keys,values)
        fused_output = fused_output.view(batch_size, -1, self.pooled_size[-1], embed_size)


        # if index == 0:
        #     query = processed_features[0]
        #     keys = torch.cat(processed_features, dim=1)
        #     values = torch.cat(processed_features, dim=1)
        # elif index == 1:
        #     query = processed_features[1]
        #     keys = torch.cat(processed_features, dim=1)
        #     values = torch.cat(processed_features, dim=1)
        # elif index == 2:
        #     query = processed_features[2]
        #     keys = torch.cat(processed_features, dim=1)
        #     values = torch.cat(processed_features, dim=1)
        # elif index == 3:
        #     query = processed_features[3]
        #     keys = torch.cat(processed_features, dim=1)
        #     values = torch.cat(processed_features, dim=1)
        # elif index == 4:
        #     query = processed_features[4]
        #     keys = torch.cat(processed_features, dim=1)
        #     values = torch.cat(processed_features, dim=1)
        # stacked_features = torch.stack(processed_features, dim=1)  
        # index_expanded = index.unsqueeze(-1).unsqueeze(-1) 
        # query = (stacked_features * index_expanded).sum(dim=1) 
        # keys = torch.cat(processed_features, dim=1)  
        # values = torch.cat(processed_features, dim=1)  
        
        return fused_output


def splitdata(x, Cname):  # modify
    category_dict = {cat: [] for cat in ['F', 'P', 'T', 'C', 'O']}
    PF = ['AF3', 'FP1', 'FPZ', 'FP2', 'AF4']
    FF = ['F1', 'FZ', 'F2', 'FC1', 'FCZ', 'FC2']
    LF = ['F7', 'F5', 'F3', 'FC5', 'FC3']
    RF = ['F4', 'F6', 'F8', 'FC4', 'FC6', 'FT7']
    LT = ['T7', 'C5', 'C3', 'TP7', 'CP5', 'CP3']
    RT = ['C4', 'C6', 'T8', 'CP4', 'CP6', 'TP8', 'FT8']
    C = ['C1', 'CZ', 'C2', 'CP1', 'CPZ', 'CP2', 'P1', 'PZ', 'P2']
    LP = ['P7', 'P5', 'P3', 'PO7', 'PO5', 'CB1']
    RP = ['P4', 'P6', 'P8', 'PO6', 'PO8', 'CB2']
    O = ['PO3', 'POZ', 'PO4', 'O1', 'OZ', 'O2']

    for i, label in enumerate(Cname):
        if Cname[i] in PF or Cname[i] in FF or Cname[i] in LF or Cname[i] in RF:
            category_dict["F"].append(i)
        if Cname[i] in LT or Cname[i] in RT:
            category_dict["T"].append(i)
        if Cname[i] in LP or Cname[i] in RP:
            category_dict["P"].append(i)
        if Cname[i] in C:
            category_dict["C"].append(i)
        if Cname[i] in O:
            category_dict["O"].append(i)

    
    # for i, label in enumerate(Cname):
    #     if "F" in Cname[i]:
    #         category_dict["F"].append(i)
    #     if "P" in Cname[i]:
    #         if "FP" not in Cname[i]:
    #             category_dict["P"].append(i)
    #     if "T" in Cname[i]:
    #         category_dict["T"].append(i)
    #     if "C" in Cname[i]:
    #         if "CB" in Cname[i]:
    #             category_dict["P"].append(i)
    #         else:
    #             category_dict["C"].append(i)
    #     if "O" in Cname[i]:
    #         category_dict["O"].append(i)
    split_data = []

    for category, indices in category_dict.items():
        sliced_data = x[:, indices, :, :]
        split_data.append(sliced_data)

    return split_data


@register_model
def labram_base_patch200_200(source='', pretrained=False, **kwargs):
    model = NeuralTransformer(
        patch_size=200, embed_dim=200, depth=3, num_heads=10, mlp_ratio=4, pretrained_weights_path=source,
        qk_norm=partial(nn.LayerNorm, eps=1e-6),  # qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    return model


def alternate_sampling_2s(data):
    batch_size, channels, samples, points = data.shape
    new_data = torch.zeros(batch_size, channels, samples, points)

    for i in range(batch_size):
        for j in range(channels):
            for k in range(0, samples, 2):
                concatenated = torch.cat((data[i, j, k, :], data[i, j, k + 1, :]), dim=-1)
                new_data[i, j, k, :] = concatenated[0::2]
                new_data[i, j, k + 1, :] = concatenated[1::2]

    return new_data.to(data.device)

class AttentionPool2d(nn.Module):
    """
    Attention-based 2D pooling.
    Input:  [B, C, H, W]
    Output: [B, C, output_h, output_w]
    """
    def __init__(self, embed_dim, output_size=(10, 1), num_heads=4):
        super().__init__()
        self.output_h, self.output_w = output_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.query = nn.Parameter(torch.randn(1, self.output_h * self.output_w, embed_dim))
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim,
                                          num_heads=num_heads,
                                          batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H*W).transpose(1, 2)
        query = self.query.expand(B, -1, -1)
        pooled, _ = self.attn(query, x_flat, x_flat)
        pooled = pooled.transpose(1, 2).view(B, C, self.output_h, self.output_w)
        return pooled


def alternate_sampling_4s(data):
    batch_size, channels, samples, points = data.shape
    new_data = torch.zeros(batch_size, channels, samples, points)

    for i in range(batch_size):
        for j in range(channels):
            for k in range(0, samples, 4):
                concatenated = torch.cat((data[i, j, k, :],
                                          data[i, j, k + 1, :],
                                          data[i, j, k + 2, :],
                                          data[i, j, k + 3, :]), dim=-1)

                new_data[i, j, k, :] = concatenated[0::4]
                new_data[i, j, k + 1, :] = concatenated[1::4]
                new_data[i, j, k + 2, :] = concatenated[2::4]
                new_data[i, j, k + 3, :] = concatenated[3::4]

    return new_data.to(data.device)


class MLP(nn.Module):
    def __init__(self, input_size=200 * 5 * 10 * 3, hidden_size=128, output_size=5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.drop = nn.Dropout(0.2)

        #if pretrained_weights_path != '' and pretrained_weights_path is not None:
        #    self.load_pretrained_weights(pretrained_weights_path)

    def load_pretrained_weights(self, pretrained_weights_path):
        pretrained_dict = torch.load(pretrained_weights_path)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
            
    def forward(self, x,tau2=0.2,hard2=True):
        x = self.fc1(x)
        x = self.drop(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        feature_sums = x.sum(dim=0)
        att = F.softmax(feature_sums, dim=0)
        predictions = F.gumbel_softmax(x,tau=tau2,hard=hard2, dim=1)
        # predictions = F.gumbel_softmax(x,tau=0.3,hard=False, dim=1)
        # predictions = F.softmax(x / 0.01, dim=1)
        # output, _ = torch.mode(predictions)
        # return output, att
        return predictions, att

class AttnPool2d(nn.Module):
    def __init__(self, in_channels, out_h, out_w):
        super().__init__()
        self.out_h = out_h
        self.out_w = out_w
        self.attn = nn.Conv2d(in_channels, out_h * out_w, kernel_size=1, bias=False)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        score = self.attn(x) 
        score = score.view(B, self.out_h*self.out_w, -1)
        weight = F.softmax(score, dim=-1) 
        x_flat = x.view(B, C, -1) 
        pooled = torch.einsum('bcn, bkn -> bck', x_flat, weight)
        pooled = pooled.view(B, C, self.out_h, self.out_w)
        return pooled

class MSTAEEG(nn.Module):
    def __init__(self, channel_name=[], patch_size=200, embed_dim=200, num_classes=3, source="", **kwargs):
        super(MSTAEEG, self).__init__()
        self.encoders = nn.ModuleList(labram_base_patch200_200(source=source) for _ in range(15))
        self.heads = nn.ModuleList(nn.Linear(embed_dim, num_classes) for _ in range(5))
        self.attn2 = FeatureFusionWithAdaptivePoolingAttention(embed_dim * 3, 5)
        self.Cname = channel_name
        self.head1 = nn.Linear(embed_dim * 3, 128)
        self.head2 = nn.Linear(128, num_classes)
        self.drop = nn.Dropout(0.3)
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.ELU = nn.ELU()
        self.pool_layers = nn.AdaptiveAvgPool2d((10, 1))
        # self.pool_layers = AttnPool2d((embed_dim * 3,10, 1))
        self.MLP = MLP()

        self.n_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(self.n_parameters)

    def get_num_layers(self):
        return self.encoders[0].get_num_layers()

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def process_eeg_data(self, data):
        downsampled_2s = alternate_sampling_2s(data)
        downsampled_4s = alternate_sampling_4s(data)
        return [data, downsampled_2s.float(), downsampled_4s.float()]

    def forward(self, x, tau=0.3, hard=True):
        n, l, p = x.shape
        # set this
        x = x.reshape(-1, l, a, b)
        # x = x.reshape(-1, l, 4, 200)
        fea = [[] for _ in range(5)]
        concatfea = [None] * 5
        last_re = [None] * 5
        last_fea = [None] * 5
        last_feat = [None] * 5
        #set this
        channel_size1 = [1, 2, 3, 4, 5]

        x_list = splitdata(x, self.Cname)
        for i in range(5):
            x_data = x_list[i]
            list = self.process_eeg_data(x_data)
            for j in range(3):
                fea[i].append(self.encoders[i * 3 + j](list[j].reshape(n, channel_size1[i], 1, 200)))
            feat_list = []
            concatfea[i] = torch.cat(fea[i], dim=-1)
            concatfea[i] = concatfea[i].reshape(-1, channel_size1[i], 1, p * 3).permute(0, 3, 1, 2)
            for idx, tensor in enumerate(fea[i]):
                reshaped_tensor = tensor.reshape(-1, channel_size1[i], 1, p).permute(0, 3, 1, 2)
                feat_list.append(reshaped_tensor)
            last_fea[i] = self.encoders[i * 3 + 2](list[2].reshape(n, channel_size1[i], 1, 200))
            last_feat[i] = last_fea[i].reshape(n, -1, 1, p).permute(0, 3, 1, 2)
            last_re[i] = self.heads[i](last_fea[i]).reshape(n, -1, 1, self.num_classes)
            last_re[i] = last_re[i].mean(dim=1, keepdim=True)
            last_re[i] = last_re[i].reshape(n, -1)
        pooled_data = [None] * 5
        for i in range(len(concatfea)):
            pooled_data[i] = self.pool_layers(concatfea[i].reshape(n, p * 3, -1, 1))[:, :, :, :]
        concatenated_features = torch.cat(pooled_data, dim=1)
        flattened_features = concatenated_features.reshape(n, -1)
        index, att = self.MLP(flattened_features)
        # index, att = self.MLP(flattened_features,tau,hard)
        final_fe = self.attn2(pooled_data, index)
        final = self.head2(self.drop(self.ELU(self.head1(final_fe.reshape(n, -1, p * 3))))).reshape(n, -1, 1,
                                                                                                    self.num_classes)
        final = final.mean(dim=1, keepdim=True)
        final = final.mean(dim=2, keepdim=True)
        final = final.reshape(n, -1)


        return last_re, final, att

@register_model
def MSTA(num_classes=3, source='your path here', **kwargs):
    channel_name = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7',
                'FC5', 'FC3', 'FC1', 'FCZ', 'FC2',
                'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3',
                'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3',
                'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1',
                'OZ', 'O2', 'CB2']  # modify
    model = MSTAEEG(channel_name=channel_name, patch_size=200, embed_dim=200, num_classes=num_classes, source=source,
                   **kwargs)

    return model
