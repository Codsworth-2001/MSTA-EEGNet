import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from engine import train_one_epoch, evaluate
import time, datetime
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from MSTAEEGNet import MSTA
from copy import deepcopy
import collections


def save_model(output, name, model):
    model.save_checkpoint(save_dir=output, tag="checkpoint-%s" % name)


class NetModel(object):
    def __init__(self, args):
        self.args = args
        self.seed_all()
        self.set_model()

    def seed_all(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.args.seed)
            torch.cuda.manual_seed_all(self.args.seed)

    def set_dataset(self, train_set, val_set=None, test_set=None):
        self.normalizer = MinMaxScaler(feature_range=(-1 * self.args.data_range, self.args.data_range))
        self.train_set = train_set
        print(collections.Counter(self.train_set.label))
        if val_set is not None:
            n, l, p = self.train_set.dataset.shape
            self.val_set = val_set
            print(collections.Counter(self.val_set.label))
            self.train_set.dataset = self.normalizer.fit_transform(self.train_set.dataset.reshape(n, l * p)).reshape(n,
                                                                                                                     l,
                                                                                                                     p)
            n, l, p = self.val_set.dataset.shape
            self.val_set.dataset = self.normalizer.transform(self.val_set.dataset.reshape(n, l * p)).reshape(n, l, p)
            self.val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False,
                                          num_workers=self.args.num_workers, pin_memory=self.args.pin_mem,
                                          drop_last=True)
        if test_set is not None:
            self.test_set = test_set
            n, l, p = self.test_set.dataset.shape
            self.test_set.dataset = self.normalizer.transform(self.test_set.dataset.reshape(n, l * p)).reshape(n, l, p)
            self.test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False,
                                          num_workers=self.args.num_workers, pin_memory=self.args.pin_mem,
                                          drop_last=True)
        self.train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True,
                                       num_workers=self.args.num_workers, pin_memory=self.args.pin_mem, drop_last=True)
        self.total_batch_size = self.batch_size * self.args.update_freq
        self.num_training_steps_per_epoch = len(train_set) // self.total_batch_size
        self.scheduler = self.get_scheduler()

    def restore_data_range(self):
        self.train_set.dataset.dataset = self.normalizer.inverse_transform(self.train_set.dataset.dataset)
        if self.val_set is not None:
            self.val_set.dataset.dataset = self.normalizer.inverse_transform(self.val_set.dataset.dataset)
        if self.test_set is not None:
            self.test_set.dataset.dataset = self.normalizer.inverse_transform(self.test_set.dataset.dataset)

    def set_model(self):
        self.batch_size = self.args.batch_size * self.args.num_patch
        self.lr = self.args.lr
        self.epochs = self.args.epochs
        self.num_classes = self.args.num_classes
        self.clip_grad = self.args.clip_grad
        self.checkpoint_path = self.args.checkpoint_path
        self.update_freq = self.args.update_freq
        self.device = torch.device(self.args.device if torch.cuda.is_available() else "cpu")
        self.network = self.get_network().to(self.device)
        self.optimizer = self.get_optimizer()
        self.criterion = self.get_criterion()
        self.n_parameters = sum(p.numel() for p in self.network.parameters() if p.requires_grad)

    def train_with_Kfold(self, logger, alldataset,testdata, k=5, test_after_one_epoch=False, ):
        logger.info('number of params:%d' % self.n_parameters)
        logger.info("LR = %.8f" % self.lr)
        logger.info("Batch size = %d" % self.total_batch_size)
        logger.info("Update frequent = %d" % self.args.update_freq)
        logger.info("Number of training examples = %d" % len(self.train_set))
        logger.info("Number of training training per epoch = %d" % self.num_training_steps_per_epoch)
        logger.info("criterion = %s" % str(self.criterion))
        logger.info("scheduler = %s" % str(self.scheduler))
        logger.info("Start training for %d Folds" % self.args.k)
        loss_list = []
        acc_list = []
        precision_list = []
        recall_list = []
        F1_list = []
        kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=self.args.seed)
        for i, (train_index, val_index) in enumerate(kfold.split(alldataset.dataset, alldataset.label.astype(float))):
            train_fold = deepcopy(alldataset)
            train_fold.dataset = train_fold.dataset[train_index]
            train_fold.label = train_fold.label[train_index]
            val_fold = deepcopy(alldataset)
            val_fold.dataset = val_fold.dataset[val_index]
            val_fold.label = val_fold.label[val_index]
            test_fold = deepcopy(testdata)

            self.set_dataset(train_fold, val_fold,test_fold)
            logger.info("Start training for %d epochs with Fold %d" % (self.epochs, i))

            # self.network.reset_parameters()

            self.network = self.get_network().to(self.device)
            self.optimizer = self.get_optimizer()
            self.scheduler = self.get_scheduler()

            start_time = time.time()
            for epoch in range(0, self.epochs):
                train_stats = train_one_epoch(model=self.network, criterion=self.criterion,
                                              data_loader=self.train_loader,
                                              optimizer=self.optimizer, device=self.device, epoch=epoch,
                                              lr_schedule=self.scheduler, logger=logger,
                                              clip_grad=self.clip_grad,
                                              num_training_steps_per_epoch=self.num_training_steps_per_epoch,
                                              update_freq=self.update_freq,
                                              epoch2 = epoch)
                if epoch >= 0:
                    val_stats = self.val(logger)
                if epoch == self.epochs - 1:
                    self.save_params(checkpoint_path='./checkpoint_path', name=self.args.special)
            # self.restore_data_range()
            val_stats = self.val(logger)
            # val_stats = self.test(logger)
            loss_list.append(val_stats['avg loss'])
            acc_list.append(val_stats['acc'])
            precision_list.append(val_stats['precision'])
            recall_list.append(val_stats['recall'])
            F1_list.append(val_stats['F1'])

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            logger.info('Training time {}'.format(total_time_str))

        logger.info("Final evaluation on the test set.")
        test_stats = self.val(logger)
        logger.info("Test Stats: %s" % str(test_stats))
        
        return acc_list, precision_list, recall_list, F1_list

    def val(self, logger):
        start_time = time.time()
        val_stats = evaluate(model=self.network, data_loader=self.val_loader, device=self.device, logger=logger,
                              method=self.args.dataset, num_class=self.args.num_classes)
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info('Testing time {}'.format(total_time_str))
        return val_stats
    
    def test(self, logger):
        start_time = time.time()
        test_stats = evaluate(model=self.network, data_loader=self.test_loader, device=self.device, logger=logger,
                              method=self.args.dataset, num_class=self.args.num_classes)
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info('Testing time {}'.format(total_time_str))
        return test_stats

    def get_network(self):
        self.network_name = self.args.model
        if self.args.model == 'MSTAEEG':
            return MSTA(num_classes=self.num_classes)

    def get_optimizer(self):
        opt_args = dict(lr=self.args.lr, weight_decay=self.args.weight_decay)
        if self.args.opt == 'sgd':
            return torch.optim.SGD(self.network.parameters(), momentum=self.args.momentum, **opt_args)
        elif self.args.opt == 'adam':
            return torch.optim.Adam(self.network.parameters(), betas=self.args.opt_betas, **opt_args)
        elif self.args.opt == 'adamw':
            return torch.optim.AdamW(self.network.parameters(), betas=self.args.opt_betas, **opt_args)
        else:
            raise Exception('Optimizer Not Implementation!')

    def get_criterion(self):
        if self.args.criterion == 'l1':
            return torch.nn.L1Loss()
        elif self.args.criterion == 'smooth l1':
            return torch.nn.SmoothL1Loss()
        elif self.args.criterion == 'mse':
            return torch.nn.MSELoss()
        elif self.args.criterion == 'cross entropy':
            return torch.nn.CrossEntropyLoss()
        elif self.args.criterion == 'nll':
            return torch.nn.NLLLoss()
        else:
            raise Exception('Criterion Not Implementation!')

    def get_scheduler(self):
        if self.args.scheduler == 'CosineAnnealingLR':
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max= \
                self.epochs, eta_min=self.args.min_lr)
        elif self.args.schduler == 'StepLR':
            return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        elif self.args.scheduler == 'ExponentialLR':
            return torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
        else:
            return None

    def save_params(self, name='', checkpoint_path=None):
        if not isinstance(checkpoint_path, str):
            checkpoint_path = self.checkpoint_path
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        save_path = "{}/{}_{}.pth".format(checkpoint_path, self.network_name, name)
        torch.save(self.network.state_dict(), save_path)
        print("Save network parameters to {}".format(save_path))

    def load_params(self, name='', checkpoint_path=None):
        if not isinstance(checkpoint_path, str):
            checkpoint_path = self.checkpoint_path
            load_path = "{}/{}_{}.pth".format(checkpoint_path, self.network_name, name)
        else:
            load_path = checkpoint_path
        self.network.load_state_dict(torch.load(load_path, map_location=self.device))
        print("Load network parameters from {}".format(load_path))

    def load_feature_params_only(self, checkpoint_path):
        params = torch.load(checkpoint_path, map_location=self.device)
        remove_keys = []
        for key in params.keys():
            if "head" in key:
                print('removed head!')
                remove_keys.append(key)
        for key in remove_keys:
            params.__delitem__(key)
        self.network.load_state_dict(params, strict=False)
        print("Load network features parameters only from {}".format(checkpoint_path))




