import argparse
import logging, os
from net_model import NetModel
from dataset import MyDataset
import collections


def get_args_parser():
    parser = argparse.ArgumentParser('Image recognition training(and evaluation) script', add_help=False)
    parser.add_argument('--batch_size', type=int, help='batch size when the model receive data')
    parser.add_argument('--num_patch', type=int, help='num of patch')
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--update_freq', default=10, type=int, help='gradient accumulation steps')
    parser.add_argument('--model', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--channel', default=62, type=int, help='eeg channel')
    parser.add_argument('--dataset', type=str, help='Name of dataset')
    parser.add_argument('--dataset_test', type=str, help='Name of dataset for test')
    parser.add_argument('--special', default='', type=str, help='sth important')
    parser.add_argument('--input_value', default='', type=str, help='sth important')
    parser.add_argument('--input_size', default=0, type=int, help='input size or dim of the data feed into the model')
    parser.add_argument('--dropout', default=0.5, type=float,
                        help='Probabilities of the drop rate in the dropout layer')
    parser.add_argument('--num_classes', default=10, type=int, help='Number of classes')
    parser.add_argument('--opt', default='sgd', type=str, metavar='Optimizer', help='Optimizer (default: "sgd/adaw"')
    parser.add_argument('--criterion', default='cross entropy', type=str, metavar='criterion',
                        help='criterion (default: "cross entropy"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='Epsilon',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=[0.9, 0.999], type=list, nargs='+', metavar='Beta',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='Norm',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay (default: 0.05)')
    parser.add_argument('--scheduler', type=str, default='CosineAnnealingLR')
    parser.add_argument('--lr', type=float, default=5e-3, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    parser.add_argument('--kfold', type=bool, default=False, help='whether to use kfold')
    parser.add_argument('--k', type=int, default=10, help='number of k in kfold,valid only when kfold=True')
    parser.add_argument('--data_path', default='./cifar-10-batches-py', type=str,
                        help='dataset path')
    parser.add_argument('--sample_size', default=0, type=int,
                        help='sample size used to build the dataset')
    parser.add_argument('--checkpoint_path', default='your_path_here', type=str,
                        help='checkpoint path')
    parser.add_argument('--log_dir', default='your_path_here',
                        help='path where to save log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--pin_mem', type=bool, default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    return parser


def main(args):
    if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)
    log_filename = os.path.join(args.log_dir, str(args.model) + '.log')
    if os.path.exists(log_filename): os.remove(log_filename)
    logger = logging.getLogger()
    file_hanlder = logging.FileHandler(log_filename)
    console = logging.StreamHandler()

    file_hanlder.setLevel(logging.INFO)
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    file_hanlder.setFormatter(formatter)
    console.setFormatter(formatter)
    logger.addHandler(file_hanlder)
    logger.addHandler(console)
    logger.setLevel(logging.INFO)

    logger.info(str(args))
    accc = []

    alldataset = MyDataset(args.sample_size, args.num_patch, args.dataset, args.input_value, transform=None)
    testdataset = MyDataset(args.sample_size, args.num_patch, args.dataset_test, args.input_value, transform=None)
    print('get_data')
    print(collections.Counter(alldataset.label))
    model = NetModel(args)
    

    acc_list, precision_list, recall_list, F1_list = model.train_with_Kfold(logger, alldataset,testdataset, args.k, True)

    with open('result_' + args.dataset + '/result' + args.model + '.txt', 'a') as file:
        acc = sum(acc_list) / len(acc_list)
        accc.append(acc)

        recall = sum(recall_list) / len(recall_list)
        precision = sum(precision_list) / len(precision_list)
        F1 = sum(F1_list) / len(F1_list)

        file.write('acc: ')
        file.write(str(acc))
        file.write(' | ')
        for item in acc_list:
            file.write(str(item) + ' ')
        file.write('\n')
        file.write('precision:')
        file.write(str(precision))
        file.write(' | ')
        for item in precision_list:
            file.write(str(item) + ' ')
        file.write('\n')
        file.write('recall:')
        file.write(str(recall))
        file.write(' | ')
        for item in recall_list:
            file.write(str(item) + ' ')
        file.write('\n')
        file.write('F1:')
        file.write(str(F1))
        file.write(' | ')
        for item in F1_list:
            file.write(str(item) + ' ')
        file.write('\n')
        file.write(args.special)
        file.write(' ')
        file.write(args.input_value)
        file.write('\n')



# 即得到engine中修改loss相关
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Image recognition training(and evaluation) script', parents=[get_args_parser()])
    arglist = ['--model', 'MSTAEEG',
               '--input_size','1',
               '--dataset', '',
               '--dataset_test','',
               '--input_value', 'all',
               '--num_classes', '3',
               '--sample_size', '200',
               '--batch_size', '',
               '--epochs', '50',
               '--update_freq', '1',
               '--seed', '',
               '--kfold', 'True', '--k', '5',
               '--opt', 'adamw', '--opt_eps', '1e-8',
               '--clip_grad', '1',
               '--momentum', '0.9',
               '--weight_decay', '0.005',
               '--lr', '5e-4', '--min_lr', '1e-6', ]
    args = parser.parse_args(args=arglist)
    main(args)
