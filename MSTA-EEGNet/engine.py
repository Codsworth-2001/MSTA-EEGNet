import math
from typing import Iterable
import torch
from sklearn.metrics import precision_score, recall_score, f1_score


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, lr_schedule=None, logger=None, clip_grad=None,
                    num_training_steps_per_epoch=None, update_freq=None, print_freq=10,epoch2=0):
    model.train(True)
    optimizer.zero_grad()
    header = 'Epoch: [{}]   '.format(epoch)
    num_batches = len(data_loader)
    num_samples = 0
    total_correct_samples = 0
    loss_sum = 0
    index = 0
    for data_iter_step, (samples, targets) in enumerate(data_loader):
        step = data_iter_step // update_freq  # update step
        if step >= num_training_steps_per_epoch:
            continue
        num_samples += samples.shape[0]
        samples = samples.to(device)
        targets = targets.to(device)
        tau = 0.95**epoch
        hard = False
        if tau<=0.2:
            hard = True
        re, output, index = model(samples,tau,hard)
        loss = criterion(output, targets) * 1
        loss_value = loss.item()
        loss_sum += loss_value

        correct_samples = acc_count(output, targets)
        total_correct_samples += correct_samples

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)

        loss /= update_freq
        loss.backward()
        if (data_iter_step + 1) % update_freq == 0:
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            optimizer.zero_grad()
        curren_lr = optimizer.param_groups[0]['lr']
        if data_iter_step == 0 or (data_iter_step + 1) % print_freq == 0:
            tail = '[{}/{}]   lr:{}   loss:{}'.format(data_iter_step, num_batches, curren_lr, round(loss_value, 6))
            logger.info(header + tail)
        torch.cuda.synchronize()
    if lr_schedule is not None:
        lr_schedule.step()
    logger.info('average stat: loss:{}   acc:{}   index:{}'.format(round(loss_sum / (data_iter_step + 1), 6),
                                                                   round(total_correct_samples / num_samples, 6),
                                                                   index))
    return {'avg loss': loss_sum / (data_iter_step + 1), 'acc': total_correct_samples / num_samples}


@torch.no_grad()
def evaluate(model, data_loader, device, logger, print_freq=10, num_class=10, ):
    criterion = torch.nn.CrossEntropyLoss()
    header = 'Test: '
    model.eval()
    num_samples = 0
    total_correct_samples = 0
    num_batches = len(data_loader)
    loss_sum = 0
    total_target = []
    total_predict = []
    index = 0

    for data_iter_step, (images, target) in enumerate(data_loader):

        images = images.to(device)
        target = target.to(device)

        re, output, index = model(images)

        loss = criterion(output, target) * 1
        loss_value = loss.item()
        loss_sum += loss_value

        correct_samples = acc_count(output, target)
        total_correct_samples += correct_samples
        batches = images.shape[0]
        num_samples += batches
        predict = torch.argmax(output, dim=1)
        total_target.extend(target.detach().tolist())
        total_predict.extend(predict.detach().tolist())
        if data_iter_step == 0 or (data_iter_step + 1) % print_freq == 0:
            tail = '[{}/{}]   loss:{}   acc:{}'.format(data_iter_step, num_batches, round(loss.item(), 6),
                                                       round(correct_samples / batches, 6))
            logger.info(header + tail)

    precision = precision_score(total_target, total_predict, labels=[i for i in range(num_class)], average='weighted')
    recall = recall_score(total_target, total_predict, labels=[i for i in range(num_class)], average='weighted')
    f1 = f1_score(total_target, total_predict, labels=[i for i in range(num_class)], average='weighted')

    logger.info('average stat: loss:{}   acc:{}'.format(round(loss_sum / (data_iter_step + 1), 6),
                                                        round(total_correct_samples / num_samples, 6)))
    return {'avg loss': loss_sum / (data_iter_step + 1), 'acc': total_correct_samples / num_samples, 'recall': recall,
            'precision': precision, 'F1': f1}


def acc_count(output, labels):
    predicts = torch.argmax(output, dim=1)
    count = torch.sum(predicts == labels)
    count = count.cpu().item()
    return count






