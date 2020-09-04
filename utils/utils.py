import csv
import torch
import shutil
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TensorboardLogger:
    def __init__(self, output_dir):
        self.writer = SummaryWriter(output_dir)

    def log_iteration(self, values, iteration, subset):
        for key in values.keys():
            self.writer.add_scalar("{}/batch/{}".format(subset, key), values[key], global_step=iteration)

    def log_epoch(self, values, epoch, subset):
        for key in values.keys():
            self.writer.add_scalar("{}/epoch/{}".format(subset, key), values[key], global_step=epoch)


class Scheduler:
    def __init__(self, optimizer, opts):
        self.optimizer = optimizer
        self.initial_lr = opts.learning_rate
        self.lr_factor = opts.lr_factor
        if opts.scheduler == "MultiStepLR":
            self.lr_steps = opts.lr_steps
        elif opts.scheduler == "ReduceLROnPlateau":
            self.lr_patience = opts.lr_patience
        else:
            raise ValueError("Scheduler type {} not supported".format(opts.scheduler))
        print("Using scheduler " + str(opts.scheduler))
        self.scheduler_type = opts.scheduler

    def _adjust_learning_rate_multistep(self, epoch):
        lr_new = self.initial_lr * (0.1 ** (sum(epoch >= np.array(self.lr_steps))))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr_new

    def adjust_learning_rate(self, epoch):
        if self.scheduler_type == "MultiStepLR":
            self._adjust_learning_rate_multistep(epoch)
        # elif self.scheduler_type == "ReduceLROnPlateau":
        #     self._adjust_learning_rate_plateau()
        else:
            raise ValueError("Unsupported scheduler")

    # def adjust_epoch_begin(self):
    #     if self.scheduler_type == "MultiStepLR":
    #         self.scheduler.step()
    #
    # def adjust_epoch_end(self, val_loss):
    #     if self.scheduler_type == "ReduceLROnPlateau":
    #         self.scheduler.step(val_loss)
    #
    # def state_dict(self):
    #     return self.scheduler.state_dict()
    #
    # def load_state_dict(self, state_dict):
    #     self.scheduler.load_state_dict(state_dict)

class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def calculate_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(state, is_best, opt):
    torch.save(state, '%s/%s_checkpoint.pth' % (opt.result_path, opt.store_name))
    if is_best:
        shutil.copyfile('%s/%s_checkpoint.pth' % (opt.result_path, opt.store_name),'%s/%s_best.pth' % (opt.result_path, opt.store_name))


def adjust_learning_rate(optimizer, epoch, opt):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr_new = opt.learning_rate * (0.1 ** (sum(epoch >= np.array(opt.lr_steps))))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_new
        #param_group['lr'] = opt.learning_rate


