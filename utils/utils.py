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
            self.threshold = opts.plateau_threshold
        else:
            raise ValueError("Scheduler type {} not supported".format(opts.scheduler))
        print("Using scheduler " + str(opts.scheduler))
        self.scheduler_type = opts.scheduler

        self.validation_losses = []
        # key -> epoch of adjustment value -> new lr
        self.lr_history = dict()

    def _adjust_learning_rate_multistep(self, epoch):
        lr_new = self.initial_lr * (0.1 ** (sum(epoch >= np.array(self.lr_steps))))
        self._set_new_lr(epoch, lr_new)

    def _adjust_learning_rate_plateau(self, epoch, current_loss):
        start_index = len(self.validation_losses) - self.lr_patience
        if start_index < 0:
            self.validation_losses.append(current_loss)
            return

        should_adjust = True
        for i in range(start_index, len(self.validation_losses)):
            validation_loss = self.validation_losses[i]

            if (validation_loss + self.threshold) < current_loss:
                should_adjust = False
                break

        self.validation_losses.append(current_loss)

        if should_adjust:
            latest_lr = self._get_latest_lr()
            new_lr = self.lr_factor * latest_lr
            self._set_new_lr(epoch, new_lr)

    def adjust_learning_rate(self, epoch):
        if self.scheduler_type == "MultiStepLR":
            self._adjust_learning_rate_multistep(epoch)
        elif self.scheduler_type == "ReduceLROnPlateau":
            return
        else:
            raise ValueError("Unsupported scheduler")

    def adjust_learning_rate_validation(self, epoch, validation_loss):
        if self.scheduler_type == "MultiStepLR":
            return
        elif self.scheduler_type == "ReduceLROnPlateau":
            self._adjust_learning_rate_plateau(epoch, validation_loss)
        else:
            raise ValueError("Unsupported scheduler")

    def _get_latest_lr(self):
        if len(self.lr_history) == 0:
            return self.initial_lr
        else:
            latest_epoch = max(self.lr_history.keys())
            return self.lr_history[latest_epoch]

    def _set_new_lr(self, epoch, new_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        self.lr_history[epoch] = new_lr

    def state_dict(self):
        scheduler_state = {
            "validation_losses": self.validation_losses,
            "lr_history": self.lr_history
        }
        return scheduler_state

    def load_state_dict(self, state_dict):
        if "scheduler" not in state_dict:
            print("Scheduler state dict not found")
            return

        scheduler_state = state_dict["scheduler"]
        if "validation_losses" not in scheduler_state:
            print("Validation losses not found scheduler state dict")
        else:
            print("Loaded validation losses from scheduler state dict")
            self.validation_losses = scheduler_state["validation_losses"]

        if "lr_history" not in scheduler_state:
            print("Lr history not found scheduler state dict")
        else:
            print("Loaded lr history from scheduler state dict")
            self.lr_history = scheduler_state["lr_history"]

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


