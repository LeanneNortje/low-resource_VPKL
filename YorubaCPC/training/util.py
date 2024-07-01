#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________

import json
import torch
# from alive_progress import alive_bar
import os
from os import popen
from math import ceil
from itertools import chain
import torch.optim as optim
from collections import Counter
from pathlib import Path
import warnings

terminal_rows, terminal_width = popen('stty size', 'r').read().split()
terminal_width = int(terminal_width)


def getParameters(models):
    valid_models = [model.parameters() for model in models if model is not None]
    return chain.from_iterable(valid_models)

def saveModelAttriburesAndTraining(
    exp_dir, acoustic_model, cpc_loss, optimizer, scheduler, info, epoch, global_step, best_epoch, acc, best_acc, loss, 
    epoch_time):
    
    overwrite_best_ckpt = False
    if acc > best_acc:
        best_epoch = epoch
        best_acc = acc
        overwrite_best_ckpt = True

    assert int(epoch) not in info
    info[int(epoch)] = {
        "global_step": global_step,
        "best_epoch": best_epoch,
        "acc": acc,
        "best_acc": best_acc,
        "loss": loss,
        "epoch_time": epoch_time
    }
    with open(exp_dir / "training_metadata.json", "w") as f:
        json.dump(info, f)

    checkpoint = {
        "acoustic_model": acoustic_model.state_dict(),
        "cpc_loss": cpc_loss.state_dict(),
        # "amp": amp.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_epoch": best_epoch,
        "acc": acc,
        "best_acc": best_acc,
        "loss": loss,
        "epoch_time": epoch_time
    }

    if not os.path.isdir(exp_dir / "models"): os.makedirs(exp_dir / "models")
    torch.save(checkpoint, exp_dir / "models" / "last_ckpt.pt")
    if overwrite_best_ckpt:
        torch.save(checkpoint, exp_dir / "models" / "best_ckpt.pt")
    if epoch % 100 == 0: torch.save(checkpoint, exp_dir / "models" / f'epoch_{epoch}.pt')

    return best_acc, best_epoch

def loadModelAttriburesAndTraining(
    exp_dir, acoustic_model, cpc_loss, optimizer, scheduler, device, last_not_best=True):

    info_fn = exp_dir / "training_metadata.json"
    with open(info_fn, "r") as f:
        info = json.load(f)

    if last_not_best:
        checkpoint_fn = exp_dir / "models" / "last_ckpt.pt"
    else:
        checkpoint_fn = exp_dir / "models" / "best_ckpt.pt"

    checkpoint = torch.load(checkpoint_fn, map_location=device)
    
    print(checkpoint["acoustic_model"].keys())
    print(isinstance(acoustic_model, torch.nn.DataParallel))
    acoustic_model.load_state_dict(checkpoint["acoustic_model"])
    print(checkpoint["cpc_loss"].keys())
    cpc_loss.load_state_dict(checkpoint["cpc_loss"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    # amp.load_state_dict(checkpoint["amp"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    epoch = checkpoint["epoch"]
    global_step = checkpoint["global_step"]
    best_epoch = checkpoint["best_epoch"]
    best_acc = checkpoint["best_acc"]  
    print(f'\nLoading model parameters from:\n\t\t{checkpoint_fn}')

    return info, epoch, global_step, best_epoch, best_acc


def loadModelAttriburesAndTrainingAtEpoch(
    exp_dir, acoustic_model, cpc_loss, optimizer, scheduler, device, load_epoch):

    info_fn = exp_dir / "training_metadata.json"
    with open(info_fn, "r") as f:
        info = json.load(f)

    checkpoint_fn = exp_dir / "models" / f'epoch_{load_epoch}.pt'

    checkpoint = torch.load(checkpoint_fn, map_location=device)

    acoustic_model.load_state_dict(checkpoint["acoustic_model"])
    cpc_loss.load_state_dict(checkpoint["cpc_loss"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    # amp.load_state_dict(checkpoint["amp"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    epoch = checkpoint["epoch"]
    global_step = checkpoint["global_step"]
    best_epoch = checkpoint["best_epoch"]
    best_acc = checkpoint["best_acc"]  
    print(f'\nLoading model parameters from:\n\t\t{checkpoint_fn}')

    return info, epoch, global_step, best_epoch, best_acc    


class valueTracking(object):
    # function adapted from https://github.com/dharwath

    def __init__(self):
        self.average = 0
        self.sum = 0
        self.num_values = 0
        self.epoch_average = 0
        self.epoch_sum = 0
        self.num_epoch_values = 0

    def update(self, value, n=1):

        self.sum += value * n
        self.num_values += n
        self.average = self.sum / self.num_values

        self.epoch_sum += value * n
        self.num_epoch_values += n
        self.epoch_average = self.epoch_sum / self.num_epoch_values

    def new_epoch(self):
        self.epoch_average = 0
        self.epoch_sum = 0
        self.num_epoch_values = 0


def floatFormat(number):
    return f'{number:.6f}' 

def timeFormat(start_time, end_time):   

    total_time = end_time-start_time

    days = total_time // (24 * 60 * 60) 
    total_time = total_time % (24 * 60 * 60)

    hours = total_time // (60 * 60)
    total_time = total_time % (60 * 60)

    minutes = total_time // 60
    
    seconds =total_time % (60)

    return int(days), int(hours), int(minutes), int(seconds)

def getCharacter(remaining, num_steps_in_single_width):
    if remaining >= 0 and remaining < num_steps_in_single_width*1/7: return "\u258F"
    elif remaining >= num_steps_in_single_width*1/7 and remaining < num_steps_in_single_width*2/7: return "\u258E"
    elif remaining >= num_steps_in_single_width*2/7 and remaining < num_steps_in_single_width*3/7: return "\u258D"
    elif remaining >= num_steps_in_single_width*3/7 and remaining < num_steps_in_single_width*4/7: return "\u258C"
    elif remaining >= num_steps_in_single_width*4/7 and remaining < num_steps_in_single_width*5/7: return "\u258B"
    elif remaining >= num_steps_in_single_width*5/7 and remaining < num_steps_in_single_width*6/7: return "\u258A"
    elif remaining >= num_steps_in_single_width*6/7 and remaining < num_steps_in_single_width: return "\u2589"

def printEpoch(epoch, step, num_steps, loss_tracker, best_acc, start_time, end_time, lr):
    
    terminal_width = tuple(os.get_terminal_size())
    terminal_width = terminal_width[0]

    days, hours, minutes, seconds = timeFormat(start_time, end_time)

    column_separator = ' | '
    epoch_string = f'Epoch: {epoch:<4} |'
    epoch_info_string = f'Loss: ' + floatFormat (loss_tracker.epoch_average) + column_separator 
    epoch_info_string +=  f'Average loss: ' + floatFormat(loss_tracker.average) + column_separator
    epoch_info_string += f'Best accuracy: ' + floatFormat(best_acc) + column_separator
    epoch_info_string += f'LR: ' + floatFormat(lr) + column_separator
    epoch_info_string += f'Epoch time: {hours:>2} hours  {minutes:>2} minutes  {seconds:>2} seconds'
    step_count_string = f'| [{step:>{len(str(num_steps))}}/{num_steps}]' 

    animation_width = len(epoch_string) + len(epoch_info_string) + len(column_separator) + len(step_count_string) + 1
    if animation_width >= terminal_width:
        # print(
        #     epoch_string + f'{int((step/num_steps)*100)}%' + step_count_string + column_separator + epoch_info_string, end=end_character
        #     )
        animation_width = terminal_width - len(step_count_string) - len(epoch_string) - 1
        num_steps_in_single_width = (num_steps) / animation_width #num_steps // animation_width if num_steps >= animation_width else num_steps / animation_width
        animation_progress = "\u2588"*ceil(step / num_steps_in_single_width)
        remaining = float(step % num_steps_in_single_width)
        animation_progress += getCharacter(remaining, num_steps_in_single_width)
        animation_blank = "-"*int(animation_width - len(animation_progress))

        if len(epoch_info_string) >= terminal_width:
            rewind = "\033[A"*3 if step != 0 else ""
            parts = epoch_info_string.split(column_separator)
            print(rewind + epoch_string + animation_progress + animation_blank + step_count_string)
            print(column_separator.join(parts[0:3]))
            print(column_separator.join(parts[3:]))
        else:
            rewind = "\033[A"*2 if step != 0 else ""
            print(
                rewind + epoch_string + animation_progress + animation_blank + step_count_string
                )
            print(epoch_info_string)

    else:
        rewind = "\033[A" if step != 0 else ""
        animation_width = terminal_width - animation_width
        num_steps_in_single_width = num_steps / animation_width #num_steps // animation_width if num_steps >= animation_width else num_steps / animation_width
        animation_progress = "\u2588"*ceil(step / num_steps_in_single_width)
        remaining = float(step % num_steps_in_single_width)
        animation_progress += getCharacter(remaining, num_steps_in_single_width)
        animation_blank = "-"*int(animation_width - len(animation_progress))
        print(
            rewind + epoch_string + animation_progress + animation_blank + step_count_string + column_separator + epoch_info_string
            )

def tablePrinting(headings, row_headings, values):

    assert(len(headings) - 1 == values.shape[-1])
    assert(len(row_headings) == values.shape[0])

    column_width = 10

    max_with = 0
    for entry in row_headings:
        if len(entry) > 0: max_with = len(entry)

    heading = f''
    for i, a_heading in enumerate(headings):
        heading += f'{a_heading:<{column_width}}' if i != 0 else f'{a_heading:<{max_with}}'
        if i != len(headings) - 1: heading += ' | '
    else: heading += '   '

    print("\t" + heading, flush=True)
    print(f'\t{"-"*len(heading)}', flush=True)

    for i in range(len(values)):
        row = f'\t{row_headings[i]:<{column_width}}'
        for j in range(values.shape[-1]):
            value = floatFormat(values[i, j])
            row += f' | {value:>{column_width}}'
        print(row, flush=True)

def adjust_learning_rate(args, optimizer, epoch):
    # epoch -= 1
    # base_lr = args["learning_rate_scheduler"]["initial_learning_rate"]
    # lr_decay = args["learning_rate_scheduler"]["learning_rate_decay"]
    # lr_decay_multiplier = args["learning_rate_scheduler"]["learning_rate_decay_factor"]
    # reset_at_epoch = args["learning_rate_scheduler"]["reset_at_epoch"]
    
    # if epoch >= reset_at_epoch: ep = epoch - reset_at_epoch
    # else: ep = epoch
    # lr = base_lr * (lr_decay_multiplier ** (ep // lr_decay))
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr
    # return lr

    epoch -= 1
    pretrain_n_epoch = args["learning_rate_scheduler"]["pretrain_epochs"]
    reset_at_epoch = args["learning_rate_scheduler"]["reset_at_epoch"]

    i_lr = ((epoch - pretrain_n_epoch) // reset_at_epoch) % (len(args["learning_rate_scheduler"]["learning_rates"]) - 1) + 1
    lr = args["learning_rate_scheduler"]["learning_rates"][i_lr if epoch >= pretrain_n_epoch else 0]
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def NFrames(audio_input, audio_output, nframes, with_torch=True):
    pooling_ratio = round(audio_input.size(-1) / audio_output.size(-1))
    if with_torch: pooling_ratio = torch.tensor(pooling_ratio, dtype=torch.int32)
    nframes = nframes.float()
    nframes.div_(pooling_ratio)
    nframes = nframes.int()
    zeros = (nframes == 0).nonzero()
    if zeros.nelement() != 0: nframes[zeros[:, 0]] += 1

    return nframes

class WarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, initial_lr, max_lr, milestones, gamma=0.1, last_epoch=-1):
        assert warmup_epochs < milestones[0]
        self.warmup_epochs = warmup_epochs
        self.milestones = Counter(milestones)
        self.gamma = gamma

        initial_lrs = self._format_param("initial_lr", optimizer, initial_lr)
        max_lrs = self._format_param("max_lr", optimizer, max_lr)
        if last_epoch == -1:
            for idx, group in enumerate(optimizer.param_groups):
                group["initial_lr"] = initial_lrs[idx]
                group["max_lr"] = max_lrs[idx]

        super(WarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", DeprecationWarning)

        if self.last_epoch <= self.warmup_epochs:
            pct = self.last_epoch / self.warmup_epochs
            return [
                (group["max_lr"] - group["initial_lr"]) * pct + group["initial_lr"]
                for group in self.optimizer.param_groups]
        else:
            if self.last_epoch not in self.milestones:
                return [group['lr'] for group in self.optimizer.param_groups]
            return [group['lr'] * self.gamma ** self.milestones[self.last_epoch]
                    for group in self.optimizer.param_groups]

    @staticmethod
    def _format_param(name, optimizer, param):
        """Return correctly formatted lr/momentum for each param group."""
        if isinstance(param, (list, tuple)):
            if len(param) != len(optimizer.param_groups):
                raise ValueError("expected {} values for {}, got {}".format(
                    len(optimizer.param_groups), name, len(param)))
            return param
        else:
            return [param] * len(optimizer.param_groups)

def loadPretrainedWeights(acoustic_model, CPC_loss, optimizer, scheduler, args):

    device = torch.device(args["device"] if torch.cuda.is_available() else "cpu")
    acoustic_model = torch.nn.DataParallel(acoustic_model)
    checkpoint_fn = Path("./pretrained_cpc/best_ckpt.pt")
    checkpoint = torch.load(checkpoint_fn, map_location=device)
    acoustic_model.load_state_dict(checkpoint["acoustic_model"])
    CPC_loss.load_state_dict(checkpoint["cpc_loss"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    # amp.load_state_dict(checkpoint["amp"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    epoch = checkpoint["epoch"]
    global_step = checkpoint["global_step"]
    best_epoch = checkpoint["best_epoch"]
    best_acc = checkpoint["best_acc"] 
    return epoch, global_step, best_epoch, best_acc, acoustic_model, CPC_loss, optimizer, scheduler
