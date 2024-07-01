#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________

import time
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import json
from .util import *
from evaluation.calculations import *
from losses import CPCLoss
from tqdm import tqdm
import sys
import warnings
warnings.filterwarnings("ignore")

optimizers = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam
}

def train(
    acoustic_model, train_loader, args):
    # function adapted from https://github.com/dharwath

    writer = SummaryWriter(args["exp_dir"] / "tensorboard")
    device = torch.device(args["device"] if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    torch.set_grad_enabled(True)
    loss_tracker = valueTracking()
    info = {}
    best_epoch, best_acc = 0, 0
    global_step, start_epoch = 0, 0
    start_time = time.time()

    acoustic_model = acoustic_model.to(device)

    cpc_loss = CPCLoss(args)
    # cpc_loss = nn.DataParallel(cpc_loss) if not isinstance(cpc_loss, torch.nn.DataParallel) and args["device"] == 'cuda' else cpc_loss
    cpc_loss = cpc_loss.to(device)

    trainable_parameters = getParameters([acoustic_model, cpc_loss])

    if args["optimizer"] == 'sgd':
        optimizer = torch.optim.SGD(
            trainable_parameters, args["learning_rate_scheduler"]["initial_learning_rate"],
            momentum=args["momentum"], weight_decay=args["weight_decay"]
            )
    elif args["optimizer"] == 'adam':
        optimizer = torch.optim.Adam(
            trainable_parameters, args["learning_rate_scheduler"]["initial_learning_rate"]
            )
    else:
        raise ValueError('Optimizer %s is not supported' % args["optimizer"])

    # [acoustic_model, cpc_loss], optimizer = amp.initialize(
    #     [acoustic_model, cpc_loss], optimizer, opt_level='O1'
    #     )
    acoustic_model = nn.DataParallel(acoustic_model)# if not isinstance(acoustic_model, torch.nn.DataParallel) and args["device"] == 'cuda' else acoustic_model
    

    scheduler = WarmupScheduler(
        optimizer,
        warmup_epochs=args["learning_rate_scheduler"]["warmup_epochs"],
        initial_lr=args["learning_rate_scheduler"]["initial_learning_rate"],
        max_lr=args["learning_rate_scheduler"]["max_lr"],
        milestones=args["learning_rate_scheduler"]["milestones"],
        gamma=args["learning_rate_scheduler"]["gamma"])

    if args["resume"]:
        if "restore_epoch" in args:
            info, start_epoch, global_step, best_epoch, best_acc = loadModelAttriburesAndTrainingAtEpoch(
                args["exp_dir"], acoustic_model, cpc_loss, optimizer, scheduler, device, args["restore_epoch"]
                )
            print(f'\nEpoch particulars:\n\t\tepoch = {start_epoch}\n\t\tglobal_step = {global_step}\n\t\tbest_epoch = {best_epoch}\n\t\tbest_acc = {best_acc}\n')
        else:
            info, start_epoch, global_step, best_epoch, best_acc = loadModelAttriburesAndTraining(
                args["exp_dir"], acoustic_model, cpc_loss, optimizer, scheduler, device
                )
            print(f'\nEpoch particulars:\n\t\tepoch = {start_epoch}\n\t\tglobal_step = {global_step}\n\t\tbest_epoch = {best_epoch}\n\t\tbest_acc = {best_acc}\n')

    start_epoch += 1
    
    for epoch in np.arange(start_epoch, args["n_epochs"] + 1):

        # current_learning_rate = adjust_learning_rate(args, optimizer, epoch)
        current_learning_rate = args["learning_rate_scheduler"]["initial_learning_rate"]
        average_accuracies = np.zeros(args["cpc"]["n_prediction_steps"])

        acoustic_model.train()
        cpc_loss.train()

        loss_tracker.new_epoch()
        start_time = time.time()
        printEpoch(epoch, 0, len(train_loader), loss_tracker, best_acc, start_time, start_time, current_learning_rate)
        for i, (mel, nframes) in enumerate(train_loader):

            optimizer.zero_grad()
            mel = mel.view(
                args["cpc"]["n_speakers_per_batch"] *
                args["cpc"]["n_utterances_per_speaker"],
                args["audio_config"]["num_mel_bins"], -1)
            z, c = acoustic_model(mel.to(device))

            # if args["language"] == "English":
               
            #     english_input = english_input.to(device)
            #     z, c = acoustic_model(english_input)
            #     nframes = english_nframes

            # elif args["language"] == "Hindi":
                
            #     hindi_input = hindi_input.to(device)
            #     z, c = acoustic_model(hindi_input)
            #     nframes = hindi_nframes

            # elif args["language"] == "English+Hindi":
   
            #     english_input = english_input.to(device)
            #     z, c = acoustic_model(english_input)
            #     nframes = english_nframes

            #     hindi_input = hindi_input.to(device)
            #     hindi_z, hindi_c = acoustic_model(hindi_input)

            #     z = torch.cat((z, hindi_z), dim=0)
            #     c = torch.cat((c, hindi_c), dim=0)
            #     nframes = torch.cat((nframes, hindi_nframes), dim=0)
            # print(z.size(), c.size(), "\n")
            loss, accuracies = cpc_loss(z, c, nframes)

            loss.backward()
            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
            optimizer.step()

            loss_tracker.update(loss.item(), mel.size(0)) #####
            end_time = time.time()
            printEpoch(epoch, i+1, len(train_loader), loss_tracker, best_acc, start_time, end_time, current_learning_rate)
            if np.isnan(loss_tracker.average):
                print("training diverged...")
                return
            global_step += 1  

            average_accuracies += (np.array(accuracies) - average_accuracies) / (i + 1)
         

        heading = [" "]        
        for entry in list(np.arange(args["cpc"]["n_prediction_steps"]) + 1):
            heading.append(str(entry))

        tablePrinting(
            heading, ["Prediction accuracy"],
            np.expand_dims(average_accuracies, axis=0)*100
            )
        avg_acc = np.mean(average_accuracies)
        print(f'Average accuracy = {avg_acc*100} %\n')
        scheduler.step()

        # val_accuracies = validation(acoustic_model, cpc_loss, val_loader, args)
        # tablePrinting(
        #     heading, ["Prediction accuracy"],
        #     np.expand_dims(val_accuracies, axis=0)*100
        #     )
        # avg_acc = np.mean(val_accuracies)
        # print(f'Average accuracy = {avg_acc*100} %')

        end_time = time.time()

        writer.add_scalar("loss/train", loss_tracker.average, epoch)
        writer.add_scalar("loss/val", avg_acc, epoch)

        best_acc, best_epoch = saveModelAttriburesAndTraining(
            args["exp_dir"], acoustic_model, cpc_loss, optimizer, scheduler,
            info, int(epoch), global_step, best_epoch, avg_acc, best_acc, loss_tracker.average, end_time-start_time)   

def validation(acoustic_model, cpc_loss, val_loader, args):
    # Validation
    average_accuracies = np.zeros(args["cpc"]["n_prediction_steps"])
    acoustic_model.eval()
    cpc_loss.eval()
    device = torch.device(args["device"] if torch.cuda.is_available() else "cpu")

    for i, (mel, nframes) in enumerate(val_loader):
        # english_input = english_input.to(device)
        # z, c = acoustic_model(english_input)
        # hindi_input = hindi_input.to(device)
        # hindi_z, hindi_c = acoustic_model(hindi_input)
        z, c = acoustic_model(mel.to(device))

        # z = torch.cat((z, hindi_z), dim=0)
        # c = torch.cat((c, hindi_c), dim=0)
        _, accuracies = cpc_loss(z, c, nframes)
        average_accuracies += (np.array(accuracies) - average_accuracies) / (i + 1)

    return average_accuracies