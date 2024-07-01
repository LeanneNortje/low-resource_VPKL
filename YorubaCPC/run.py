#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
# adapted from https://github.com/dharwath

import argparse
import torch
from dataloaders import *
from training import train
from models.setup import *
from models.util import *

# Commandline arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--resume", action="store_true", dest="resume",
        help="load from exp_dir if True")
parser.add_argument("--config-file", type=str, default='multilingual', choices=['multilingual', 'multilingual+matchmap', 'english', 'english+matchmap', 'hindi', 'hindi+matchmap'],
        help="Model config file.")
parser.add_argument("--device", type=str, default="0", help="gpu_device")
parser.add_argument("--restore-epoch", type=int, default=-1, help="Epoch to resore training from.")
parser.add_argument("--image-base", default="..", help="Path to images.")
command_line_args = parser.parse_args()

# # Setting up model specifics
heading(f'\nSetting up model files ')
args, image_base = modelSetup(command_line_args)

# Create dataloaders
heading(f'\nLoading training data ')
train_loader = torch.utils.data.DataLoader(
    ImageAudioData('data_train', args),
    batch_size=args["cpc"]["n_speakers_per_batch"], shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

# heading(f'\nLoading validation data ')
# args["image_config"]["center_crop"] = True
# val_loader = torch.utils.data.DataLoader(
#     ImageAudioData('data_val', args),
#     batch_size=args["cpc"]["n_speakers_per_batch"], shuffle=False, num_workers=8, pin_memory=True)

# Create models
acousticModel(args)
acoustic_model = AcousticEncoder(args).to("cpu")
# acoustic_model = CPCEncoder(args['audio_model']['z_dim'], args['audio_model']['c_dim']).to("cpu")
# summary(acoustic_model, (40, 2048), device="cpu")#, depth=5)

# Train model
if args["resume"]: heading(f'\nResume training on {torch.cuda.get_device_name(args["device"]) if torch.cuda.is_available() else "CPU"} ')
else: heading(f'\nTraining on {torch.cuda.get_device_name(args["device"]) if torch.cuda.is_available() else "CPU"} ')
train(acoustic_model, train_loader, args)