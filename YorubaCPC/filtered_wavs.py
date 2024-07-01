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
from tqdm import tqdm
from multiprocessing import cpu_count
import multiprocessing
import time

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
args, image_base = modelSetup(command_line_args, make_model_files=False)

args['device'] = 'cpu'

with open(args['data_train_unfiltered'], 'r') as fp:
    data = json.load(fp)

print(f'\rRead in data paths from:')
printDirectory(args['data_train_unfiltered'])
print(f'\n\rRead in {len(data)} data points')

n_sample_frames = args["cpc"]["n_sample_frames"]
num_files = cpu_count() - 1

def validLength(data_fn):
    data_point = np.load(data_fn + ".npz")
    if data_point["audio_feat"].shape[1] >= n_sample_frames: return data_fn
    else: return -1

starttime = time.time()
with multiprocessing.Pool(num_files) as pool:
    filtered_fns = pool.map(validLength, data)
print('That took {} seconds'.format(time.time() - starttime))

json_fn = args['data_train']
new_filtered = [fn for fn in filtered_fns if fn != -1]
with open(json_fn, "w") as json_file: json.dump(new_filtered, json_file, indent="")
print(f'Wrote {len(new_filtered)} data points to {json_fn}.')