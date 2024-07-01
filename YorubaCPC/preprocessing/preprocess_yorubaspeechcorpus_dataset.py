#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________

import argparse
import os
import pickle
import sys
import time
import torch
import numpy as np
from tqdm import tqdm
from image_caption_dataset_preprocessing import AudioDatasetYoruba1
import json
from pathlib import Path
import numpy
from collections import Counter
import sys
from os import popen

terminal_rows, terminal_width = popen('stty size', 'r').read().split()
terminal_width = int(terminal_width)
def heading(string):
    print("_"*terminal_width + "\n")
    print("-"*10 + string + "-"*10 + "\n")

# parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument("--data-fn", default="/mnt/HDD/leanne_HDD", help="Path to images.")
# parser.add_argument("--dataset", type=str)
# parser.add_argument("--language", type=str)
# command_line_args = parser.parse_args()
language = 'yoruba'
dataset = 'yorubaspeechcorpus'
data_fn = Path('../../Datasets/yorubaspeechcorpus').absolute()
print(data_fn, data_fn.is_dir())

with open("yorubaspeechcorpus_preprocessing_config.json") as file:
  args = json.load(file)

# args["data_train"] = data_fn / args["data_train"]
out_dir = Path(args["out_dir"])
out_dir = out_dir / Path(dataset + "_" + language)

if not os.path.isdir((Path("..") / out_dir).absolute()):
    (Path("..") / out_dir).absolute().mkdir(parents=True)


# Load in json files
#_________________________________________________________________________________________________

def load(fn, split):
    
    data = list(Path(fn).rglob(f'{split}/**/*.wav'))
    print(f'{len(data)} data points in {fn}\n')
    return data

train_data = load(data_fn, "corpus")

def SaveDatapoins(dataloader, subset, datasets):

    data_paths = []
    for i, (audio_feat, name) in enumerate(tqdm(dataloader, leave=False)):

        save_fn = out_dir / str(name[0])

        if (audio_feat.numpy() == 0).all():
            print(f'Audio is zero: {name[0]}')
        else:
            if not (Path("..") / save_fn).absolute().parent.is_dir(): (Path("..") / save_fn).absolute().parent.mkdir(parents=True)
            if not save_fn.is_file():   
                numpy.savez_compressed(
                    (Path("..") / save_fn).absolute(), 
                    audio_feat=audio_feat.squeeze().numpy()
                    )
                data_paths.append(str(save_fn))       

    json_fn = (Path("..") / Path(args['out_dir'])).absolute() / Path(datasets + "_" + subset + ".json")      
    with open(json_fn, "w") as json_file: json.dump(data_paths, json_file, indent="")
    print(f'Wrote {len(data_paths)} data points to {json_fn}.')
    return json_fn

heading(f'Preprocessing training data points.')
train_loader = torch.utils.data.DataLoader(
    AudioDatasetYoruba1(
        train_data, args["audio_config"]),
    batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
train_json_fn = SaveDatapoins(train_loader, "train", dataset + "_" + language)