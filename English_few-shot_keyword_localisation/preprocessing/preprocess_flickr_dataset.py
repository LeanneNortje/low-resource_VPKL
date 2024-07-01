#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2022
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
from image_caption_dataset_preprocessing import flickrData
import json
from pathlib import Path
import numpy
from collections import Counter
import sys
from os import popen
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial

terminal_rows, terminal_width = popen('stty size', 'r').read().split()
terminal_width = int(terminal_width)
def heading(string):
    print("_"*terminal_width + "\n")
    print("-"*10 + string + "-"*10 + "\n")

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--image-base", default="../..", help="Path to images.")
command_line_args = parser.parse_args()

image_base = Path(command_line_args.image_base).absolute()

with open("preprocessing_flickr_config.json") as file:
  args = json.load(file)

args["data_train"] = Path('..') / Path(args["data_train"])
args["data_val"] = Path('..') / Path(args["data_val"])
args["audio_base"] = image_base / args["audio_base"]
args["image_base"] = image_base / args["image_base"]
args["out_dir"] = Path(args["out_dir"])

if not os.path.isdir((Path("..") / args["out_dir"]).absolute()):
    (Path("..") / args["out_dir"]).absolute().mkdir(parents=True)


# Load in txt files
#_________________________________________________________________________________________________
key = np.load(Path('../data/label_key.npz'), allow_pickle=True)['id_to_word_key'].item()
fn = Path('../../Datasets/flickr_audio/wav2spk.txt')
spearkers = {}
with open(fn, 'r') as f:
    for line in f:
        a, b = line.split()
        a = a.split('.')[0]
        if a not in spearkers: spearkers[a] = b

def load(fn):

    raw_data = np.load(fn, allow_pickle=True)['lookup'].item()
    data = []
    wav_to_id = {}
    wav_name_to_fn = {}
    for id in raw_data:
        for wav_name in raw_data[id]:
            # for wav in raw_data[id][wav_name]:
            if wav_name not in wav_to_id: wav_to_id[wav_name] = []
            wav_to_id[wav_name].append(id)
                # if wav_name not in wav_name_to_fn: 
                #     wav_name_to_fn[wav_name] = Path('..') / Path(wav)


    # print(targets[wav_name], wav_to_id[wav_name])

    for id in raw_data:
        word = key[id]
        
        for wav_name in raw_data[id]:

            for wav in raw_data[id][wav_name]:
                word_wavs = {}
                for sub_id in wav_to_id[wav_name]:
                    word_fn = (wav.parent / f'{wav.stem}_word_{key[sub_id]}').with_suffix('.wav')
                    word_wavs[sub_id] = word_fn
                    
                spkr = spearkers[wav_name]
                image = '_'.join(wav_name.split('_')[:-1])
                point = {
                    "speaker": spkr,
                    "wav_name": wav_name,
                    "wav": Path('..') / Path(wav),
                    "ids": wav_to_id[wav_name],
                    "image": image,
                    'word_wavs': word_wavs
                }
                data.append(point)
                

    print(f'{len(data)} data points in {fn}\n')
    return data

speaker_fn = Path(args['audio_base']) / Path('wav2spk.txt')
image2wav = {}
with open(speaker_fn, 'r') as file:
    for line in file:
        wav, speaker = line.strip().split()
        image_name = '_'.join(wav.split('_')[0:2]) + '.jpg'
        if image_name not in image2wav: image2wav[image_name] = []
        image2wav[image_name].append((wav, speaker))
    
train_data = load(args['data_train'])
val_data = load(args['data_val'])

def saveADatapoint(ids, save_fn, image_fn, audio_feat, word_feats, args):
    ids = list(set(ids))
    
    word_dict = {}
    for id in word_feats:
        fn = word_feats[id]['fn'][0]
        new_fn = save_fn.parent / Path(fn).stem
        numpy.savez_compressed(
            new_fn.absolute(), 
            audio_feat=word_feats[id]['feat'].squeeze().numpy()
            )
        word_dict[id] = str(Path(*new_fn.parts[1:]).with_suffix('.npz'))

    numpy.savez_compressed(
        save_fn.absolute(), 
        image=str(image_fn), 
        ids=ids,
        audio_feat=audio_feat.squeeze().numpy(),
        word_fns=word_dict
        )
    # print(list(np.load(str(save_fn) + '.npz', allow_pickle=True)['ids']))
    return "/".join(str((save_fn).with_suffix('.npz')).split("/")[1:])

def SaveDatapointsWithMasks(dataloader, subset, datasets):

    ouputs = []
    executor = ProcessPoolExecutor(max_workers=cpu_count()) 
    lengths = []
    # exclude = []
    
    save_fn = Path("..") / args["out_dir"] / Path(datasets)
    masks = Path("../data/flickr_image_masks")
    if not save_fn.absolute().is_dir(): 
        save_fn.absolute().mkdir(parents=True)
        print(f'Made {save_fn}.')

    for i, (image_fn, audio_feat, audio_name, image_name, speaker, ids, word_feats) in enumerate(tqdm(dataloader, leave=False)):

        this_save_fn = save_fn / str(audio_name[0].split(".")[0]) 

        # if audio_feat.size(-1) > args["audio_config"]["target_length"]: 
        #     print(str(audio_name[0].split(".")[0]), audio_feat.size())
        #     exclude.append("/".join(str((this_save_fn).with_suffix('.npz')).split("/")[1:]))
        #     continue
        
        ouputs.append(executor.submit(partial(saveADatapoint, ids[0], this_save_fn, image_fn[0], audio_feat, word_feats, args)))
        lengths.append(audio_feat.size(-1))
    data_paths = [entry.result() for entry in tqdm(ouputs)]
    json_fn = (Path("..") / args["out_dir"]).absolute() / Path(datasets + "_" + subset + ".json")      
    with open(json_fn, "w") as json_file: json.dump(data_paths, json_file, indent="")
    print(f'Wrote {len(data_paths)} data points to {json_fn}.')
    print(f'Min {np.min(lengths)}')
    print(f'Mean {np.mean(lengths)}')
    print(f'Max {np.max(lengths)}')
    
    return json_fn      

heading(f'Preprocessing training data points.')
train_loader = torch.utils.data.DataLoader(
    flickrData(train_data, args['out_dir'] / Path('wavs'), args['image_base'], args["audio_config"]),
    batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
train_json_fn = SaveDatapointsWithMasks(train_loader, "train", 'flickr')

heading(f'Preprocessing validation data points.')
# args["image_config"]["center_crop"] = True
val_loader = torch.utils.data.DataLoader(
    flickrData(val_data, args['out_dir'] / Path('wavs'), args['image_base'], args["audio_config"]),
    batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
val_json_fn = SaveDatapointsWithMasks(val_loader, "val", 'flickr')