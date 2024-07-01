#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2022
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________

import argparse
import torch
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import scipy
import scipy.signal
import librosa
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

sampled_data = np.load(Path('data/sampled_query_audio_data.npz'), allow_pickle=True)['data'].item()
# key = np.load(Path('data/label_key.npz'), allow_pickle=True)['id_to_word_key'].item()

base = Path('../Datasets/Flickr8k_text')
train_names = set()
for line in open(base / Path('Flickr_8k.trainImages.txt'), 'r'):
    train_names.add(line.split('.')[0])

val_names = set()
for line in open(base / Path('Flickr_8k.devImages.txt'), 'r'):
    val_names.add(line.split('.')[0])

train_audio = []
val_audio = []
id_lookup = {}
targets = {}

audio_dir = Path('data/english_wavs')
train = []
val = []

key = {}
id_to_word_key = {}
for i, l in enumerate(sampled_data):
    key[l] = i
    id_to_word_key[i] = l
    print(f'{i}: {l}')

np.savez_compressed(
    Path('data/label_key'),
    key=key,
    id_to_word_key=id_to_word_key
)

for word in sampled_data:
    id = key[word]
    for wav_name in sampled_data[word]:
        
        image_name = '_'.join(wav_name.split('_')[0:-1])

        if image_name in train_names:
            train.append((image_name, wav_name, word, id))
            train_audio.append(wav_name)
        elif image_name in val_names:
            val.append((image_name, wav_name, word, id))
            val_audio.append(wav_name)

counting = {}
unique = set()
duplicates = 0
for wav in (Path('data/support_set_english_audio')).rglob('*.wav'):
    temp = '_'.join(wav.stem.split('_')[:-1])
    image_name = '_'.join(temp.split('_')[0:-1])

    word = wav.stem.split('_')[-1]
    id = key[word]
    if word not in counting: counting[word] = 0
    counting[word] += 1

    train.append((image_name, temp, word, id))
    train_audio.append(temp)
    train_names.add(temp)

    val.append((image_name, temp, word, id))
    val_audio.append(temp)
    val_names.add(temp)
    
    if temp not in unique: unique.add(temp)
    else: duplicates += 1

total = 0
for word in counting:
    print(word, counting[word])
    total += counting[word]
print(total, len(unique), duplicates)

print(len(set(train_names)), len(set(train_audio)))
print(len(set(val_names)), len(set(val_audio)))
print(len(set(train_audio).intersection(val_audio)))
print(len(set(train_names).intersection(val_names)))

for image_name, wav_name, word, id in train:

    if id not in id_lookup: id_lookup[id] = {}
    if wav_name not in id_lookup: id_lookup[id][wav_name] = []
    fn = (audio_dir / wav_name).with_suffix('.wav')
    id_lookup[id][wav_name].append(fn)

neg_id_lookup = {}
for id in sorted(id_lookup):
    
    images_with_id = list(set([a for a in id_lookup[id]]))
    
    all_ids = list(id_lookup.keys())
    all_ids.remove(id)
    
    neg_id_lookup[id] = {}
    
    for neg_id in tqdm(all_ids, desc=f'ID: {id}'):
        temp = [i for name in id_lookup[neg_id] for i in id_lookup[neg_id][name] if name not in images_with_id]
        if len(temp) > 0:
            neg_id_lookup[id][neg_id] = temp

np.savez_compressed(
    Path("./data/train_lookup"), 
    lookup=id_lookup,
    neg_lookup=neg_id_lookup
)

for id in sorted(id_lookup):
    print(id, len(id_lookup[id]))

# for id in sorted(neg_id_lookup):
#     c = 0
#     for neg_id in neg_id_lookup[id]:
#         c += len(neg_id_lookup[id][neg_id])
#     print(id, c)

id_lookup = {}
for image_name, wav_name, word, id in val:

    if wav_name not in targets: targets[wav_name] = {'word': word, 'id': id}

    if id not in id_lookup: id_lookup[id] = {}
    if wav_name not in id_lookup: id_lookup[id][wav_name] = []
    fn = (audio_dir / wav_name).with_suffix('.wav')
    id_lookup[id][wav_name].append(fn)

neg_id_lookup = {}
for id in sorted(id_lookup):
    
    images_with_id = list(set([a for a in id_lookup[id]]))
    
    all_ids = list(id_lookup.keys())
    all_ids.remove(id)
    
    neg_id_lookup[id] = {}
    
    for neg_id in tqdm(all_ids, desc=f'ID: {id}'):
        temp = [i for name in id_lookup[neg_id] for i in id_lookup[neg_id][name] if name not in images_with_id]
        if len(temp) > 0:
            neg_id_lookup[id][neg_id] = temp

for id in sorted(id_lookup):
    print(id, len(id_lookup[id]))

# for id in sorted(neg_id_lookup):
#     c = 0
#     for neg_id in neg_id_lookup[id]:
#         c += len(neg_id_lookup[id][neg_id])
#     print(id, c)

np.savez_compressed(
    Path("./data/val_lookup"), 
    lookup=id_lookup,
    neg_lookup=neg_id_lookup
)