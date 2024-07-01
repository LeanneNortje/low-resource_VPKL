#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2022
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
# adapted from https://github.com/dharwath

import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from dataloaders import *
from models.setup import *
from models.util import *
from models.GeneralModels import *
from models.multimodalModels import *
from training.util import *
from evaluation.calculations import *
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from training import validate
import time
from tqdm import tqdm

import numpy as trainable_parameters
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import scipy
import scipy.signal
from scipy.spatial import distance
import librosa
import matplotlib.lines as lines

import itertools
import seaborn as sns
import textgrids

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--resume", action="store_true", dest="resume",
        help="load from exp_dir if True")
parser.add_argument("--config-file", type=str, default='multilingual+matchmap', choices=['multilingual', 'multilingual+matchmap'], help="Model config file.")
parser.add_argument("--restore-epoch", type=int, default=-1, help="Epoch to generate accuracies for.")
parser.add_argument("--image-base", default="/storage", help="Model config file.")
parser.add_argument("--path", type=str, help="Path to Flicker8k_Dataset.")
command_line_args = parser.parse_args()
restore_epoch = command_line_args.restore_epoch

def get_detection_metric_count(hyp_trn, gt_trn):
    # Get the number of true positive (n_tp), true positive + false positive (n_tp_fp) and true positive + false negative (n_tp_fn) for a one sample on the detection task
    correct_tokens = set([token for token in gt_trn if token in hyp_trn])
    n_tp = len(correct_tokens)
    n_tp_fp = len(hyp_trn)
    n_tp_fn = len(set(gt_trn))

    return n_tp, n_tp_fp, n_tp_fn

def eval_detection_prf(n_tp, n_tp_fp, n_tp_fn):
    precision = n_tp / n_tp_fp
    recall = n_tp / n_tp_fn
    fscore = 2 * precision * recall / (precision + recall)

    return precision, recall, fscore

def eval_detection_accuracy(hyp_loc, gt_loc):
    score = 0
    total = 0

    for gt_start_end_frame, gt_token in gt_loc:
    
        if gt_token in [hyp_token for _, hyp_token in hyp_loc]:
            score += 1
        total += 1

    return score, total

def get_localisation_metric_count(hyp_loc, gt_loc):
    # Get the number of true positive (n_tp), true positive + false positive (n_tp_fp) and true positive + false negative (n_tp_fn) for a one sample on the localisation task
    n_tp = 0
    n_fp = 0
    n_fn = 0

    for hyp_frame, hyp_token in hyp_loc:
        if hyp_token not in [gt_token for _, gt_token in gt_loc]:
            n_fp += 1

    for gt_start_end_frame, gt_token in gt_loc:
        if gt_token not in [hyp_token for _, hyp_token in hyp_loc]:
            n_fn += 1
            continue
        for hyp_frame, hyp_token in hyp_loc:
            if hyp_token == gt_token and (gt_start_end_frame[0] <= hyp_frame < gt_start_end_frame[1] or gt_start_end_frame[0] < hyp_frame <= gt_start_end_frame[1]):
                n_tp += 1
            elif hyp_token == gt_token and (hyp_frame < gt_start_end_frame[0] or gt_start_end_frame[1] < hyp_frame):
                n_fp += 1


    return n_tp, n_fp, n_fn

def eval_localisation_accuracy(hyp_loc, gt_loc):
    score = 0
    total = 0

    for gt_start_end_frame, gt_token in gt_loc:
        if gt_token not in [hyp_token for _, hyp_token in hyp_loc]:
            total += 1
    
        if gt_token in [hyp_token for _, hyp_token in hyp_loc]:
            total += 1
        
        for hyp_frame, hyp_token in hyp_loc:
            if hyp_token == gt_token and (gt_start_end_frame[0] <= hyp_frame < gt_start_end_frame[1] or gt_start_end_frame[0] < hyp_frame <= gt_start_end_frame[1]):
                score += 1

    return score, total

def eval_localisation_prf(n_tp, n_fp, n_fn):
    precision = n_tp / (n_tp + n_fp)
    recall = n_tp / (n_tp + n_fn)
    fscore = 2 * precision * recall / (precision + recall)

    return precision, recall, fscore

speaker = 'S001'
base_path = Path('../Datasets')
flickr_boundaries_fn = base_path / Path('flickr_audio/flickr_8k.ctm')
flickr_audio_dir = flickr_boundaries_fn.parent / "wavs"
flickr_images_fn = base_path / Path('Flicker8k_Dataset')
flickr_segs_fn = Path('./data/flickr_image_masks/')

key = np.load(Path('data/label_key.npz'), allow_pickle=True)['key'].item()
translation = np.load(Path('data/label_key.npz'), allow_pickle=True)['translation_e_to_y'].item()
translation_y_to_e = np.load(Path('data/label_key.npz'), allow_pickle=True)['translation_y_to_e'].item()
yoruba_vocab = list(key.keys())

vocab = []
keywords = []
with open('./data/34_keywords.txt', 'r') as f:
    for keyword in f:
        vocab.append(keyword.strip())
        keywords.append(keyword.strip())

translation = {}
translation_y_to_e = {}
# yoruba_vocab = []
with open(Path('../Datasets/yfacc_v6/Flickr8k_text/eng_yoruba_keywords.txt'), 'r') as f:
    for line in f:
        e, y = line.strip().split(', ')
        if y not in yoruba_vocab: continue
        if e in vocab:
            translation[e] = y
            if y not in translation_y_to_e: translation_y_to_e[y] = []
            translation_y_to_e[y].append(e)
            # yoruba_vocab.append(y)
            print(y, e)
print(len(yoruba_vocab))

with open('data/flickr8k.pickle', "rb") as f:
    data = pickle.load(f)

word_order = []
ind_to_word = {}
for word in data['VOCAB']:
    if word not in vocab: continue
    ind = data['VOCAB'][word]
    word_order.append(translation[word])
    if ind not in ind_to_word:
        ind_to_word[ind] = translation[word]
        print(ind, word, translation[word])
    else: print(f'problem with {word} -- {ind}')

        
d_n_tp = 0
d_n_tp_fp = 0
d_n_tp_fn = 0
det_score = 0
det_total = 0

l_n_tp = 0
l_n_fp = 0
l_n_fn = 0
score = 0
total = 0

threshold = 0.3

all_wav_names_kayode_used = list()
similarities = np.load(Path('data/yor-5k-init-preds/yor-5k-init/all_full_sigmoid_out.npz'), mmap_mode='r')
att_scores = np.load(Path('data/yor-5k-init-preds/yor-5k-init/all_attention_weight.npz'), mmap_mode='r')


for wav_name in similarities.files:
    all_wav_names_kayode_used.append(speaker + '_' + wav_name)
print(len(all_wav_names_kayode_used))

valid_wavs = set()
yoruba_alignments = {}
for txt_grid in Path('../Datasets/yfacc_v6/Flickr8k_alignment').rglob('*.TextGrid'):
    if str(txt_grid) == '../Datasets/yfacc_v6/Flickr8k_alignment/3187395715_f2940c2b72_0.TextGrid': continue
    grid = textgrids.TextGrid(txt_grid)
    wav = speaker + '_' + txt_grid.stem
    wav_fn = base_path / Path('yfacc_v6/flickr_audio_yoruba_test') / Path(str(wav) + '.wav')
    if wav_fn.is_file() is False: continue
    if wav not in yoruba_alignments: yoruba_alignments[wav] = []

    for interval in grid['words']:
        x = str(interval).split()
        label = str(interval).split('"')[1].lower()
        # if label in yoruba_vocab:
        start = x[-2].split('=')[-1]
        end = x[-1].split('=')[-1].split('>')[0]

        if label in yoruba_vocab: valid_wavs.add(wav)
        # if wav not in yoruba_alignments: yoruba_alignments[wav] = []
        # if label not in yoruba_alignments[wav]: yoruba_alignments[wav][label] = []
        yoruba_alignments[wav].append(((float(start), float(end)), label))
valid_wavs = list(valid_wavs)

wavs = set(list(yoruba_alignments.keys())).intersection(set(all_wav_names_kayode_used))
print(len(wavs))

d_n_tp = 0
d_n_tp_fp = 0
d_n_tp_fn = 0
det_score = 0
det_total = 0

l_n_tp = 0
l_n_fp = 0
l_n_fn = 0
score = 0
total = 0

i = 1
# indices = [i for i in range(len(classes)) if classes[i] in vocab]

for wav_name in tqdm(sorted(wavs)):
    name = '_'.join(wav_name.split('_')[1:])
    gt_trn = [w for _, w in yoruba_alignments[wav_name] if w in yoruba_vocab]

    target_dur = []
    for (start, end), tok in yoruba_alignments[wav_name]:
        # start, end = yoruba_alignments[wav_name][tok]
        if tok in yoruba_vocab:
            target_dur.append((((float(start)*100), (float(end)*100)), tok))

    indices = np.where(similarities[name] >= threshold)[0]
    hyp_trn = [ind_to_word[i] for i in indices if i in ind_to_word]  
    scores = att_scores[name]#[indices]
    
    # if i == 1:
    #     sim = [targets[name]['values'][i] for i in indices]
    #     print(sim)
    #     print(gt_trn)
    #     print(hyp_trn)
    #     i += 1

    d_analysis = get_detection_metric_count(hyp_trn, gt_trn)
    d_n_tp += d_analysis[0]
    d_n_tp_fp += d_analysis[1]
    d_n_tp_fn += d_analysis[2] 

    hyp_duration = []
    for i, word in enumerate(hyp_trn):
        max_frame = np.argmax(scores[indices[i], :])
        hyp_duration.append((max_frame, word))


    l_analysis = get_localisation_metric_count(hyp_duration, target_dur)
    l_n_tp += l_analysis[0]
    l_n_fp += l_analysis[1]
    l_n_fn += l_analysis[2]


    s, t = eval_detection_accuracy(hyp_duration, target_dur)
    det_score += s
    det_total += t
    

    s, t = eval_localisation_accuracy(hyp_duration, target_dur)
    score += s
    total += t
    # break


d_precision, d_recall, d_fscore = eval_detection_prf(d_n_tp, d_n_tp_fp, d_n_tp_fn)     
print("DETECTION SCORES: ")
print("No. predictions:", d_n_tp_fp)
print("No. true tokens:", d_n_tp_fn)
print("Precision: {} / {} = {:.4f}%".format(d_n_tp, d_n_tp_fp, d_precision*100.))
print("Recall: {} / {} = {:.4f}%".format(d_n_tp, d_n_tp_fn, d_recall*100.))
print("F-score: {:.4f}%".format(d_fscore*100.))
print("Accuracy: {} / {} =  {:.4f}%".format(det_score, det_total, (det_score/det_total) * 100.0))


l_precision, l_recall, l_fscore = eval_localisation_prf(l_n_tp, l_n_fp, l_n_fn)
print("LOCALISATION SCORES: ")
print("No. predictions:", l_n_fp)
print("No. true tokens:", l_n_fn)
print("Precision: {} / {} = {:.4f}%".format(l_n_tp, (l_n_tp + l_n_fp), l_precision*100.))
print("Recall: {} / {} = {:.4f}%".format(l_n_tp, (l_n_tp + l_n_fn), l_recall*100.))
print("F-score: {:.4f}%".format(l_fscore*100.))
print("Accuracy: {} / {} =  {:.4f}%".format(score, total, (score/total) * 100.0))
