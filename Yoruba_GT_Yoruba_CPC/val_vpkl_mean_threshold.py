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

BACKEND = "nccl"
INIT_METHOD = "tcp://localhost:54321"

# config_library = {
#     "multilingual": "English_Hindi_DAVEnet_config.json",
#     "multilingual+matchmap": "English_Hindi_matchmap_DAVEnet_config.json",
#     "english": "English_DAVEnet_config.json",
#     "english+matchmap": "English_matchmap_DAVEnet_config.json",
#     "hindi": "Hindi_DAVEnet_config.json",
#     "hindi+matchmap": "Hindi_matchmap_DAVEnet_config.json",
# }

scipy_windows = {
    'hamming': scipy.signal.hamming,
    'hann': scipy.signal.hann, 
    'blackman': scipy.signal.blackman,
    'bartlett': scipy.signal.bartlett
    }

categories = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90, 92, 93, 95, 100, 107, 109, 112, 118, 119, 122, 125, 128, 130, 133, 138, 141, 144, 145, 147, 148, 149, 151, 154, 155, 156, 159, 161, 166, 168, 171, 175, 176, 177, 178, 180, 181, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200]

def preemphasis(signal,coeff=0.97):  
    # function adapted from https://github.com/dharwath
    
    return np.append(signal[0],signal[1:]-coeff*signal[:-1])

categories_to_ind = {}

for i, cat in enumerate(categories):
    categories_to_ind[cat] = i

def myRandomCrop(im, resize, to_tensor):

        im = resize(im)
        im = to_tensor(im)
        return im

def LoadAudio(path, audio_conf):

    audio_type = audio_conf.get('audio_type')
    if audio_type not in ['melspectrogram', 'spectrogram']:
        raise ValueError('Invalid audio_type specified in audio_conf. Must be one of [melspectrogram, spectrogram]')

    preemph_coef = audio_conf.get('preemph_coef')
    sample_rate = audio_conf.get('sample_rate')
    window_size = audio_conf.get('window_size')
    window_stride = audio_conf.get('window_stride')
    window_type = audio_conf.get('window_type')
    num_mel_bins = audio_conf.get('num_mel_bins')
    target_length = audio_conf.get('target_length')
    fmin = audio_conf.get('fmin')
    n_fft = audio_conf.get('n_fft', int(sample_rate * window_size))
    win_length = int(sample_rate * window_size)
    hop_length = int(sample_rate * window_stride)

    # load audio, subtract DC, preemphasis
    y, sr = librosa.load(path, sample_rate)
    dur = librosa.get_duration(y=y, sr=sr)
    nsamples = y.shape[0]
    if y.size == 0:
        y = np.zeros(target_length)
    y = y - y.mean()
    y = preemphasis(y, preemph_coef)

    # compute mel spectrogram / filterbanks
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        window=scipy_windows.get(window_type, scipy_windows['hamming']))
    spec = np.abs(stft)**2 # Power spectrum
    if audio_type == 'melspectrogram':
        mel_basis = librosa.filters.mel(sr, n_fft, n_mels=num_mel_bins, fmin=fmin)
        melspec = np.dot(mel_basis, spec)
        logspec = librosa.power_to_db(melspec, ref=np.max)
    elif audio_type == 'spectrogram':
        logspec = librosa.power_to_db(spec, ref=np.max)
    # n_frames = logspec.shape[1]
    logspec = torch.FloatTensor(logspec)
    return torch.tensor(logspec), nsamples#, n_frames

def LoadImage(impath, resize, image_normalize, to_tensor):
    img = Image.open(impath).convert('RGB')
    # img = self.image_resize_and_crop(img)
    img = myRandomCrop(img, resize, to_tensor)
    img = image_normalize(img)
    return img

def PadFeat(feat, target_length, padval):
    nframes = feat.shape[1]
    pad = target_length - nframes

    if pad > 0:
        feat = np.pad(feat, ((0, 0), (0, pad)), 'constant',
            constant_values=(padval, padval))
    elif pad < 0:
        nframes = target_length
        feat = feat[:, 0: pad]

    return torch.tensor(feat).unsqueeze(0), torch.tensor(nframes).unsqueeze(0)

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

def get_gt_token_duration(target_dur, valid_gt_trn):
            
    token_dur = []
    for start_end, dur, tok in target_dur:
        if tok not in valid_gt_trn:
            continue
        token_dur.append((start_end, tok.casefold()))
    return token_dur

def spawn_training(rank, world_size, image_base, args):

    # # Create dataloaders
    dist.init_process_group(
        BACKEND,
        rank=rank,
        world_size=world_size,
        init_method=INIT_METHOD,
    )

    args["path"] = Path('../Datasets')
    flickr_boundaries_fn = Path(args["path"]) / Path('flickr_audio/flickr_8k.ctm')
    flickr_audio_dir =  Path(args["path"]) / Path('yfacc_v6')
    flickr_images_fn = Path(args["path"]) / Path('Flicker8k_Dataset')

    key = np.load(Path('data/label_key.npz'), allow_pickle=True)['key'].item()
    translation = np.load(Path('data/label_key.npz'), allow_pickle=True)['translation_e_to_y'].item()
    yoruba_vocab = list(key.keys())
    print(len(yoruba_vocab))

    yoruba_samples = {}
    if rank == 0: 
        fn = Path(f'../Datasets/yfacc_v6/Flickr8k_text/Flickr8k.token.dev_yoruba.txt')
        
        with open(fn, 'r') as f:
            for line in f:
                
                jpg_name = line.strip().split()[0].split('.')[0]
                jpg_num = line.strip().split()[0].split('#')[-1]

                wav_name = f'S001_{jpg_name}_{jpg_num}'

                wav_fn = Path(args["path"]) / Path('yfacc_v6/flickr_audio_yoruba_dev') / Path(str(wav_name) + '.wav')
                if wav_fn.is_file() is False: continue
                transcription = ' '.join(line.strip().split()[1:]).lower()
                transcription = '_' + transcription + ' '
                transcription = transcription.replace("_", " ")
                if wav_name not in yoruba_samples: yoruba_samples[wav_name] = []

                for word in yoruba_vocab:

                    search_word = ' ' + word + ' '
                    if search_word in transcription: 
                        
                        yoruba_samples[wav_name].append(word)

        # with open('data/flickr8k.pickle', "rb") as f:
        #     data = pickle.load(f)

        # samples = data['dev']

        # for entry in samples:
        #     print(entry['wave'])
        #     print(entry)
        #     break

        audio_conf = args["audio_config"]
        target_length = audio_conf.get('target_length', 1024)
        padval = audio_conf.get('padval', 0)
        image_conf = args["image_config"]
        crop_size = image_conf.get('crop_size')
        center_crop = image_conf.get('center_crop')
        RGB_mean = image_conf.get('RGB_mean')
        RGB_std = image_conf.get('RGB_std')

        resize = transforms.Resize((256, 256))
        to_tensor = transforms.ToTensor()
        image_normalize = transforms.Normalize(mean=RGB_mean, std=RGB_std)

        image_resize = transforms.transforms.Resize((256, 256))

        vocab = []
        with open('./data/34_keywords.txt', 'r') as f:
            for keyword in f:
                vocab.append(keyword.strip())

        trans = transforms.ToPILImage()

        # Create models
        audio_model = mutlimodal(args).to(rank)

        image_model = vision(args).to(rank)

        attention = ScoringAttentionModule(args).to(rank)
        contrastive_loss = ContrastiveLoss(args).to(rank)

        model_with_params_to_update = {
            "audio_model": audio_model,
            "attention": attention,
            "contrastive_loss": contrastive_loss,
            "image_model": image_model
            }
        model_to_freeze = {
            }
        trainable_parameters = getParameters(model_with_params_to_update, model_to_freeze, args)

        if args["optimizer"] == 'sgd':
            optimizer = torch.optim.SGD(
                trainable_parameters, args["learning_rate_scheduler"]["initial_learning_rate"],
                momentum=args["momentum"], weight_decay=args["weight_decay"]
                )
        elif args["optimizer"] == 'adam':
            optimizer = torch.optim.Adam(
                trainable_parameters, args["learning_rate_scheduler"]["initial_learning_rate"],
                weight_decay=args["weight_decay"]
                )
        else:
            raise ValueError('Optimizer %s is not supported' % args["optimizer"])

        scaler = torch.cuda.amp.GradScaler()

        audio_model = DDP(audio_model, device_ids=[rank])
        image_model = DDP(image_model, device_ids=[rank])

        heading(f'\nRetoring model parameters from best epoch ')
        info, epoch, global_step, best_epoch, best_acc = loadModelAttriburesAndTrainingAMP(
            args["exp_dir"], audio_model, image_model, attention, contrastive_loss, optimizer, rank, False
            )
        
        image_base = Path(args["path"]) / Path('Flicker8k_Dataset/')

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

        images_for_keywords = np.load(Path('data/words_to_images_for_det_and_loc.npz'), allow_pickle=True)['word_images'].item()

        with torch.no_grad():

            images = []
            iVOCAB = {}
            words_to_iVOCAB = {}
            num_word = 0
            for english_word in images_for_keywords:
                if translation[english_word] in words_to_iVOCAB: continue
                for im_fn, _ in images_for_keywords[english_word]:
                    image = LoadImage(
                        Path(args['path']) / Path('Flicker8k_Dataset') / Path((im_fn.stem) + '.jpg'), resize, image_normalize, to_tensor)
                    images.append(image.unsqueeze(0).cpu())

                iVOCAB[num_word] = translation[english_word]
                words_to_iVOCAB[translation[english_word]] = num_word
                num_word += 1

            images = torch.cat(images, dim=0)
            image_output = image_model(images.to(rank))
            image_output = image_output.view(image_output.size(0), image_output.size(1), -1).transpose(1, 2)
            num = image_output.size(0)//len(yoruba_vocab)
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)

            i = 1
            values = []
            for wav_name in tqdm(yoruba_samples):

                gt_trn = yoruba_samples[wav_name]
                wav_fn = Path(args["path"]) / Path('yfacc_v6/flickr_audio_yoruba_dev') / Path(str(wav_name) + '.wav') 

                english_audio_feat, nsamples = LoadAudio(wav_fn, audio_conf)
                english_audio_feat, english_nframes = PadFeat(english_audio_feat, target_length, padval)

                _, _, english_output = audio_model(english_audio_feat.to(rank))
                temp = english_nframes
                english_nframes = NFrames(english_audio_feat, english_output, english_nframes) 

                english_output = english_output.repeat(image_output.size(0), 1, 1)
                english_nframes = english_nframes.repeat(image_output.size(0)).to(rank)

                sim = attention.encodeSimilarity(image_output, None, english_output, english_nframes)
                
                similarities = torch.zeros(len(iVOCAB), device=english_output.device)
                for n in range(sim.size(0)):
                    similarities[n//num] += sim[n].squeeze()

                similarities = similarities / num
                # similarities = torch.nn.functional.normalize(similarities.unsqueeze(0)).squeeze(0)
    

                for w in gt_trn:
                    values.append(similarities[words_to_iVOCAB[w]].item())

                
            values = np.asarray(values)
            print(values.mean())
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--resume", action="store_true", dest="resume",
            help="load from exp_dir if True")
    parser.add_argument("--config-file", type=str, default='matchmap', choices=['multilingual', 'multilingual+matchmap'], help="Model config file.")
    parser.add_argument("--restore-epoch", type=int, default=-1, help="Epoch to generate accuracies for.")
    parser.add_argument("--image-base", default="/storage", help="Model config file.")
    parser.add_argument("--path", type=str, help="Path to Flicker8k_Dataset.")
    command_line_args = parser.parse_args()
    restore_epoch = command_line_args.restore_epoch

    # Setting up model specifics
    heading(f'\nSetting up model files ')
    args, image_base = modelSetup(command_line_args, True)

    world_size = 1
    mp.spawn(
        spawn_training,
        args=(world_size, image_base, args),
        nprocs=world_size,
        join=True,
    )