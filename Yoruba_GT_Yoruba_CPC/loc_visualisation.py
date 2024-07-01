#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
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

flickr_boundaries_fn = Path('/home/leannenortje/Datasets/flickr_audio/flickr_8k.ctm')
flickr_audio_dir = flickr_boundaries_fn.parent / "wavs"
flickr_images_fn = Path('/home/leannenortje/Datasets/Flicker8k_Dataset/')
flickr_segs_fn = Path('./data/flickr_image_masks/')

config_library = {
    "multilingual": "English_Hindi_DAVEnet_config.json",
    "multilingual+matchmap": "English_Hindi_matchmap_DAVEnet_config.json",
    "english": "English_DAVEnet_config.json",
    "english+matchmap": "English_matchmap_DAVEnet_config.json",
    "hindi": "Hindi_DAVEnet_config.json",
    "hindi+matchmap": "Hindi_matchmap_DAVEnet_config.json",
}

# vocab_ids = {
#     'air': [187],
#     'baby': [1],
#     'ball': [37],
#     'beach': [155, 154],
#     'bike': [2],
#     'boy': [1],
#     'building': [197, 128],
#     'car': [3],
#     'children': [1],
#     'climbing': [192, 161],
#     'dirt': [194],
#     'dogs': [18],
#     'face': [1],
#     'field': [193, 145],
#     'football': [37],
#     'grass': [193],
#     'hair': [1, 89],
#     'mountain': [192],
#     'mouth': [1],
#     'ocean': [155],
#     'orange': [55],
#     'park': [145],
#     'pool': [178],
#     'rides': [2, 3, 4, 5, 6, 7, 8, 9],
#     'riding': [2, 3, 4, 5, 6, 7, 8, 9],
#     'road': [149],
#     'rock': [198],
#     'sand': [154],
#     'sits': [15, 62, 63],
#     'sitting': [15, 62, 63],
#     'skateboard': [41],
#     'smiling': [1],
#     'snow': [159],
#     'snowy': [159],
#     'soccer': [37],
#     'street': [149],
#     'toy': [88],
#     'tree': [184],
#     'water': [178],
#     'women': [1]
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

def LoadRawImage(impath):
    img = Image.open(impath).convert('RGB')
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


def spawn_training(rank, world_size, image_base, args):

    # # Create dataloaders
    dist.init_process_group(
        BACKEND,
        rank=rank,
        world_size=world_size,
        init_method=INIT_METHOD,
    )

    names = ['3170897628_3054087f8c']

    english_samples = []

    if rank == 0: 

        args["path"] = Path('../Datasets')
        boundaries = {}
        with open(flickr_boundaries_fn, 'r') as file:
            for line in tqdm(file):
                parts = line.strip().split()
                name = parts[0].split(".")[0]
                
                seg = Path(flickr_segs_fn) / Path(name + ".npz")
                if seg.is_file():
                    
                    number = parts[0].split("#")[-1]
                    wav = flickr_audio_dir / Path(name + "_" + number + ".wav")
                    img = Path(flickr_images_fn) / Path(name + ".jpg")
                    if name + "_" + number not in boundaries:
                        boundaries[name + "_" + number] = {"wav": wav, "img": img, "seg": seg}
                    boundaries[name + "_" + number].setdefault('boundaries', []).append((parts[2], parts[3], parts[4].lower()))
        print('Num boundaries: ', len(boundaries))

        

        samples = {}
        for dev in ['train', 'dev', 'test']:
            fn = Path(f'/media/leannenortje/HDD/Datasets/yfacc_v6/Flickr8k_text/Flickr8k.token.{dev}_yoruba.txt')
            with open(fn, 'r') as f:
                for line in f:
                    
                    jpg_name = line.strip().split()[0].split('.')[0]
                    if jpg_name not in names: continue
                    jpg_num = line.strip().split()[0].split('#')[-1]
                    word = line.strip().split()[-1]

                    wav_name = f'S001_{jpg_name}_{jpg_num}'

                    wav_fn = Path('/home/leannenortje/HDD/Datasets/yfacc_v6/flickr_audio_yoruba_test') / Path(str(wav_name) + '.wav')
                    if wav_name not in samples: samples[wav_name] = []
                    samples[wav_name].append(word)    

        print(samples)

    #     with open('data/flickr8k.pickle', "rb") as f:
    #         data = pickle.load(f)

    #     samples = data['test']

        audio_conf = args["audio_config"]
        target_length = audio_conf.get('target_length', 1024)
        padval = audio_conf.get('padval', 0)
        image_conf = args["image_config"]
        crop_size = image_conf.get('crop_size')
        center_crop = image_conf.get('center_crop')
        RGB_mean = image_conf.get('RGB_mean')
        RGB_std = image_conf.get('RGB_std')

        # image_resize_and_crop = transforms.Compose(
        #         [transforms.Resize(224), transforms.ToTensor()])
        resize = transforms.Resize((256, 256))
        to_tensor = transforms.ToTensor()
        image_normalize = transforms.Normalize(mean=RGB_mean, std=RGB_std)

        image_resize = transforms.transforms.Resize((256, 256))

    #     vocab = [key for key in vocab_ids]

    #     cat_ids_to_labels = np.load(Path("data/mask_cat_id_labels.npz"), allow_pickle=True)['cat_ids_to_labels'].item()
        
        trans = transforms.ToPILImage()

    #     # Create models
    #     audio_model = mutlimodal(args).to(rank)

    #     image_model_name = imageModel(args)
    #     image_model = image_model_name(args, pretrained=args["pretrained_image_model"]).to(rank)

    #     attention = ScoringAttentionModule(args).to(rank)
    #     contrastive_loss = ContrastiveLoss(args).to(rank)

    #     model_with_params_to_update = {
    #         "audio_model": audio_model,
    #         "attention": attention,
    #         "contrastive_loss": contrastive_loss
    #         }
    #     model_to_freeze = {
    #         "image_model": image_model
    #         }
    #     trainable_parameters = getParameters(model_with_params_to_update, model_to_freeze, args)

    #     if args["optimizer"] == 'sgd':
    #         optimizer = torch.optim.SGD(
    #             trainable_parameters, args["learning_rate_scheduler"]["initial_learning_rate"],
    #             momentum=args["momentum"], weight_decay=args["weight_decay"]
    #             )
    #     elif args["optimizer"] == 'adam':
    #         optimizer = torch.optim.Adam(
    #             trainable_parameters, args["learning_rate_scheduler"]["initial_learning_rate"],
    #             weight_decay=args["weight_decay"]
    #             )
    #     else:
    #         raise ValueError('Optimizer %s is not supported' % args["optimizer"])

    #     scaler = torch.cuda.amp.GradScaler()

    #     audio_model = DDP(audio_model, device_ids=[rank])
    #     image_model = DDP(image_model, device_ids=[rank])
    #     # attention = DDP(attention, device_ids=[rank])
    #     # contrastive_loss = DDP(contrastive_loss, device_ids=[rank])


    #     heading(f'\nRetoring model parameters from best epoch ')
    #     info, epoch, global_step, best_epoch, best_acc = loadModelAttriburesAndTrainingAMP(
    #         args["exp_dir"], audio_model, image_model, attention, contrastive_loss, optimizer, rank, False
    #         )
        
    #     image_base = Path('/storage/Datasets/Flicker8k_Dataset/')

    #     threshold = 0.5885123830063176

    #     images_for_keywords = np.load(Path('data/words_to_images_for_det_and_loc.npz'), allow_pickle=True)['word_images'].item()

    #     cat_ids_to_labels = np.load(Path("data/mask_cat_id_labels.npz"), allow_pickle=True)['cat_ids_to_labels'].item()
    #     keywords = []
    #     for k in cat_ids_to_labels: 
    #         if len(cat_ids_to_labels[k].split('-')) == 1: keywords.append(cat_ids_to_labels[k])

        # base = Path('loc_examples')
        # if base.is_dir() is False: base.mkdir(parents=True)

        with torch.no_grad():

            i = 1
            for entry in tqdm(samples):
                print(entry)

                gt_trn = [j for j in entry["trn"] if j in images_for_keywords]
                wav_name = str(Path(entry['wave']).stem)
                wav_fn = Path('/home/leannenortje/Datasets/flickr_audio/wavs') / Path(str(Path(entry['wave']).stem) + '.wav') 
                if str(Path(entry['wave']).stem) not in boundaries: continue
                target_dur = [(((float(start)*100)//2, (float(start)*100)//2 + (float(dur)*100)//2), tok) for (start, dur, tok) in boundaries[str(Path(entry['wave']).stem)]['boundaries'] if tok.casefold() in vocab]
                
                raw_image = LoadRawImage(Path('/home/leannenortje/Datasets/Flicker8k_Dataset') / Path('_'.join(str(Path(entry['wave']).stem).split('_')[0:2]) + '.jpg'))
                image = LoadImage(
                    Path('/home/leannenortje/Datasets/Flicker8k_Dataset') / Path('_'.join(str(Path(entry['wave']).stem).split('_')[0:2]) + '.jpg'), resize, image_normalize, to_tensor)
                image_output = image_model(image.unsqueeze(0).to(rank))
                image_output = image_output.view(image_output.size(0), image_output.size(1), -1).transpose(1, 2)

                english_audio_feat, nsamples = LoadAudio(wav_fn, audio_conf)
                english_audio_feat, english_nframes = PadFeat(english_audio_feat, target_length, padval)

    #             _, _, english_output = audio_model(english_audio_feat.to(rank))
    #             temp = english_nframes
    #             english_nframes = NFrames(english_audio_feat, english_output, english_nframes) 

    #             scores, _, _, _, sim, _ = attention.encode(image_output, english_output, english_nframes)

    #             fig = plt.figure(figsize=(100, 25), constrained_layout=True)
    #             gs = GridSpec(1, 6, figure=fig)

    #             ax_raw_image = fig.add_subplot(gs[0, 0:2])
    #             ax_raw_image.imshow(raw_image)
    #             ax_raw_image.axis('off')

    #             # ax_spectogram = fig.add_subplot(gs[0, 2:])
    #             pooling_ratio = english_audio_feat.size(-1) / english_output.size(-1)
    #             # english_audio_feat = english_audio_feat[0, :, 0:temp[0].item()]
                
    #             # sns.heatmap(np.flip(english_audio_feat.squeeze().cpu().numpy(), axis=0), cmap='viridis', ax=ax_spectogram)
    #             # ax_spectogram.set_xlabel('Audio frames')
    #             # ax_spectogram.set_ylabel("Mel-bins")
    #             if wav_name not in boundaries: continue
    #             these_boundaries = boundaries[wav_name]['boundaries']
    #             boundaries_print_downsampled_2 = []
    #             boundaries_print_original_size = []

    #             for word in these_boundaries:
    #                 start = float(word[0])
    #                 dur = float(word[1])
    #                 label = word[2]
                    
    #                 begin = int(start * 100)
    #                 end = int((start + dur) * 100)# + 1
                    
    #                 boundaries_print_original_size.append((label, begin, end))

    #                 begin = begin // pooling_ratio
    #                 end = end // pooling_ratio

    #                 boundaries_print_downsampled_2.append((label, begin, end))

    #             # for (label, begin, end) in boundaries_print_original_size:
    #             #     c = 'w'
    #             #     ax_spectogram.add_artist(lines.Line2D([begin, begin], [0, 40], color=c)) 
    #             #     if label == these_boundaries[-1][2]: 
    #             #         ax_spectogram.add_artist(lines.Line2D([end, end], [0, 40], color=c, linewidth=3.5)) 
    #             #     ax_spectogram.text(begin+(end-begin)//2 + 0.5, 20, label, horizontalalignment='center', verticalalignment='center', color=c, fontfamily='serif', fontsize='large', fontweight='demibold', rotation=90)
    #             #     ax_spectogram.set_xticks(list(np.arange(0, temp[0].item())))

    #             scores = scores[:, :, 0:english_nframes].squeeze(0).cpu()
    #             # scores_heat = fig.add_subplot(gs[0, 2:7])
    #             # # sns.set(font_scale=1.8)
    #             # sns.heatmap(scores.numpy(), cmap='viridis', ax=scores_heat)
    #             # scores_heat.set_xlabel('Audio frames')
    #             # scores_heat.set_ylabel("Image embeddings")
    #             # scores_heat.axis('off')

    #             scores_plot = fig.add_subplot(gs[0, 2:])
    #             scores_plot.plot(scores.squeeze().cpu().numpy(), linewidth=10, c='seagreen')
    #             scores_plot.set_xticks([])
    #             scores_plot.set_yticks([])
    #             scores_plot.axis('off')
    #             save = False
    #             for (label, begin, end) in boundaries_print_downsampled_2:
    #                 # c = 'w'
    #                 # scores_heat.add_artist(lines.Line2D([begin, begin], [0, scores.size(-1)], color=c)) 
    #                 # if label == these_boundaries[-1][2]: 
    #                 #     scores_heat.add_artist(lines.Line2D([end, end], [0, scores.size(-1)], color=c, linewidth=3.5)) 
    #                 # scores_heat.text(begin+(end-begin)//2 + 0.5, 0.5*scores.size(0), label, horizontalalignment='center', verticalalignment='center', color=c, fontfamily='serif', fontsize=100, fontweight='demibold', rotation=90)
    #                 # scores_heat.figure.axes[-1].set_ylabel(['low', '', '', '', '', 'high'])

    #                 c = 'black'
    #                 min_val = torch.min(scores)
    #                 max_val = torch.max(scores)
    #                 scores_plot.add_artist(lines.Line2D([begin, begin], [min_val, max_val+0.05], color=c)) 
    #                 if label == these_boundaries[-1][2]: 
    #                     scores_plot.add_artist(lines.Line2D([end, end], [min_val-0.05, max_val+0.05], color='grey', linewidth=3.5))
    #                 if label in vocab_ids: 
    #                     scores_plot.text(begin+(end-begin)//2 + 0.5, min_val+0.025, label, horizontalalignment='center', verticalalignment='center', color=c, fontfamily='serif', fontsize=80, fontweight='demibold')
    #                     save = True

    #             if save: plt.savefig(base / Path(f'{i}.png'))
    #             # plt.show()

    #             i += 1

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--resume", action="store_true", dest="resume",
            help="load from exp_dir if True")
    parser.add_argument("--config-file", type=str, default='matchmap', choices=['multilingual', 'multilingual+matchmap'], help="Model config file.")
    parser.add_argument("--restore-epoch", type=int, default=-1, help="Epoch to generate accuracies for.")
    parser.add_argument("--image-base", default="/media/leannenortje/HDD", help="Model config file.")
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