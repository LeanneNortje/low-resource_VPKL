#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2022
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________

import torchaudio
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
from QbERT.align import align_semiglobal, score_semiglobal
import shutil
from statistics import mode
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models._meta import _COCO_CATEGORIES


model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).eval().to(0)
feature_number = '2'
pooling = torchvision.ops.MultiScaleRoIAlign([feature_number], 1, 2).to(0)

resize = transforms.Resize((256, 256))
to_tensor = transforms.ToTensor()
# image_normalize = transforms.Normalize(mean=RGB_mean, std=RGB_std)

def myRandomCrop(im):
    im = to_tensor(im)
    return im

def load_image(impath):
    img = Image.open(impath).convert('RGB')
    img = myRandomCrop(img)
    return img.unsqueeze(0)

def load_image_with_size(impath):
    img = Image.open(impath).convert('RGB')
    s = img.size
    img = myRandomCrop(img)
    return img.unsqueeze(0), s


audio_segments_dir = Path('QbERT/utterances_segmented')
ss_save_fn = 'data/support_set_images.npz'
support_set = np.load(ss_save_fn, allow_pickle=True)['support_set'].item()

VOCAB = []
with open('./data/34_keywords.txt', 'r') as f:
    for keyword in f:
        VOCAB.append(keyword.strip())

other_base = Path('../Datasets/Flickr8k_text')
labels_to_images = {}
for line in open(other_base / Path('Flickr8k.token.txt'), 'r'):
    parts = line.strip().split()
    name = parts[0].split('.')[0] + '_' + parts[0].split('#')[-1]
    # sentence = ' '.join(parts[1:]).lower()
    # tokenized = sent_tokenize(sentence)
    for word in parts[1:]:
        if word in VOCAB is False: continue
        if word not in labels_to_images: labels_to_images[word] = []
        labels_to_images[word].append(name)


s_images = {}
s_image_fns = []
s_image_fns_dict = {}

for word in tqdm(support_set):
    
    for im in support_set[word]:
        
        if im not in s_images: s_images[word] = []
        if word not in s_image_fns_dict: s_image_fns_dict[word] = []
        s_images[word].append(im)
        s_image_fns.append(Path(im).stem)
        s_image_fns_dict[word].append(Path(im).stem)


base = Path('/media/leannenortje/HDD/Datasets/Flickr8k_text')
train_names = set()
for line in open(base / Path('Flickr_8k.trainImages.txt'), 'r'):
    train_names.add(line.split('.')[0])
for line in open(base / Path('Flickr_8k.devImages.txt'), 'r'):
    train_names.add(line.split('.')[0])

data = []

for fn in tqdm(audio_segments_dir.rglob('*.npz')):
    image_name = '_'.join(fn.stem.split('_')[0:2])
    wav_name = fn.stem
    if image_name in train_names: data.append(fn)


query_scores = {}
record = {}

for q_word in s_images:
    id = q_word

    for wav in tqdm(data, desc=f'{q_word}'):
        image_name = '_'.join(wav.stem.split('_')[0:2])
        wav_name = wav.stem
        print(image_name, wav_name)
        break
    break