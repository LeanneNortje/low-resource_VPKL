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

VOCAB = []
with open('./data/34_keywords.txt', 'r') as f:
    for keyword in f:
        VOCAB.append(keyword.strip())


labels_to_images = {}

translation = {}
translation_y_to_e = {} 
yoruba_vocab = []
with open(Path('../Datasets/yfacc_v6/Flickr8k_text/eng_yoruba_keywords.txt'), 'r') as f:
    for line in f:
        e, y = line.strip().split(', ')
        if e in VOCAB:
            translation[e] = y
            if y not in translation_y_to_e: translation_y_to_e[y] = []
            translation_y_to_e[y].append(e)
            yoruba_vocab.append(y)

base = Path('../Datasets/yfacc_v6/Flickr8k_text')
train_names = set()
for subset in ['train', 'dev']:
    for line in open(base / Path(f'Flickr8k.token.{subset}_yoruba.txt'), 'r'):

        parts = line.strip().split()

        name = parts[0].split('.')[0] + '_' + parts[0].split('#')[-1]
        train_names.add(name)

print(list(train_names)[0])

K = 10
audio_segments_dir = Path('QbERT/utterances_segmented')
audio_query_dir = Path('QbERT/queries_segmented')
audio_dir = Path('../Datasets/flickr_audio/wavs/')
ss_save_fn = 'data/support_set_audio.npz'
support_set = np.load(ss_save_fn, allow_pickle=True)['support_set'].item()
pam = np.load("QbERT/sim.npy")


other_base = Path('../Datasets/Flickr8k_text')
labels_to_images = {}
for line in open(other_base / Path('Flickr8k.token.txt'), 'r'):

    parts = line.strip().split()
    name = parts[0].split('.')[0] + '_' + parts[0].split('#')[-1]
    # if name not in train_names: continue

    sentence = ' '.join(parts[1:]).lower()
    # tokenized = sent_tokenize(sentence)
    for word in VOCAB:
        search_word = ' ' + word + ' '
        if search_word not in sentence: continue
        # print(word, sentence)
        if word not in labels_to_images: labels_to_images[word] = []
        labels_to_images[word].append(name)


s_audio = {}
s_wavs = []
s_wav_dict = {}
# word_thresholds = {}
for word in tqdm(support_set):
    # lengths = []
    for wav in support_set[word]:
        fn = (audio_query_dir / wav.stem).with_suffix('.npz')
        
        query = np.load(fn)
        # print(query["codes"], query["boundaries"])
        x = query["codes"]#[query["boundaries"][:-1]]
        # lengths.append(len(x))

        if word not in s_audio: s_audio[word] = []
        if word not in s_wav_dict: s_wav_dict[word] = []
        s_audio[word].append(x)
        s_wavs.append(Path(wav).stem)
        s_wav_dict[word].append(Path(wav).stem)
    
    # if word not in word_thresholds: 
    #     if mode(lengths) <= 3:
    #         word_thresholds[word] = 1
    #     else:
    #         word_thresholds[word] = 1


base = Path('../Datasets/Flickr8k_text')
# train_names = set()
# for line in open(base / Path('Flickr_8k.trainImages.txt'), 'r'):
#     train_names.add(line.split('.')[0])
# for line in open(base / Path('Flickr_8k.devImages.txt'), 'r'):
#     train_names.add(line.split('.')[0])

data = []

for fn in tqdm(audio_segments_dir.rglob('*.npz')):
    name = Path(fn).stem
    if name in train_names: data.append(fn)
    

query_scores = {}
record = {}

for q_word in s_audio:
    id = q_word
    # print(len(labels_to_images[q_word]))
    for wav in tqdm(data, desc=f'{q_word}'):

        wav_name = Path(wav).stem
        if wav_name in s_wavs: continue

        if id not in query_scores: query_scores[id] = {'values': [], 'wavs': []}

        fn = (audio_segments_dir / wav.stem).with_suffix('.npz')
        test = np.load(fn)
        y = test["codes"]#[test["boundaries"][:-1]]

        mean_score = []

        for x_i, x in enumerate(s_audio[q_word]):
            
            path, p, q, score = align_semiglobal(x, y, pam, 1)
            # H, T = score_semiglobal(x, y, pam, 1)

            indexes, = np.where(np.array(p) != -1)
            if len(indexes) > 2:
                start, end = indexes[0], indexes[-1]
                norm_score = score / (end - start)
                mean_score.append(norm_score)
                
                if wav_name not in record: record[wav_name] = {}
                record[wav_name][id] = (path, start, end, p, q, indexes)

        query_scores[id]['values'].append(np.mean(mean_score))
        query_scores[id]['wavs'].append(wav)   
    # break 

save_dir = Path('data/english_segments')
top_N = 200
newly_labeled = {}

for id in tqdm(query_scores):

    indices = np.argsort(query_scores[id]['values'])[::-1]
    i = 0
    count = 0
    # print(len(labels_to_images[id]), len(labels_to_images[id])//3)
    while i < len(labels_to_images[id])//2 and count < len(query_scores[id]['wavs']):

        wav = Path(query_scores[id]['wavs'][indices[count]])
        wav_name = wav.stem

        fn = (audio_dir / wav_name).with_suffix('.wav')
        # aud, sr = torchaudio.load(fn)
        new_fn = (Path('data/english_wavs') / wav_name).with_suffix('.wav')
        new_fn.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(fn, new_fn)


        segment_fn = (audio_segments_dir / wav_name).with_suffix('.npz')
        test = np.load(segment_fn)
        path, start, end, p, q, indexes = record[str(wav_name)][id]
        _, b0 = path[start - 1]
        _, bT = path[end]
        w0, wT = 0.02 * test["boundaries"][b0 - 1], 0.02 * test["boundaries"][bT]
        offset = int(w0 * 16000)
        frames = int(np.abs(wT - w0) * 16000)
        aud, sr = torchaudio.load(new_fn, frame_offset=offset, num_frames=frames)
        word_new_fn = (Path('data/english_wavs') / f'{wav_name}_word_{id}').with_suffix('.wav')

        # print(i, count, aud.size(1))
        if frames == aud.size(1) and wav_name in labels_to_images[id]:

            torchaudio.save(word_new_fn, aud, sr)
            if id not in newly_labeled: newly_labeled[id] = []
            newly_labeled[id].append(wav_name)
            i += 1
            # if i == 20: break
        count += 1


    for wav_name in s_wav_dict[id]:

        new_name = '_'.join(wav_name.split('_')[:-1])
  
        fn = (audio_dir / new_name).with_suffix('.wav')
        new_fn = (Path('data/english_wavs') / new_name).with_suffix('.wav')
        new_fn.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(fn, new_fn)


        orig_word_fn = (Path('data/support_set_english_audio') / f'{wav_name}').with_suffix('.wav')
        word_new_fn = (Path('data/english_wavs'  ) / f'{new_name}_word_{id}').with_suffix('.wav')
        shutil.copy(orig_word_fn, word_new_fn)
        newly_labeled[id].append(new_name)


acc = {}
for word in newly_labeled:
    if word not in acc: acc[word] = {'correct': 0, 'total': 0}
    for wav_name in newly_labeled[word]:
        if wav_name in labels_to_images[word]: acc[word]['correct'] += 1
        acc[word]['total'] += 1

for word in acc:
    print(word, acc[word]['correct'], acc[word]['total'], 100 * acc[word]['correct'] / acc[word]['total'])

np.savez_compressed(
    Path("./data/sampled_query_audio_data"), 
    data=newly_labeled
)