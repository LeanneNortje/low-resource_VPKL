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

audio_segments_dir = Path('QbERT/utterances_segmented')
audio_query_dir = Path('QbERT/queries_segmented')
audio_dir = Path('../Datasets/yfacc_v6')
ss_save_fn = 'data/support_set.npz'
support_set = np.load(ss_save_fn, allow_pickle=True)['support_set'].item()
pam = np.load("QbERT/sim.npy")
K = 5

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



s_audio = {}
s_wavs = []
s_wav_dict = {}
for name in tqdm(support_set):
    wav, start, end, word = support_set[name]
    
    fn = (audio_query_dir / wav.stem).with_suffix('.npz')

    query = np.load(fn)
    x = query["codes"]

    if word not in s_audio: s_audio[word] = []
    if word not in s_wav_dict: s_wav_dict[word] = []
    s_audio[word].append(x)
    s_wavs.append(Path(wav).stem)
    s_wav_dict[word].append(wav.stem)

key = {}
id_to_word_key = {}
for i, l in enumerate(s_audio):
    key[l] = i
    id_to_word_key[i] = l

np.savez_compressed(
    Path('data/label_key'),
    key=key,
    id_to_word_key=id_to_word_key, 
    translation_y_to_e=translation_y_to_e,
    translation_e_to_y=translation
)

base = Path('../Datasets/yfacc_v6/Flickr8k_text')
train_names = set()
for subset in ['train', 'dev']:
    for line in open(base / Path(f'Flickr8k.token.{subset}_yoruba.txt'), 'r'):

        parts = line.strip().split()

        name = 'S001_' + parts[0].split('.')[0] + '_' + parts[0].split('#')[-1]
        train_names.add(name)
        sentence = ' '.join(parts[1:]).lower()
        sentence = ' ' + sentence + ' '

        for word in yoruba_vocab:
            search_word = ' ' + word + ' '
            if search_word not in sentence: continue
            if word not in labels_to_images: labels_to_images[word] = []
            labels_to_images[word].append(name)


# for word in labels_to_images: print(f'{word} {len(labels_to_images[word])}')
# for line in open(base / Path('Flickr_8k.token.dev_yoruba.txt'), 'r'):
#     train_names.add(line.split('.')[0])

data = []

for fn in tqdm(audio_segments_dir.rglob('*.npz')):
    name = fn.stem #'_'.join(fn.stem.split('_')[0:2])
    if name in train_names: data.append(fn)
    

query_scores = {}
record = {}

for q_word in s_audio:
    id = q_word

    for wav in tqdm(data, desc=f'{q_word}'):
        
        wav_name = wav.stem
        if wav_name in s_wavs: continue

        if id not in query_scores: query_scores[id] = {'values': [], 'wavs': []}

        fn = (audio_segments_dir / wav_name).with_suffix('.npz')
        test = np.load(fn)
        y = test["codes"]

        mean_score = []

        for x_i, x in enumerate(s_audio[q_word]):
            
            path, p, q, score = align_semiglobal(x, y, pam, 1)

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


save_dir = Path('data/yoruba_training_samples')
newly_labeled = {}

for id in tqdm(query_scores):

    top_N = len(labels_to_images[id]) - K

    indices = np.argsort(query_scores[id]['values'])[::-1]

    i = 0
    count = 0
    if id not in newly_labeled: newly_labeled[id] = []

    while i < top_N and count < len(query_scores[id]['wavs']):
        
        wav = Path(query_scores[id]['wavs'][indices[count]])
        wav_name = wav.stem
        # if wav_name not in labels_to_images[id]: continue

        fn = list(audio_dir.rglob(f'*{wav_name}.wav'))[0]
        new_fn = (Path('data/yoruba_wavs') / wav_name).with_suffix('.wav')
        new_fn.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(fn, new_fn)


        segment_fn = (audio_segments_dir / wav_name).with_suffix('.npz')
        test = np.load(segment_fn)
        path, start, end, p, q, indexes = record[str(wav_name)][id]
        _, b0 = path[start]
        _, bn = path[end]
        w0, wT = 0.02 * test["boundaries"][b0 - 1], 0.02 * test["boundaries"][bn]
        offset = int(w0 * 48000)
        frames = int(np.abs(wT - w0) * 48000)
        # b0 = round(test["boundaries"][b0 - 1] * 0.02, 2)
        # bn = round(test["boundaries"][bn] * 0.02, 2)
        # offset = int(b0 * 16000)
        # print(b0, bn, bn - b0)
        # frames = int((bn - b0) * 16000)
        aud, sr = torchaudio.load(new_fn, frame_offset=offset, num_frames=frames)
        word_new_fn = (Path('data/yoruba_wavs') / f'{wav_name}_word_{key[id]}').with_suffix('.wav')

        if frames == aud.size(1) and wav_name in labels_to_images[id]:
            torchaudio.save(word_new_fn, aud, sr)
            newly_labeled[id].append(wav_name)
            i += 1
        count += 1

    for wav_name in s_wav_dict[id]:

        new_name = '_'.join(wav_name.split('_')[:-1])
  
        fn = (Path('data/yoruba_utterances_for_support_set') / wav_name).with_suffix('.wav')
        new_fn = (Path('data/yoruba_wavs') / new_name).with_suffix('.wav')
        new_fn.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(fn, new_fn)


        orig_word_fn = (Path('data/yoruba_support_set_words') / f'{wav_name}').with_suffix('.wav')
        word_new_fn = (Path('data/yoruba_wavs'  ) / f'{new_name}_word_{key[id]}').with_suffix('.wav')
        shutil.copy(orig_word_fn, word_new_fn)
        newly_labeled[id].append(new_name)


for id in newly_labeled:
    print(id, len(newly_labeled[id]))

acc = {}
for word in newly_labeled:
    if word not in acc: acc[word] = {'correct': 0, 'total': 0}
    for wav_name in newly_labeled[word]:
        if wav_name in labels_to_images[word]: acc[word]['correct'] += 1
        acc[word]['total'] += 1

for word in acc:
    print(word, acc[word]['correct'], acc[word]['total'], 100 * acc[word]['correct'] / acc[word]['total'])

np.savez_compressed(
    Path("./data/sampled_audio_data"), 
    data=newly_labeled
)