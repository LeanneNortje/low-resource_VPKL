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

BACKEND = "nccl"
INIT_METHOD = "tcp://localhost:54321"

audio_segments_dir = Path('QbERT/utterances_segmented')
audio_query_dir = Path('QbERT/queries_segmented')
audio_dir = Path('/media/leannenortje/HDD/Datasets/flickr_audio/wavs/')
ss_save_fn = 'data/support_set_audio.npz'
support_set = np.load(ss_save_fn, allow_pickle=True)['support_set'].item()
pam = np.load("QbERT/pam.npy")

vocab = []
with open('./data/34_keywords.txt', 'r') as f:
    for keyword in f:
        vocab.append(keyword.strip())

s_audio = {}
s_wavs = []

for word in tqdm(support_set):
    for wav in support_set[word]:
        fn = (audio_query_dir / wav.stem).with_suffix('.npz')
        
        query = np.load(fn)
        x = query["codes"][query["boundaries"][:-1]]

        if word not in s_audio: s_audio[word] = []
        s_audio[word].append(x)
        s_wavs.append(Path(wav).stem)


base = Path('/media/leannenortje/HDD/Datasets/Flickr8k_text')
train_names = set()
for line in open(base / Path('Flickr_8k.trainImages.txt'), 'r'):
    train_names.add(line.split('.')[0])
for line in open(base / Path('Flickr_8k.devImages.txt'), 'r'):
    train_names.add(line.split('.')[0])

data = []

for fn in audio_segments_dir.rglob('*.npz'):
    name = '_'.join(fn.stem.split('_')[0:2])
    if name in train_names: data.append(fn)

query_scores = {}
record = {}

for q_word in s_audio:
    id = q_word
    for wav in tqdm(data, desc=f'{q_word}'):

        wav_name = Path(wav).stem
        if wav_name in s_wavs: continue

        if id not in query_scores: query_scores[id] = {'values': [], 'wavs': []}

        fn = (audio_segments_dir / wav.stem).with_suffix('.npz')
        test = np.load(fn)
        y = test["codes"][test["boundaries"][:-1]]

        max_score = -np.inf
        for x in s_audio[q_word]:
            path, p, q, score = align_semiglobal(x, y, pam, 3)
            indexes, = np.where(np.array(p) != -1)
            if len(indexes) > 2 :
                start, end = indexes[1], indexes[-1]
                norm_score = score / (end - start)

                if norm_score > max_score: 
                    max_score = norm_score
                    if wav not in record: record[wav] = {}
                    record[wav][id] = (path, start, end, p, q, indexes)
        query_scores[id]['values'].append(max_score)
        query_scores[id]['wavs'].append(wav)    
    
        # if len(query_scores[id]['values']) == 100: break

for id in query_scores:
    print(len(query_scores[id]['values']), len(query_scores[id]['wavs']))

save_dir = Path('data/english_segments')
# audio_dir = Path("../../Datasets/spokencoco/SpokenCOCO")
top_N = 600
newly_labeled = {}

for id in tqdm(query_scores):
    indices = np.argsort(query_scores[id]['values'])[::-1]
    i = 0
    while i < top_N:
        wav = Path(query_scores[id]['wavs'][indices[i]])
        
        wav_name = wav.stem
        fn = (audio_dir / wav_name).with_suffix('.wav')
        new_fn = (Path('data/english_wavs') / wav_name).with_suffix('.wav')
        new_fn.parent.mkdir(parents=True, exist_ok=True)

        # segment_fn = wav.relative_to(*wav.parts[:1]).with_suffix('.npz')
        # segment_fn = audio_segments_dir / segment_fn
        # test = np.load(segment_fn)
        # path, start, end, p, q, indexes = record[str(wav)][id]
        # _, b0 = path[start - 1]
        # _, bT = path[end]
        # w0, wT = 0.02 * test["boundaries"][b0 - 1], 0.02 * test["boundaries"][bT]
        # offset = int(w0 * 16000)
        # frames = int(np.abs(wT - w0) * 16000)
        # aud, sr = torchaudio.load(audio_dir / wav, frame_offset=offset, num_frames=frames)
        
        # if frames == aud.size(1):
        
        #     if wav_name == 'm3t1oiftx18474-3TR2532VIPUCJEHB0694KVVG1QFJ6A_266069_594194': print(aud.size(), frames, offset)
        # torchaudio.save(fn.with_suffix('.wav'), aud, sr)
        shutil.copy(fn, new_fn)

        if id not in newly_labeled: newly_labeled[id] = []
        newly_labeled[id].append(wav_name)
        i += 1

for word in newly_labeled:
    print(word, len(newly_labeled[word]))

np.savez_compressed(
    Path("./data/sampled_audio_data"), 
    data=newly_labeled
)