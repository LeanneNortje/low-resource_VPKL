#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________

import json
import random
import numpy as np
from models.setup import *
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import scipy
import scipy.signal
import librosa
from tqdm import tqdm
sys.path.append("..")
# from preprocessing.audio_preprocessing import extract_features
import warnings
warnings.filterwarnings("ignore")

scipy_windows = {
    'hamming': scipy.signal.hamming,
    'hann': scipy.signal.hann, 
    'blackman': scipy.signal.blackman,
    'bartlett': scipy.signal.bartlett
    }

def preemphasis(signal,coeff=0.97):  
    # function adapted from https://github.com/dharwath
    
    return np.append(signal[0],signal[1:]-coeff*signal[:-1])

class ImageAudioData(Dataset):
    def __init__(self, key, args):

        with open(args[key], 'r') as fp:
            data = json.load(fp)

        print(f'\rRead in data paths from:')
        printDirectory(args['data_train'])

        print(f'\n\rRead in {len(data)} data points')
        
        self.audio_conf = args["audio_config"]
        self.target_length = self.audio_conf.get('target_length', 1024)
        self.n_sample_frames = args["cpc"]["n_sample_frames"]
        self.n_utterances_per_speaker = args["cpc"]["n_utterances_per_speaker"]
        self.n_speakers_per_batch = args["cpc"]["n_speakers_per_batch"]
        self.num_mel_bins = args["audio_config"]["num_mel_bins"]

        self.speakers = []
        for fn in data:
            name = Path(fn).stem
            speaker = name.split('_')[-2].split('-')[0]
            if speaker not in self.speakers: self.speakers.append(speaker)

        metadata_by_speaker = dict()
        for fn in data:
            fn = Path(fn)
            name = Path(fn).stem
            speaker = name.split('_')[-2].split('-')[0]
            dataset = name.split('_')[0]
            # if dataset == "ENGLISH" or dataset == "HINDI":
            # if dataset == "LIBRISPEECH": 
            metadata_by_speaker.setdefault(speaker, []).append(fn)
        self.metadata = [
            (k, v) for k, v in metadata_by_speaker.items()
            if len(v) >= self.n_utterances_per_speaker]

    def _SampleMultipleSpeakersFeat(self, paths):
        
        mels = list()
        paths = random.sample(paths, self.n_utterances_per_speaker)
        for path in paths:
            path = path.with_suffix(".npz")
            mel = np.load(path)['audio_feat']
            pos = random.randint(0, mel.shape[1] - self.n_sample_frames)
            mel = mel[:, pos:pos + self.n_sample_frames]
            mels.append(mel)
        mels = np.stack(mels)

        return torch.from_numpy(mels)

    def __getitem__(self, index):
        speaker, paths = self.metadata[index]

        audio_feat = self._SampleMultipleSpeakersFeat(paths)
                
        return audio_feat, self.speakers.index(speaker)

    def __len__(self):
        return len(self.metadata)

# class ImageAudioData(Dataset):
#     def __init__(self, key, args):

#         with open(args[key], 'r') as fp:
#             data = json.load(fp)

#         print(f'\rRead in data paths from:')
#         printDirectory(args['data_train'])

#         print(f'\n\rRead in {len(data)} data points')
        
#         self.audio_conf = args["audio_config"]
#         self.target_length = self.audio_conf.get('target_length', 1024)
#         self.n_sample_frames = args["cpc"]["n_sample_frames"]
#         self.n_utterances_per_speaker = args["cpc"]["n_utterances_per_speaker"]
#         self.n_speakers_per_batch = args["cpc"]["n_speakers_per_batch"]
#         self.num_mel_bins = args["audio_config"]["num_mel_bins"]

#         # filtered_data = []

#         # for fn in tqdm(data, desc="Filtering data"):
#         #     data_point = np.load(fn + ".npz")

#         #     if 'audio_feat' in data_point:

#         #         if data_point["audio_feat"].shape[1] >= self.n_sample_frames:
#         #             filtered_data.append(fn)

#         #     else:

#         #         if data_point["eng_audio_feat"].shape[1] >= self.n_sample_frames and data_point["hindi_audio_feat"].shape[1] >= self.n_sample_frames:
#                     # filtered_data.append(fn)
            
#         self.data = data

#     def _PadFeat(self, feat):
#         nframes = feat.shape[1]
#         pad = self.target_length - nframes
        
#         if pad > 0:
#             feat = np.pad(feat, ((0, 0), (0, pad)), 'constant',
#                 constant_values=(self.padval, self.padval))
#         elif pad < 0:
#             nframes = self.target_length
#             feat = feat[:, 0: pad]

#         return feat, nframes

#     def _SampleFeat(self, feat):
        
#         pos = random.randint(0, feat.shape[1] - self.n_sample_frames)
#         feat = feat[:, pos:pos + self.n_sample_frames]

#         return feat, self.n_sample_frames

#     def _SampleMultipleSpeakersFeat(self, speakers, paths, key):
        
#         mels = list()
#         paths = random.sample(paths, self.n_utterances_per_speaker)
#         for path in paths:
#             path = path  + ".npz"
#             mel = np.load(path)[key]
#             pos = random.randint(0, mel.shape[1] - self.n_sample_frames)
#             mel = mel[:, pos:pos + self.n_sample_frames]
#             mels.append(mel)
#         mels = np.stack(mels)

#         return mels, self.n_sample_frames

#     def __getitem__(self, index):

#         data_point = np.load(self.data[index] + ".npz")
#         audio_feat = data_point["audio_feat"]
#         audio_feat, nframes = self._SampleFeat(audio_feat)
        
#         return audio_feat, nframes

#     def __len__(self):
#         # return max(len(self.english_metadata), len(self.hindi_metadata))
#         return len(self.data)

class CPCFilteredAudioData(Dataset):
    def __init__(self, key, args):

        with open(args[key], 'r') as fp:
            data = json.load(fp)

        print(f'\rRead in data paths from:')
        printDirectory(args['data_train'])

        print(f'\n\rRead in {len(data)} data points')
        
        self.audio_conf = args["audio_config"]
        self.target_length = self.audio_conf.get('target_length', 1024)
        self.n_sample_frames = args["cpc"]["n_sample_frames"]
        self.n_utterances_per_speaker = args["cpc"]["n_utterances_per_speaker"]
        self.n_speakers_per_batch = args["cpc"]["n_speakers_per_batch"]
        self.num_mel_bins = args["audio_config"]["num_mel_bins"]


        # filtered_data = []

        # for fn in tqdm(data, desc="Filtering data"):
        #     data_point = np.load(fn + ".npz")

        #     if 'audio_feat' in data_point:

        #         if data_point["audio_feat"].shape[1] >= self.n_sample_frames:
        #             filtered_data.append(fn)

        #     else:

        #         if data_point["eng_audio_feat"].shape[1] >= self.n_sample_frames and data_point["hindi_audio_feat"].shape[1] >= self.n_sample_frames:
        #             filtered_data.append(fn)
            
        self.data = data

    def __getitem__(self, index):
        data_point = np.load(self.data[index] + ".npz")
        if data_point["audio_feat"].shape[1] >= self.n_sample_frames: return self.data[index]
        else: return " " 

    def __len__(self):
        # return max(len(self.english_metadata), len(self.hindi_metadata))
        return len(self.data)

class AudioData(Dataset):
    def __init__(self, root, args):

        self.root = root

        self.data = list(self.root.rglob("*.wav"))

        print(f'\n\rRead in {len(self.data)} data points')
        
        self.audio_conf = args["audio_config"]
        self.target_length = self.audio_conf.get('target_length', 1024)
        self.padval = self.audio_conf.get('padval', 0)
    
    def _LoadAudio(self, path):

        audio_type = self.audio_conf.get('audio_type')
        if audio_type not in ['melspectrogram', 'spectrogram']:
            raise ValueError('Invalid audio_type specified in audio_conf. Must be one of [melspectrogram, spectrogram]')
        
        preemph_coef = self.audio_conf.get('preemph_coef')
        sample_rate = self.audio_conf.get('sample_rate')
        window_size = self.audio_conf.get('window_size')
        window_stride = self.audio_conf.get('window_stride')
        window_type = self.audio_conf.get('window_type')
        num_mel_bins = self.audio_conf.get('num_mel_bins')
        target_length = self.audio_conf.get('target_length')
        fmin = self.audio_conf.get('fmin')
        n_fft = self.audio_conf.get('n_fft', int(sample_rate * window_size))
        win_length = int(sample_rate * window_size)
        hop_length = int(sample_rate * window_stride)

        # load audio, subtract DC, preemphasis
        y, sr = librosa.load(path, sample_rate)
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
        return torch.tensor(logspec)#, n_frames

    def _PadFeat(self, feat):
        nframes = feat.shape[1]
        pad = self.target_length - nframes
        
        if pad > 0:
            feat = np.pad(feat, ((0, 0), (0, pad)), 'constant',
                constant_values=(self.padval, self.padval))
        elif pad < 0:
            nframes = self.target_length
            feat = feat[:, 0: pad]

        return feat, nframes

    def __getitem__(self, index):

        data_path = self.data[index].with_suffix(".wav")

        audio_feat = self._LoadAudio(data_path)
        # audio_feat, nframes = self._PadFeat(audio_feat)
        return audio_feat, str(self.data[index].with_suffix(''))

    def __len__(self):
        return len(self.data)

class ImageCaptionDatasetWithPreprocessing(Dataset):
    def __init__(self, dataset_json_file, audio_conf=None, image_conf=None, add="train"):
        """
        Dataset that manages a set of paired images and audio recordings

        :param dataset_json_file
        :param audio_conf: Dictionary containing the sample rate, window and
        the window length/stride in seconds, and normalization to perform (optional)
        :param image_transform: torchvision transform to apply to the images (optional)
        """

        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)
        data = data_json['data']
        self.image_base_path = data_json['image_base_path']
        self.audio_base_path = data_json['audio_base_path']

        # if add == "train":
        #     with open("train_test.json", "r") as outfile:
        #         paths = json.load(outfile)

        #     self.data = []
        #     for entry in data:
        #         test = f'data/PlacesAudio_400k_distro+imagesPlaces205_resize/speaker_{entry["speaker"]}+{entry["wav"].split("/")[-1].split(".")[0]}+{entry["image"].split("/")[-1].split(".")[0]}'
        #         if test in paths: self.data.append(entry)
        #     print(len(self.data))
        # else:
        self.data = data
        
        if not audio_conf:
            self.audio_conf = {}
        else:
            self.audio_conf = audio_conf

        if not image_conf:
            self.image_conf = {}
        else:
            self.image_conf = image_conf

        crop_size = self.image_conf.get('crop_size', 224)
        center_crop = self.image_conf.get('center_crop', False)

        if center_crop:
            self.image_resize_and_crop = transforms.Compose(
                [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
        else:
            self.image_resize_and_crop = transforms.Compose(
                [transforms.RandomResizedCrop(crop_size), transforms.ToTensor()])

        RGB_mean = self.image_conf.get('RGB_mean', [0.485, 0.456, 0.406])
        RGB_std = self.image_conf.get('RGB_std', [0.229, 0.224, 0.225])
        self.image_normalize = transforms.Normalize(mean=RGB_mean, std=RGB_std)

        self.windows = {'hamming': scipy.signal.hamming,
        'hann': scipy.signal.hann, 'blackman': scipy.signal.blackman,
           'bartlett': scipy.signal.bartlett}

    def _LoadAudio(self, path):
        audio_type = self.audio_conf.get('audio_type', 'melspectrogram')
        if audio_type not in ['melspectrogram', 'spectrogram']:
            raise ValueError('Invalid audio_type specified in audio_conf. Must be one of [melspectrogram, spectrogram]')
        preemph_coef = self.audio_conf.get('preemph_coef', 0.97)
        sample_rate = self.audio_conf.get('sample_rate', 16000)
        window_size = self.audio_conf.get('window_size', 0.025)
        window_stride = self.audio_conf.get('window_stride', 0.01)
        window_type = self.audio_conf.get('window_type', 'hamming')
        num_mel_bins = self.audio_conf.get('num_mel_bins', 40)
        target_length = self.audio_conf.get('target_length', 1024)
        use_raw_length = self.audio_conf.get('use_raw_length', False)
        padval = self.audio_conf.get('padval', 0)
        fmin = self.audio_conf.get('fmin', 20)
        n_fft = self.audio_conf.get('n_fft', int(sample_rate * window_size))
        win_length = int(sample_rate * window_size)
        hop_length = int(sample_rate * window_stride)

        # load audio, subtract DC, preemphasis
        y, sr = librosa.load(path, sample_rate)
        if y.size == 0:
            y = np.zeros(200)
        y = y - y.mean()
        y = preemphasis(y, preemph_coef)
        # compute mel spectrogram
        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
            win_length=win_length,
            window=self.windows.get(window_type, self.windows['hamming']))
        spec = np.abs(stft)**2
        if audio_type == 'melspectrogram':
            mel_basis = librosa.filters.mel(sr, n_fft, n_mels=num_mel_bins, fmin=fmin)
            melspec = np.dot(mel_basis, spec)
            logspec = librosa.power_to_db(melspec, ref=np.max)
        elif audio_type == 'spectrogram':
            logspec = librosa.power_to_db(spec, ref=np.max)
        n_frames = logspec.shape[1]
        if use_raw_length:
            target_length = n_frames
        p = target_length - n_frames
        if p > 0:
            logspec = np.pad(logspec, ((0,0),(0,p)), 'constant',
                constant_values=(padval,padval))
        elif p < 0:
            logspec = logspec[:,0:p]
            n_frames = target_length
        logspec = torch.FloatTensor(logspec)
        return logspec, n_frames

    def _LoadImage(self, impath):
        img = Image.open(impath).convert('RGB')
        img = self.image_resize_and_crop(img)
        img = self.image_normalize(img)
        return img

    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """
        datum = self.data[index]
        wavpath = os.path.join(self.audio_base_path, datum['wav'])
        imgpath = os.path.join(self.image_base_path, datum['image'])
        audio, nframes = self._LoadAudio(wavpath)
        image = self._LoadImage(imgpath)
        return image, audio, nframes

    def __len__(self):
        return len(self.data)

def get_english_speaker(entry):
    ID = entry.strip().split("/")[-1]
    return ID.split("+")[0].strip("_")[1].strip("-")[0]

def get_hindi_speaker(entry):
    ID = entry.strip().split("/")[-1]
    return ID.split("+")[1].strip("_")[1].strip("-")[0]

class ImageAudioDataWithCPC(Dataset):
    def __init__(self, image_base_path, dataset_json_file):

        with open(dataset_json_file, 'r') as fp:
            data = json.load(fp)
        self.image_base_path = Path(image_base_path).absolute()

        print(f'\n\rRead in {len(data)} data points')
        
        self.audio_conf = {
            "audio_type": "melspectrogram",
            "preemph_coef": 0.97,
            "sample_rate": 16000,
            "window_size": 0.025,
            "window_stride": 0.01,
            "window_type": "hamming",
            "num_mel_bins": 40,
            "target_length": 1024,
            "use_raw_length": False,
            "padval": 0,
            "fmin": 20
        }
        self.target_length = self.audio_conf.get('target_length', 1024)
        self.padval = self.audio_conf.get('padval', 0)
        self.image_conf = {
            "crop_size": 224,
            "center_crop": False,
            "RGB_mean": [0.485, 0.456, 0.406],
            "RGB_std": [0.229, 0.224, 0.225]
        }
        crop_size = self.image_conf.get('crop_size')
        center_crop = self.image_conf.get('center_crop')
        RGB_mean = self.image_conf.get('RGB_mean')
        RGB_std = self.image_conf.get('RGB_std')

        if center_crop:
            self.image_resize_and_crop = transforms.Compose(
                [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
        else:
            self.image_resize_and_crop = transforms.Compose(
                [transforms.RandomResizedCrop(crop_size), transforms.ToTensor()])

        self.image_normalize = transforms.Normalize(mean=RGB_mean, std=RGB_std)

        self.n_sample_frames = 128 + 12
        self.n_utterances_per_speaker = 8

        self.english_speakers = []
        self.hindi_speakers = []

        for entry in data:
            english_speaker = get_english_speaker(entry)
            hindi_speaker = get_hindi_speaker(entry)
            self.english_speakers.append(english_speaker)
            self.hindi_speakers.append(hindi_speaker)
        self.english_speakers = sorted(self.english_speakers)
        self.hindi_speakers = sorted(self.hindi_speakers)

        metadata_by_speaker = dict()
        for entry in data:
            english_speaker = get_english_speaker(entry)
            hindi_speaker = get_hindi_speaker(entry)
            metadata_by_speaker.setdefault(english_speaker, []).append(entry)
            metadata_by_speaker.setdefault(hindi_speaker, []).append(entry)

        self.metadata = []
        for entry in data:
            english_speaker = get_english_speaker(entry)
            hindi_speaker = get_hindi_speaker(entry)
            if len(metadata_by_speaker[english_speaker]) >= self.n_utterances_per_speaker and len(metadata_by_speaker[hindi_speaker]) >= self.n_utterances_per_speaker:
                self.metadata.append((entry, metadata_by_speaker[english_speaker], metadata_by_speaker[hindi_speaker]))

    def _LoadImage(self, impath):
        img = Image.open(impath).convert('RGB')
        img = self.image_resize_and_crop(img)
        img = self.image_normalize(img)
        return img

    def _PadFeat(self, feat):
        nframes = feat.shape[1]
        pad = self.target_length - nframes
        
        if pad > 0:
            feat = np.pad(feat, ((0, 0), (0, pad)), 'constant',
                constant_values=(self.padval, self.padval))
        elif pad < 0:
            nframes = self.target_length
            feat = feat[:, 0: pad]

        return feat, nframes

    def _sample(self, pos_feat, pos_frames, pos_path, paths, key):

        mels = list()
        
        mel, nframes = self._PadFeat(pos_feat)
        pos = random.randint(0, pos_frames - self.n_sample_frames)
        mel = mel[:, pos:pos + self.n_sample_frames]
        mels.append(mel)
        
        temp = paths.copy()
        temp.remove(pos_path)
        
        paths = random.sample(paths, self.n_utterances_per_speaker-1)
        for path in temp:
            data_point = np.load(data_path + ".npz")
            mel, nframes = self._PadFeat(data_point[key])
            pos = random.randint(0, nframes - self.n_sample_frames)
            mel = mel[:, pos:pos + self.n_sample_frames]
            mels.append(mel)
        mels = np.stack(mels)

    def __getitem__(self, index):

        data_path, english_points, hindi_points = self.metadata[index]

        data_point = np.load(data_path + ".npz")

        english_audio_feat = data_point["eng_audio_feat"]
        english_audio_feat, english_nframes = self._PadFeat(english_audio_feat)
        english_cpc = self._sample(english_audio_feat, english_nframes, data_path, english_points, "eng_audio_feat")

        hindi_audio_feat = data_point["hindi_audio_feat"]
        hindi_audio_feat, hindi_nframes = self._PadFeat(hindi_audio_feat)
        hindi_cpc = self._sample(hindi_audio_feat, hindi_nframes, data_path, hindi_points, "hindi_audio_feat")
        
        # imgpath = self.image_base_path / str(data_point['image'])
        # image = self._LoadImage(imgpath)
        # return image, english_audio_feat, english_nframes, hindi_audio_feat, hindi_nframes, 
        return english_cpc, hindi_cpc

    def __len__(self):
        return len(self.metadata)