#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2022
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as imagemodels
import torch.utils.model_zoo as model_zoo
from torch import Tensor
import numpy as np
import math
from torchvision.io import read_image
from torchvision.models import *

class mutlimodal(nn.Module):
    def __init__(self, args):
        super(mutlimodal, self).__init__()
        num_channels = args["acoustic_model"]["out_channels"]
        z_dim = args["audio_model"]["z_dim"]
        c_dim = args["audio_model"]["c_dim"]

        self.conv = nn.Conv1d(
            args["acoustic_model"]["in_channels"], num_channels, 
            args["acoustic_model"]["kernel_size"], 
            args["acoustic_model"]["stride"], 
            args["acoustic_model"]["padding"], bias=False)

        self.encoder = nn.Sequential(
            nn.LayerNorm(num_channels),
            nn.ReLU(True),
            nn.Linear(num_channels, num_channels, bias=False),
            # nn.InstanceNorm1d(num_channels),

            nn.LayerNorm(num_channels),
            nn.ReLU(True),
            nn.Linear(num_channels, num_channels, bias=False),
            # nn.InstanceNorm1d(num_channels),

            nn.LayerNorm(num_channels),
            nn.ReLU(True),
            nn.Linear(num_channels, num_channels, bias=False),
            # nn.InstanceNorm1d(num_channels),

            nn.LayerNorm(num_channels),
            nn.ReLU(True),
            nn.Linear(num_channels, num_channels, bias=False),
            # nn.InstanceNorm1d(num_channels),

            nn.LayerNorm(num_channels),
            nn.ReLU(True),
            nn.Linear(num_channels, z_dim),
            # nn.InstanceNorm1d(z_dim),
        )
        self.rnn1 = nn.LSTM(z_dim, c_dim, batch_first=True)
        self.rnn2 = nn.LSTM(c_dim, c_dim, batch_first=True)
        self.rnn3 = nn.LSTM(c_dim, c_dim, batch_first=True)
        self.rnn4 = nn.LSTM(c_dim, c_dim, batch_first=True)

        self.english_rnn1 = nn.LSTM(512, 512, batch_first=True, bidirectional=True, bias=False)
        self.english_rnn2 = nn.LSTM(1024, 1024, batch_first=True, bidirectional=True, bias=False)
        # self.english_rnn3 = nn.LSTM(2048, 2048, batch_first=True, bidirectional=False, bias=False)
        # self.bn1 = nn.BatchNorm1d(512)
        # self.bn2 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()
        # self.ln = nn.LayerNorm(512)

    def forward(self, mels):
        z = self.conv(mels)
        z = self.relu(z)
        z = self.encoder(z.transpose(1, 2))

        c, _ = self.rnn1(z)
        c, _ = self.rnn2(c)
        c, _ = self.rnn3(c)
        c, _ = self.rnn4(c)

        s, _ = self.english_rnn1(c)
        # s = self.relu(s)
        s, _ = self.english_rnn2(s)
        s = self.relu(s)
        # s, _ = self.english_rnn3(s)

        return z, z, s.transpose(1, 2)

    def encode(self, mels, feat):
        if feat == 'mels': return mels
        z = self.conv(mels)
        z = self.relu(z)
        if feat == 'pre_z': return z
        z = self.encoder(z.transpose(1, 2))
        if feat == 'z': return z

        c, _ = self.rnn1(z)
        if feat == 'c1': return z
        c, _ = self.rnn2(c)
        if feat == 'c2': return z
        c, _ = self.rnn3(c)
        if feat == 'c3': return z
        c, _ = self.rnn4(c)
        if feat == 'c4': return z

        s, _ = self.english_rnn1(c)
        # s = self.relu(s)
        if feat == 's1': return s
        s, _ = self.english_rnn2(s)
        # s = self.relu(s)
        if feat == 's2': return s


class vision(nn.Module):
    def __init__(self, args):
        super(vision, self).__init__()
        seed_model = alexnet(pretrained=args['pretrained_alexnet'])
        self.image_model = nn.Sequential(*list(seed_model.features.children()))

        last_layer_index = len(list(self.image_model.children()))
        self.image_model.add_module(str(last_layer_index),
            nn.Conv2d(256, args["audio_model"]["embedding_dim"], kernel_size=(3,3), stride=(1,1), padding=(1,1)))
    
        # self.image_encoder = nn.Sequential(
        #     # nn.LayerNorm(49),
        #     nn.Linear(49, 7),
        #     nn.LeakyReLU(),
        #     # nn.LayerNorm(49),
        #     nn.Linear(7, 1),
        #     nn.LeakyReLU()
        # )
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.image_model(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = self.relu(x)
        return x