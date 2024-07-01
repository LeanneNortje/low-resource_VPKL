#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________

import math
import pickle
import numpy as np
import torch
import torch.nn as nn
from itertools import product
from tqdm import tqdm
from models.util import *

def computeSimilarity(image, audio):
    # function adapted from https://github.com/dharwath

    assert(image.dim() == 3)
    assert(audio.dim() == 2)

    imageH = image.size(1)
    imageW = image.size(2)
    audioFrames = audio.size(1)
                                                                                                                
    Ir = image.view(image.size(0), -1).t()
    matchmap = torch.mm(Ir, audio)
    matchmap = matchmap.view(imageH, imageW, audioFrames)  
    
    return matchmap

def computeSimilarityA2A(audio_1, audio_2):
    # function adapted from https://github.com/dharwath

    assert(audio_1.dim() == 2)
    assert(audio_2.dim() == 2)
                                                                                                
    matchmap = torch.mm(audio_1.t(), audio_2)
    
    return matchmap

def matchmapSim(M, simtype):
    # function adapted from https://github.com/dharwath

    assert(M.dim() == 3)
    if simtype == 'SISA':
        return M.mean()
    elif simtype == 'MISA':
        M_maxH, _ = M.max(0)
        M_maxHW, _ = M_maxH.max(0)
        return M_maxHW.mean()
    elif simtype == 'SIMA':
        M_maxT, _ = M.max(2)
        return M_maxT.mean()
    else:
        raise ValueError

def resize(x):
    if x.dim() == 4: 
        w = x.size(2)
        h = x.size(3)
        x = x.view(x.size(0), x.size(1), -1).transpose(1, 2)
        # return x, w, h
    else: 
        x = x.squeeze(2)
    return x

def compute_matchmap_similarity_matrix(x, x_mask, y, y_mask, args):
    simtype = args["simtype"]
    # device = args["device"]

    x = resize(x)
    y = resize(y)

    assert(x.dim() == 3)
    assert(y.dim() == 3)
    
    n = x.size(0)

    if x_mask is not None:
        for i in range(n):
            x[i, :, y_mask[i]:] = 0

    for i in range(n):

        nth_entrance_in_y = torch.cat(n*[y[i, :, :].unsqueeze(0)])

        if y_mask is not None: nth_entrance_in_y[:, :, y_mask[i]:] = 0
        
        M = torch.bmm(x, nth_entrance_in_y.transpose(1, 2))
        # M = M.view(x.size(0), w, h, -1)#.transpose(1, 2)

        # assert(M.dim() == 4)
        if simtype == 'SISA':
            # M = M.mean((3))
            M = M.mean((2))
            M = M.mean((1))
        elif simtype == 'MISA':
            # M, _ = M.max(1)
            M, _ = M.max(1)
            M = M.mean((1))
        elif simtype == 'SIMA':
            M, _ = M.max(2)
            # M =  M.mean((2))
            M =  M.mean((1))
        else:
            raise ValueError

        if i == 0: S = M.unsqueeze(1)
        else: S = torch.cat([S, M.unsqueeze(1)], dim=1) 
    return S

def myisnan(x):
    return torch.isnan(x).any()

def compute_matchmap_similarity_matrix_IA(im, im_mask, audio, frames, simtype='MISA'):
    
    w = im.size(2)
    h = im.size(3)
    im = im.view(im.size(0), im.size(1), -1).transpose(1, 2)
    audio = audio.squeeze(2)

    assert(im.dim() == 3)
    assert(audio.dim() == 3)
    
    n = im.size(0)

    for i in range(n):
        nth_entrance_in_audio = torch.cat(n*[audio[i, :, 0:frames[i]].unsqueeze(0)])

        M = torch.bmm(im, nth_entrance_in_audio)
        M = M.view(im.size(0), w, h, -1)#.transpose(1, 2)

        assert(M.dim() == 4)
        if simtype == 'SISA':
            M = M.mean((3))
            M = M.mean((2))
            M = M.mean((1))
        elif simtype == 'MISA':
            M, _ = M.max(1)
            M, _ = M.max(1)
            M = M.mean((1))
        elif simtype == 'SIMA':
            M, _ = M.max(3)
            M =  M.mean((2))
            M =  M.mean((1))
        else:
            raise ValueError

        if i == 0: S = M.unsqueeze(1)
        else: S = torch.cat([S, M.unsqueeze(1)], dim=1) 
    return S

def compute_matchmap_similarity_matrix_AA(audio_1, frames_1, audio_2, frames_2, simtype='MISA'):
  
    audio_1 = audio_1.squeeze(2).transpose(1, 2)
    audio_2 = audio_2.squeeze(2)

    assert(audio_1.dim() == 3)
    assert(audio_2.dim() == 3)
    
    n = audio_1.size(0)

    for i in range(n):
        nth_entrance_in_audio_2 = torch.cat(n*[audio_2[i, :, 0:frames_2[i]].unsqueeze(0)])

        M = torch.bmm(audio_1, nth_entrance_in_audio_2)

        assert(M.dim() == 3)
        if simtype == 'SISA' or simtype == 'MISA':
            M = M.mean((2))
            M = M.mean((1))
        elif simtype == 'SIMA':
            M, _ = M.max(2)
            M, _ = M.max((1))
        else:
            raise ValueError

        if i == 0: S = M.unsqueeze(1)
        else: S = torch.cat([S, M.unsqueeze(1)], dim=1) 
    return S

def sampleImposters(positive_index, num_points):
    possible_indices = np.arange(0, num_points)
    possible_indices = np.delete(possible_indices, positive_index)
    audio_imposter = np.random.choice(possible_indices)
    image_imposter = np.random.choice(possible_indices)
    return audio_imposter, image_imposter

def computeSimilarityMatrix(image_outputs, audio_outputs, nframes, simtype='MISA'):
    # function adapted from https://github.com/dharwath

    assert(image_outputs.dim() == 4)
    assert(audio_outputs.dim() == 3)
    n = image_outputs.size(0)
    S = torch.zeros(n, n, device=image_outputs.device)

    for im_i, aud_i in tqdm(product(range(0, n), range(0, n)), leave=False):
        S[im_i, aud_i] = matchmapSim(computeSimilarity(image_outputs[im_i], audio_outputs[aud_i][:, 0:nframes[aud_i]]), simtype)

    return S

def ToPoolOrNotToPool(matrix_1, nframes=None):

    if matrix_1.squeeze().dim() == 4: 
        pool_func_1 = nn.AdaptiveAvgPool2d((1, 1))
        pooled_outputs = pool_func_1(matrix_1).squeeze(3).squeeze(2)
    elif matrix_1.squeeze().dim() == 3 and nframes is not None:
        pool_func_1 = nn.AdaptiveAvgPool2d((1, 1))
        pooled_outputs_list = []
        for idx in range(matrix_1.size(0)):
            nF = max(1, nframes[idx])
            pooled_outputs_list.append(pool_func_1(matrix_1[idx][:, :, 0:nF]).unsqueeze(0))
        pooled_outputs = torch.cat(pooled_outputs_list).squeeze(3).squeeze(2)
        # pooled_outputs = pooled_outputs/torch.norm(pooled_outputs, 2)
    return pooled_outputs

def compute_pooldot_similarity_matrix(matrix_1, mask_1, matrix_2, mask_2):

    assert(matrix_1.dim() == 4)
    assert(matrix_2.dim() == 4)
    n = matrix_1.size(0)
    
    pooled_outputs_1 = ToPoolOrNotToPool(matrix_1, mask_1)
    pooled_outputs_2 = ToPoolOrNotToPool(matrix_2, mask_2)
    S = torch.mm(pooled_outputs_1, pooled_outputs_2.t())

    return S

# def compute_pooldot_similarity_matrix(image_outputs, audio_outputs, nframes):

#     assert(image_outputs.dim() == 4)
#     assert(audio_outputs.dim() == 4)
#     n = image_outputs.size(0)
#     imagePoolfunc = nn.AdaptiveAvgPool2d((1, 1))
#     pooled_image_outputs = imagePoolfunc(image_outputs).squeeze(3).squeeze(2)
#     audioPoolfunc = nn.AdaptiveAvgPool2d((1, 1))
#     pooled_audio_outputs_list = []
#     for idx in range(n):
#         nF = max(1, nframes[idx])
#         pooled_audio_outputs_list.append(audioPoolfunc(audio_outputs[idx][:, :, 0:nF]).unsqueeze(0))
#     pooled_audio_outputs = torch.cat(pooled_audio_outputs_list).squeeze(3).squeeze(2)
#     S = torch.mm(pooled_image_outputs, pooled_audio_outputs.t())

#     return S