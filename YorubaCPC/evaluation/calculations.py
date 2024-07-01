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
import sys
import os
from losses.util import *
import warnings
warnings.filterwarnings("ignore")

def calc_recalls(A, A_mask, B, B_mask):
    # function adapted from https://github.com/dharwath

    S = compute_pooldot_similarity_matrix(A, A_mask, B, B_mask)
    n = S.size(0)
    A2B_scores, A2B_ind = S.topk(10, 1)
    B2A_scores, B2A_ind = S.topk(10, 0)

    A2B_scores = A2B_scores.detach().cpu().numpy()
    A2B_ind = A2B_ind.detach().cpu().numpy()
    B2A_scores = B2A_scores.detach().cpu().numpy()
    B2A_ind = B2A_ind.detach().cpu().numpy()

    A_foundind = -np.ones(n)
    B_foundind = -np.ones(n)
    for i in range(n):
        ind = np.where(A2B_ind[i, :] == i)[0]
        if len(ind) != 0: B_foundind[i] = ind[0]
        ind = np.where(B2A_ind[:, i] == i)[0]
        if len(ind) != 0: A_foundind[i] = ind[0]
 
    r1_A_to_B = len(np.where(B_foundind == 0)[0])/len(B_foundind)
    r5_A_to_B = len(np.where(np.logical_and(B_foundind >= 0, B_foundind < 5))[0])/len(B_foundind)
    r10_A_to_B = len(np.where(np.logical_and(B_foundind >= 0, B_foundind < 10))[0])/len(B_foundind)

    r1_B_to_A = len(np.where(A_foundind == 0)[0])/len(A_foundind)
    r5_B_to_A = len(np.where(np.logical_and(A_foundind >= 0, A_foundind < 5))[0])/len(A_foundind)
    r10_B_to_A = len(np.where(np.logical_and(A_foundind >= 0, A_foundind < 10))[0])/len(A_foundind)

    return {
        'A_to_B_r1':r1_A_to_B, 
        'A_to_B_r5':r5_A_to_B, 
        'A_to_B_r10':r10_A_to_B,
        'B_to_A_r1':r1_B_to_A, 
        'B_to_A_r5':r5_B_to_A, 
        'B_to_A_r10':r10_B_to_A
        }

def calc_recalls_IA(A, A_mask, B, B_mask, simtype='MISA'):
    # function adapted from https://github.com/dharwath

    S = compute_matchmap_similarity_matrix_IA(A, A_mask, B, B_mask, simtype)
    n = S.size(0)
    A2B_scores, A2B_ind = S.topk(10, 1)
    B2A_scores, B2A_ind = S.topk(10, 0)

    A2B_scores = A2B_scores.detach().cpu().numpy()
    A2B_ind = A2B_ind.detach().cpu().numpy()
    B2A_scores = B2A_scores.detach().cpu().numpy()
    B2A_ind = B2A_ind.detach().cpu().numpy()

    A_foundind = -np.ones(n)
    B_foundind = -np.ones(n)
    for i in tqdm(range(n), desc="Calculating recalls", leave=False):
        ind = np.where(A2B_ind[i, :] == i)[0]
        if len(ind) != 0: B_foundind[i] = ind[0]
        ind = np.where(B2A_ind[:, i] == i)[0]
        if len(ind) != 0: A_foundind[i] = ind[0]
 
    r1_A_to_B = len(np.where(B_foundind == 0)[0])/len(B_foundind)
    r5_A_to_B = len(np.where(np.logical_and(B_foundind >= 0, B_foundind < 5))[0])/len(B_foundind)
    r10_A_to_B = len(np.where(np.logical_and(B_foundind >= 0, B_foundind < 10))[0])/len(B_foundind)

    r1_B_to_A = len(np.where(A_foundind == 0)[0])/len(A_foundind)
    r5_B_to_A = len(np.where(np.logical_and(A_foundind >= 0, A_foundind < 5))[0])/len(A_foundind)
    r10_B_to_A = len(np.where(np.logical_and(A_foundind >= 0, A_foundind < 10))[0])/len(A_foundind)

    return {
        'A_to_B_r1':r1_A_to_B, 
        'A_to_B_r5':r5_A_to_B, 
        'A_to_B_r10':r10_A_to_B,
        'B_to_A_r1':r1_B_to_A, 
        'B_to_A_r5':r5_B_to_A, 
        'B_to_A_r10':r10_B_to_A
        }

def calc_recalls_AA(A, A_mask, B, B_mask, simtype='MISA'):
    # function adapted from https://github.com/dharwath

    S = compute_matchmap_similarity_matrix_AA(A, A_mask, B, B_mask, simtype)
    n = S.size(0)
    A2B_scores, A2B_ind = S.topk(10, 1)
    B2A_scores, B2A_ind = S.topk(10, 0)

    A2B_scores = A2B_scores.detach().cpu().numpy()
    A2B_ind = A2B_ind.detach().cpu().numpy()
    B2A_scores = B2A_scores.detach().cpu().numpy()
    B2A_ind = B2A_ind.detach().cpu().numpy()

    A_foundind = -np.ones(n)
    B_foundind = -np.ones(n)
    for i in range(n):
        ind = np.where(A2B_ind[i, :] == i)[0]
        if len(ind) != 0: B_foundind[i] = ind[0]
        ind = np.where(B2A_ind[:, i] == i)[0]
        if len(ind) != 0: A_foundind[i] = ind[0]
 
    r1_A_to_B = len(np.where(B_foundind == 0)[0])/len(B_foundind)
    r5_A_to_B = len(np.where(np.logical_and(B_foundind >= 0, B_foundind < 5))[0])/len(B_foundind)
    r10_A_to_B = len(np.where(np.logical_and(B_foundind >= 0, B_foundind < 10))[0])/len(B_foundind)

    r1_B_to_A = len(np.where(A_foundind == 0)[0])/len(A_foundind)
    r5_B_to_A = len(np.where(np.logical_and(A_foundind >= 0, A_foundind < 5))[0])/len(A_foundind)
    r10_B_to_A = len(np.where(np.logical_and(A_foundind >= 0, A_foundind < 10))[0])/len(A_foundind)

    return {
        'A_to_B_r1':r1_A_to_B, 
        'A_to_B_r5':r5_A_to_B, 
        'A_to_B_r10':r10_A_to_B,
        'B_to_A_r1':r1_B_to_A, 
        'B_to_A_r5':r5_B_to_A, 
        'B_to_A_r10':r10_B_to_A
        }