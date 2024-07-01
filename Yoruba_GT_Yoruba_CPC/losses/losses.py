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
from .util import *
    
def compute_matchmap_similarity_matrix_loss(
    image_outputs, english_output, english_nframes, 
    word_output, word_nframes,
    negatives, positives, attention, contrastive_loss, 
    margin, simtype, alphas, rank):
    
    i_e = compute_matchmap_similarity_matrix(image_outputs, None, english_output, english_nframes, attention)
    # i_w = compute_matchmap_similarity_matrix(image_outputs, None, word_output, word_nframes, attention)
    # w_e = compute_matchmap_similarity_matrix(word_output, word_nframes, english_output, english_nframes, attention)
    # e_e = compute_matchmap_similarity_matrix(english_output, english_nframes, english_output, english_nframes, attention)
    # w_w = compute_matchmap_similarity_matrix(word_output, word_nframes, word_output, word_nframes, attention)
    # a = torch.cat([i_e, i_w, w_e, e_e, w_w], dim=1)

    neg_1 = []
    neg_2 = []

    for neg_dict in negatives:
        s = compute_matchmap_similarity_matrix(image_outputs, None, neg_dict["english_output"], neg_dict["english_nframes"], attention)
        neg_1.append(s)
        s = compute_matchmap_similarity_matrix(neg_dict['image'], None, english_output, english_nframes, attention)
        neg_2.append(s)

        # s = compute_matchmap_similarity_matrix(neg_dict['word_output'], neg_dict['word_nframes'], word_output, word_nframes, attention)
        # neg_1.append(s)

        # s = compute_matchmap_similarity_matrix(image_outputs, None, neg_dict["word_output"], neg_dict["word_nframes"], attention)
        # neg_1.append(s)
        # s = compute_matchmap_similarity_matrix(neg_dict['image'], None, word_output, word_nframes, attention)
        # neg_2.append(s)

        # s = compute_matchmap_similarity_matrix(word_output, word_nframes, neg_dict["english_output"], neg_dict["english_nframes"], attention)
        # neg_1.append(s)
        # s = compute_matchmap_similarity_matrix(neg_dict['word_output'], neg_dict['word_nframes'], english_output, english_nframes, attention)
        # neg_2.append(s)

        # s = compute_matchmap_similarity_matrix(english_output, english_nframes, neg_dict['english_output'], neg_dict['english_nframes'], attention)
        # neg_1.append(s)
        # s = compute_matchmap_similarity_matrix(neg_dict['english_output'], neg_dict['english_nframes'], english_output, english_nframes, attention)
        # neg_2.append(s)

    neg_1 = torch.cat(neg_1, dim=1)
    neg_2 = torch.cat(neg_2, dim=1)

    pos_1 = []
    pos_2 = []

    for pos_dict in positives:
        s = compute_matchmap_similarity_matrix(image_outputs, None, pos_dict["english_output"], pos_dict["english_nframes"], attention)
        pos_1.append(s)
        s = compute_matchmap_similarity_matrix(pos_dict['image'], None, english_output, english_nframes, attention)
        pos_2.append(s)

        # s = compute_matchmap_similarity_matrix(pos_dict['word_output'], pos_dict["word_nframes"], word_output, word_nframes, attention)
        # pos_1.append(s)

        # s = compute_matchmap_similarity_matrix(image_outputs, None, pos_dict["word_output"], pos_dict["word_nframes"], attention)
        # pos_1.append(s)
        # s = compute_matchmap_similarity_matrix(pos_dict['image'], None, word_output, word_nframes, attention)
        # pos_2.append(s)

        # s = compute_matchmap_similarity_matrix(word_output, word_nframes, pos_dict["english_output"], pos_dict["english_nframes"], attention)
        # pos_1.append(s)
        # s = compute_matchmap_similarity_matrix(pos_dict['word_output'], pos_dict["word_nframes"], english_output, english_nframes, attention)
        # pos_2.append(s)


        # s = compute_matchmap_similarity_matrix(english_output, english_nframes, pos_dict['english_output'], pos_dict["english_nframes"], attention)
        # pos_1.append(s)
        # s = compute_matchmap_similarity_matrix(pos_dict['english_output'], pos_dict["english_nframes"], english_output, english_nframes, attention)
        # pos_2.append(s)


    pos_1 = torch.cat(pos_1, dim=1)
    pos_2 = torch.cat(pos_2, dim=1)

    loss = contrastive_loss(i_e, pos_1, pos_2, neg_1, neg_2) 
    # loss += contrastive_loss(i_w, pos_1, pos_2, neg_1, neg_2)  

    return loss