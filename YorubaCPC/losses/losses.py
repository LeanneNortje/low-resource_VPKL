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
import torch.nn.functional as F
from .util import *
    
def marginRankLoss(image_outputs, audio_outputs, nframes, margin=1., simtype='MISA'):
    # function adapted from https://github.com/dharwath

    assert(image_outputs.dim() == 4)
    assert(audio_outputs.dim() == 3)
    n = image_outputs.size(0)
    margin = torch.tensor(margin, device=image_outputs.device, requires_grad=False)
    zero = torch.tensor(0, device=image_outputs.device, requires_grad=False)
    loss = torch.zeros(1, device=image_outputs.device, requires_grad=True)
    for i in range(n):
        audio_imposter_i, image_imposter_i = sampleImposters(i, n)
        nF = nframes[i]
        nFimp = nframes[audio_imposter_i]
        similarityI_A = matchmapSim(computeSimilarity(image_outputs[i], audio_outputs[i][:, 0:nF]), simtype)
        similarityIimp_A = matchmapSim(computeSimilarity(image_outputs[image_imposter_i], audio_outputs[i][:, 0:nF]), simtype)
        similarityI_Aimp = matchmapSim(computeSimilarity(image_outputs[i], audio_outputs[audio_imposter_i][:, 0:nFimp]), simtype)
        
        loss = torch.add(loss, torch.max(zero, margin + similarityIimp_A - similarityI_A))
        loss = torch.add(loss, torch.max(zero, margin + similarityI_Aimp - similarityI_A))
    loss = loss / n
    return loss

def sampled_triplet_loss_from_S(S, margin):

    assert(S.dim() == 2)
    assert(S.size(0) == S.size(1))
    N = S.size(0)
    # S = S / ((S.max(dim=1)).values) ####
    positive_scores = S.diag()
    imp_indices = np.random.randint(0, N-1, size=N)
    ind_to_change = np.where(imp_indices[0:-1] >= np.arange(0, N-1))[0]
    imp_indices[ind_to_change] += 1
    # for j, ind in enumerate(imp_indices):
    #     if ind >= j:
    #         imp_indices[j] = ind + 1
    imposter_scores = S[range(N), imp_indices]
    loss = (imposter_scores - positive_scores + margin).clamp(min=0).mean()
    return loss

def semihardneg_triplet_loss_from_S(S, margin):

    assert(S.dim() == 2)
    assert(S.size(0) == S.size(1))
    
    # S = S / ((S.max(dim=1)).values)
    sampled_loss = sampled_triplet_loss_from_S(S, margin)
    N = S.size(0) 
    positive_scores = S.diag()
    mask = ((S - S.diag().view(-1,1)) < 0).float().detach()
    imposter_scores = (S * mask).max(dim=1).values
    loss = (imposter_scores - positive_scores + margin).clamp(min=0).mean()

    # loss = torch.max(zero, margin + imposter_scores - positive_scores)
    return loss + sampled_loss

def compute_pooldot_similarity_matrix_loss(image_outputs, english_output, hindi_output, english_nframes, hindi_nframes, margin):
    
    loss = 0

    func = sampled_triplet_loss_from_S

    if english_output is not None and english_nframes is not None:
        S = compute_pooldot_similarity_matrix(image_outputs, None, english_output, english_nframes)
        I2E_sampled_loss = func(S, margin)
        E2I_sampled_loss = func(S.t(), margin)
        loss += (I2E_sampled_loss + E2I_sampled_loss)

    if hindi_output is not None and hindi_nframes is not None:
        S = compute_pooldot_similarity_matrix(image_outputs, None, hindi_output, hindi_nframes)
        I2H_sampled_loss = func(S, margin)
        H2I_sampled_loss = func(S.t(), margin)
        loss += I2H_sampled_loss + H2I_sampled_loss

    if english_output is not None and english_nframes is not None and hindi_output is not None and hindi_nframes is not None:
        S = compute_pooldot_similarity_matrix(english_output, english_nframes, hindi_output, hindi_nframes)
        E2H_sampled_loss = func(S, margin)
        H2E_sampled_loss = func(S.t(), margin)
        loss += (E2H_sampled_loss + H2E_sampled_loss)

    return loss

# def compute_matchmap_similarity_matrix_loss(image_outputs, english_output, hindi_output, english_nframes, hindi_nframes, margin, simtype):

#     loss = 0

#     func = sampled_triplet_loss_from_S

#     if english_output is not None and english_nframes is not None:
#         S = compute_matchmap_similarity_matrix_IA(image_outputs, None, english_output, english_nframes, simtype)
#         I2E_sampled_loss = func(S, margin)
#         E2I_sampled_loss = func(S.t(), margin)
#         loss += (I2E_sampled_loss + E2I_sampled_loss)

#     if hindi_output is not None and hindi_nframes is not None:
#         S = compute_matchmap_similarity_matrix_IA(image_outputs, None, hindi_output, hindi_nframes, simtype)
#         I2H_sampled_loss = func(S, margin)
#         H2I_sampled_loss = func(S.t(), margin)
#         loss += (I2H_sampled_loss + H2I_sampled_loss)

#     if english_output is not None and english_nframes is not None and hindi_output is not None and hindi_nframes is not None:
#         S = compute_matchmap_similarity_matrix_AA(english_output, english_nframes, hindi_output, hindi_nframes, simtype)
#         E2H_sampled_loss = func(S, margin)
#         H2E_sampled_loss = func(S.t(), margin)
#         loss += 2*(E2H_sampled_loss + H2E_sampled_loss)

#     return loss

def compute_matchmap_similarity_matrix_loss(image_outputs, english_output, hindi_output, english_nframes, hindi_nframes, margin, simtype, alphas):

    loss = 0
    alpha_1, alpha_2, alpha_3, alpha_4, alpha_5, alpha_6 = alphas

    func = sampled_triplet_loss_from_S

    if english_output is not None and english_nframes is not None:
        S = compute_matchmap_similarity_matrix_IA(image_outputs, None, english_output, english_nframes, simtype)
        I2E_sampled_loss = func(S, margin)
        E2I_sampled_loss = func(S.t(), margin)
        loss += ((alpha_1*I2E_sampled_loss) + (alpha_2*E2I_sampled_loss))

    if hindi_output is not None and hindi_nframes is not None:
        S = compute_matchmap_similarity_matrix_IA(image_outputs, None, hindi_output, hindi_nframes, simtype)
        I2H_sampled_loss = func(S, margin)
        H2I_sampled_loss = func(S.t(), margin)
        loss += ((alpha_3*I2H_sampled_loss) + (alpha_4*H2I_sampled_loss))

    if english_output is not None and english_nframes is not None and hindi_output is not None and hindi_nframes is not None:
        S = compute_matchmap_similarity_matrix_AA(english_output, english_nframes, hindi_output, hindi_nframes, simtype)
        E2H_sampled_loss = func(S, margin)
        H2E_sampled_loss = func(S.t(), margin)
        loss += ((alpha_5*E2H_sampled_loss) + (alpha_6*H2E_sampled_loss))

    return loss


class CPCLoss(nn.Module):
    def __init__(self, args):
        super(CPCLoss, self).__init__()

        self.n_speakers_per_batch = args["cpc"]["n_speakers_per_batch"]
        self.n_utterances_per_speaker = args["cpc"]["n_utterances_per_speaker"]

        self.n_prediction_steps = args["cpc"]["n_prediction_steps"]
        self.n_negatives = args["cpc"]["n_negatives"]
        self.z_dim = args["audio_model"]["z_dim"]
        self.c_dim = args["audio_model"]["c_dim"]
        self.predictors = nn.ModuleList([
            nn.Linear(self.c_dim, self.z_dim) for _ in range(self.n_prediction_steps)
        ])

    def forward(self, z, c, nframes):
        length = z.size(1) - self.n_prediction_steps

        z = z.reshape(
            self.n_speakers_per_batch,
            self.n_utterances_per_speaker,
            -1,
            self.z_dim
        )
        c = c[:, :-self.n_prediction_steps, :]

        losses, accuracies = list(), list()
        for k in range(1, self.n_prediction_steps+1):
            z_shift = z[:, :, k:length + k, :]

            Wc = self.predictors[k-1](c)
            Wc = Wc.view(
                self.n_speakers_per_batch,
                self.n_utterances_per_speaker,
                -1,
                self.z_dim
            )

            batch_index = torch.randint(
                0, self.n_utterances_per_speaker,
                size=(
                    self.n_utterances_per_speaker,
                    self.n_negatives
                ),
                device=z.device
            )
            batch_index = batch_index.view(
                1, self.n_utterances_per_speaker, self.n_negatives, 1
            )

            seq_index = torch.randint(
                1, length,
                size=(
                    self.n_speakers_per_batch,
                    self.n_utterances_per_speaker,
                    self.n_negatives,
                    length
                ),
                device=z.device
            )
            seq_index += torch.arange(length, device=z.device)
            seq_index = torch.remainder(seq_index, length)

            speaker_index = torch.arange(self.n_speakers_per_batch, device=z.device)
            speaker_index = speaker_index.view(-1, 1, 1, 1)

            z_negatives = z_shift[speaker_index, batch_index, seq_index, :]

            zs = torch.cat((z_shift.unsqueeze(2), z_negatives), dim=2)

            f = torch.sum(zs * Wc.unsqueeze(2) / math.sqrt(self.z_dim), dim=-1)
            f = f.view(
                self.n_speakers_per_batch * self.n_utterances_per_speaker,
                self.n_negatives + 1,
                -1
            )

            labels = torch.zeros(
                self.n_speakers_per_batch * self.n_utterances_per_speaker, length,
                dtype=torch.long, device=z.device
            )

            loss = F.cross_entropy(f, labels)

            accuracy = f.argmax(dim=1) == labels
            accuracy = torch.mean(accuracy.float())

            losses.append(loss)
            accuracies.append(accuracy.item())

        loss = torch.stack(losses).mean()
        return loss, accuracies
        
# def compute_matchmap_similarity_matrix_loss(image_outputs, english_output, hindi_output, english_nframes, hindi_nframes, margin, simtype, alphas):

#     loss = 0
#     alpha_1, alpha_2, alpha_3, alpha_4, alpha_5, alpha_6 = alphas

#     func = sampled_triplet_loss_from_S

#     if english_output is not None and english_nframes is not None:
#         S = compute_matchmap_similarity_matrix_IA(image_outputs, None, english_output, english_nframes, simtype)
#         I2E_sampled_loss = func(S, margin)
#         E2I_sampled_loss = func(S.t(), margin)
#         loss += ((alpha_1*I2E_sampled_loss) + (alpha_2*E2I_sampled_loss))

#     if hindi_output is not None and hindi_nframes is not None:
#         S = compute_matchmap_similarity_matrix_IA(image_outputs, None, hindi_output, hindi_nframes, simtype)
#         I2H_sampled_loss = func(S, margin)
#         H2I_sampled_loss = func(S.t(), margin)
#         loss += ((alpha_3*I2H_sampled_loss) + (alpha_4*H2I_sampled_loss))

#     if english_output is not None and english_nframes is not None and hindi_output is not None and hindi_nframes is not None:
#         S = compute_matchmap_similarity_matrix_AA(english_output, english_nframes, hindi_output, hindi_nframes, simtype)
#         E2H_sampled_loss = func(S, margin)
#         H2E_sampled_loss = func(S.t(), margin)
#         loss += ((alpha_5*E2H_sampled_loss) + (alpha_6*H2E_sampled_loss))

#     return loss


# class CPCLoss(nn.Module):
#     def __init__(self, args):
#         super(CPCLoss, self).__init__()

#         self.n_prediction_steps = args["cpc"]["n_prediction_steps"]
#         self.n_negatives = args["cpc"]["n_negatives"]
#         self.z_dim = args["audio_model"]["z_dim"]
#         self.c_dim = args["audio_model"]["c_dim"]
#         self.predictors = nn.ModuleList([
#             nn.Linear(self.c_dim, self.z_dim) for _ in range(self.n_prediction_steps)
#         ])

#     def forward(self, z, c, nframes):
#         length = z.size(1) - self.n_prediction_steps
#         num_frames_to_predict = length - self.n_negatives + 1

#         # z = z.reshape(
#         #     self.n_speakers_per_batch,
#         #     self.n_utterances_per_speaker,
#         #     -1,
#         #     self.z_dim
#         # )
#         c = c.transpose(1, 2)
#         c = c[:, self.n_negatives-1:-self.n_prediction_steps, :]

#         losses, accuracies = list(), list()
#         for k in range(1, self.n_prediction_steps+1):
#             z_shift = z[:, self.n_negatives+k-1:length + k, :]

#             predicted_z = self.predictors[k-1](c)

#             batch_index = torch.arange(z.size(0), device=z.device)
#             batch_index = batch_index.view(-1, 1, 1)

#             seq_index = torch.randint(
#                 0, length,
#                 size=(
#                     z.size(0), 
#                     self.n_negatives,
#                     num_frames_to_predict
#                 ),
#                 device=z.device
#             )
#             seq_index += torch.randint(
#                 0, self.n_negatives,
#                 size=(
#                     z.size(0), 
#                     self.n_negatives,
#                     num_frames_to_predict
#                 ),
#                 device=z.device
#             )
#             seq_index = torch.remainder(seq_index, torch.arange(num_frames_to_predict, device=z.device)+self.n_negatives)
#             seq_index = seq_index.view(z.size(0), self.n_negatives, num_frames_to_predict)

#             z_negatives = z[batch_index, seq_index, :]
            
#             zs = torch.cat((z_shift.unsqueeze(1), z_negatives), dim=1)

#             similarity_scores = torch.sum(zs * predicted_z.unsqueeze(1) / math.sqrt(self.z_dim), dim=-1) ###

#             labels = torch.zeros(
#                 z.size(0), num_frames_to_predict,
#                 dtype=torch.long, device=z.device
#             )

#             # for i in range(labels.size(0)):
#             #     labels[i, nframes[i]:] = -1

#             loss = F.cross_entropy(similarity_scores, labels)#, ignore_index=-1)
#             accuracy = similarity_scores.argmax(dim=1) == labels
#             accuracy = torch.mean(accuracy.float())

#             losses.append(loss)
#             accuracies.append(accuracy.item())

#         loss = torch.stack(losses).mean()
#         return loss, accuracies


# class CPCLoss(nn.Module):
#     def __init__(self, args):
#         super(CPCLoss, self).__init__()

#         self.n_speakers_per_batch = args["cpc"]["n_speakers_per_batch"]
#         self.n_utterances_per_speaker = args["cpc"]["n_utterances_per_speaker"]
#         self.n_prediction_steps = args["cpc"]["n_prediction_steps"]
#         self.n_negatives = args["cpc"]["n_negatives"]
#         self.z_dim = args["audio_model"]["z_dim"]
#         self.c_dim = args["audio_model"]["c_dim"]

#         self.predictors = nn.ModuleList([
#             nn.Linear(self.c_dim, self.z_dim) for _ in range(self.n_prediction_steps)
#         ])

#     def forward(self, z, c, nframes):
#         length = z.size(1) - self.n_prediction_steps

#         z = z.reshape(
#             self.n_speakers_per_batch,
#             self.n_utterances_per_speaker,
#             -1,
#             self.z_dim
#         )
#         c = c.transpose(1, 2)
#         c = c[:, :-self.n_prediction_steps, :]

#         losses, accuracies = list(), list()
#         for k in range(1, self.n_prediction_steps+1):
#             z_shift = z[:, :, k:length + k, :]

#             Wc = self.predictors[k-1](c)
#             Wc = Wc.view(
#                 self.n_speakers_per_batch,
#                 self.n_utterances_per_speaker,
#                 -1,
#                 self.z_dim
#             )

#             batch_index = torch.randint(
#                 0, self.n_utterances_per_speaker,
#                 size=(
#                     self.n_utterances_per_speaker,
#                     self.n_negatives
#                 ),
#                 device=z.device
#             )
#             batch_index = batch_index.view(
#                 1, self.n_utterances_per_speaker, self.n_negatives, 1
#             )

#             seq_index = torch.randint(
#                 1, length,
#                 size=(
#                     self.n_speakers_per_batch,
#                     self.n_utterances_per_speaker,
#                     self.n_negatives,
#                     length
#                 ),
#                 device=z.device
#             )
#             seq_index += torch.arange(length, device=z.device)
#             seq_index = torch.remainder(seq_index, length)

#             speaker_index = torch.arange(self.n_speakers_per_batch, device=z.device)
#             speaker_index = speaker_index.view(-1, 1, 1, 1)

#             z_negatives = z_shift[speaker_index, batch_index, seq_index, :]

#             zs = torch.cat((z_shift.unsqueeze(2), z_negatives), dim=2)

#             f = torch.sum(zs * Wc.unsqueeze(2) / math.sqrt(self.z_dim), dim=-1)
#             f = f.view(
#                 self.n_speakers_per_batch * self.n_utterances_per_speaker,
#                 self.n_negatives + 1,
#                 -1
#             )

#             labels = torch.zeros(
#                 self.n_speakers_per_batch * self.n_utterances_per_speaker, length,
#                 dtype=torch.long, device=z.device
#             )

#             loss = F.cross_entropy(f, labels)

#             accuracy = f.argmax(dim=1) == labels
#             accuracy = torch.mean(accuracy.float())

#             losses.append(loss)
#             accuracies.append(accuracy.item())

#         loss = torch.stack(losses).mean()
#         return loss, accuracies
