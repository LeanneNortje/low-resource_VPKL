#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2022
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________

import torch
import torch.nn as nn
import torch.nn.functional as F

class ScoringAttentionModule(nn.Module):
    def __init__(self, args):
        super(ScoringAttentionModule, self).__init__()

        # self.embedding_dim = args["audio_model"]["embedding_dim"]
        # self.image_attention_encoder = nn.Sequential(
        #     nn.Linear(64, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 64),
        #     nn.ReLU()
        # )
        # self.audio_attention_encoder = nn.Sequential(
        #     nn.Linear(512, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 512),
        #     nn.ReLU()
        # )

        # self.relu = nn.ReLU(inplace=True)
        # self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def normalise(self, x, fr=None):
        if fr is None:
            minimum, _ = x.min(dim=-1)
            x = x - minimum
            maximum, _ = x.max(dim=-1)
            x = x / maximum
        else:
            minimum, _ = x[:, :, 0:fr].min(dim=-1)
            x = x - minimum
            maximum, _ = x[:, :, 0:fr].max(dim=-1)
            x = x / maximum
        return x

    def get_mask(self, x, size1, size2):

        x = x.to(torch.long)
        a = -1 * torch.ones((size1, size2), device=x.device)
        for i in range(a.size(0)):
            a[i, 0:x[i]] *= -1
        return a

    def forward(self, image_embedding, mask_1, audio_embeddings, mask_2):

        mask = self.get_mask(mask_2, audio_embeddings.size(0), audio_embeddings.size(-1)).unsqueeze(1)

        att = torch.bmm(image_embedding, audio_embeddings)
        att[att < 0] = 0
        att *= mask
        att, _ = att.max(dim=1)
        S, _ = att.max(dim=1)
        
        return S.unsqueeze(-1)

    def encode(self, image_embedding, mask_1, audio_embeddings, mask_2): 
        
        # att = torch.bmm(image_embedding, audio_embeddings)
        # aud_att, _ = att.max(dim=1)
        
        # s, _ = aud_att.max(dim=1)

        mask = self.get_mask(mask_2, audio_embeddings.size(0), audio_embeddings.size(-1)).unsqueeze(1)

        att = torch.bmm(image_embedding, audio_embeddings)
        att[att < 0] = 0
        att *= mask
        aud_att, _ = att.max(dim=1)
        S, _ = aud_att.max(dim=1)

        return S, aud_att

    def encodeSimilarity(self, image_embedding, mask_1, audio_embeddings, mask_2): 
        
        mask = self.get_mask(mask_2, audio_embeddings.size(0), audio_embeddings.size(-1)).unsqueeze(1)

        att = torch.bmm(image_embedding, audio_embeddings)
        att[att < 0] = 0
        att *= mask
        att, _ = att.max(dim=1)
        S, _ = att.max(dim=1)
        
        return S.unsqueeze(-1)

    # def forward(self, image_embedding, mask_1, audio_embeddings, mask_2):

        
    #     if mask_1 is not None: image_embedding = image_embedding[:, 0:mask_1, :]
    #     if mask_2 is not None: audio_embeddings = audio_embeddings[:, :, 0:mask_2]

    #     att = torch.bmm(image_embedding, audio_embeddings)

    #     att, _ = att.max(dim=1)
    #     S, _ = att.max(dim=1)

    #     return S.unsqueeze(-1)

    # def encode(self, image_embedding, mask_1, audio_embeddings, mask_2): 
    #     att = []
    #     S = []

    #     N = image_embedding.size(0)
    #     for i in range(N):

    #         if mask_1 is not None: im = image_embedding[i, 0:mask_1[i], :].unsqueeze(0)
    #         else: im = image_embedding[i, :, :].unsqueeze(0)
    #         if mask_2 is not None: aud = audio_embeddings[i, :, 0:mask_2[i]].unsqueeze(0)
    #         else: aud = audio_embeddings[i, :, :].unsqueeze(0)

    #         a = torch.bmm(im, aud)
            
    #         a, _ = a.max(dim=1)
    #         a_s, _ = a.max(dim=1)
    #         att.append(a)
    #         S.append(a_s)

    #     att = torch.cat(att, dim=0)
    #     S = torch.cat(S, dim=0)

    #     return S, att

    # def encodeSimilarity(self, image_embedding, mask_1, audio_embeddings, mask_2): 
    #     S = []

    #     N = image_embedding.size(0)
    #     for i in range(N):

    #         if mask_1 is not None: im = image_embedding[i, 0:mask_1[i], :].unsqueeze(0)
    #         else: im = image_embedding[i, :, :].unsqueeze(0)
    #         if mask_2 is not None: aud = audio_embeddings[i, :, 0:mask_2[i]].unsqueeze(0)
    #         else: aud = audio_embeddings[i, :, :].unsqueeze(0)

    #         a = torch.bmm(im, aud)

    #         a, _ = a.max(dim=1)
    #         a_s, _ = a.max(dim=1)
    #         S.append(a_s.unsqueeze(-1))

    #     S = torch.cat(S, dim=0)
    #     # att = torch.bmm(image_embedding, audio_embeddings)
    #     # aud_att, _ = att.max(dim=1)
        
    #     # s, _ = aud_att.max(dim=1)
    #     return S
        
class ContrastiveLoss(nn.Module):
    def __init__(self, args):
        super(ContrastiveLoss, self).__init__()

        self.embedding_dim = args["audio_model"]["embedding_dim"]
        self.margin = args["margin"]
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.criterion = nn.MSELoss()

    def forward(self, anchor, positives_1, positives_2, negatives_1, negatives_2):
        N = anchor.size(0)
        sim = [anchor, positives_1, positives_2, negatives_1, negatives_2]
        # if base_negatives is not None: sim.append(base_negatives)
        sim = torch.cat(sim, dim=1)
        labels = []
        labels.append(100*torch.ones((N, anchor.size(1)), device=anchor.device))
        labels.append(100*torch.ones((N, positives_1.size(1)), device=anchor.device))
        labels.append(100*torch.ones((N, positives_2.size(1)), device=anchor.device))
        labels.append(0*torch.ones((N, negatives_1.size(1)), device=anchor.device))
        labels.append(0*torch.ones((N, negatives_2.size(1)), device=anchor.device))
        # if base_negatives is not None: labels.append(0*torch.ones((N, base_negatives.size(1)), device=anchor.device))
        labels = torch.cat(labels, dim=1)
        loss = self.criterion(sim, labels)

        return loss