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

        self.embedding_dim = args["audio_model"]["embedding_dim"]
        self.image_attention_encoder = nn.Sequential(
            # nn.LayerNorm(64),
            nn.Linear(64, 64),
            nn.ReLU(),
            # nn.LayerNorm(64),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.audio_attention_encoder = nn.Sequential(
            # nn.LayerNorm(512),
            nn.Linear(512, 512),
            nn.ReLU(),
            # nn.LayerNorm(512),
            nn.Linear(512, 512),
            nn.ReLU()
        )
        # self.image_embedding_encoder = nn.Sequential(
        #     # nn.LayerNorm(64),
        #     nn.Linear(self.embedding_dim, self.embedding_dim),
        #     nn.ReLU(),
        #     # nn.LayerNorm(64),
        #     nn.Linear(self.embedding_dim, self.embedding_dim),
        #     nn.ReLU()
        # )
        # self.audio_embedding_encoder = nn.Sequential(
        #     # nn.LayerNorm(512),
        #     nn.Linear(self.embedding_dim, self.embedding_dim),
        #     nn.ReLU(),
        #     # nn.LayerNorm(512),
        #     nn.Linear(self.embedding_dim, self.embedding_dim),
        #     nn.ReLU()
        # )
        # self.similarity_encoder = nn.Sequential(
        #     # nn.LayerNorm(64),
        #     nn.Linear(self.embedding_dim, self.embedding_dim),
        #     nn.ReLU(),
        #     # nn.LayerNorm(64),
        #     nn.Linear(self.embedding_dim, 1),
        #     nn.ReLU()
        # )
        self.pool_func = nn.AdaptiveAvgPool2d((1, 1))
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

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

    def forward(self, image_embedding, audio_embeddings, audio_nframes):

        aud_att = torch.bmm(image_embedding, audio_embeddings)
        aud_att, _ = aud_att.max(dim=1)
        aud_att = aud_att.unsqueeze(1)
        aud_att = self.normalise(aud_att, audio_nframes)
        # aud_att = torch.sigmoid(aud_att)
        aud_context = (aud_att[:, :, 0:audio_nframes] * audio_embeddings[:, :, 0:audio_nframes]).mean(dim=-1)
        # aud_context = torch.bmm(aud_att[:, 0:audio_nframes, :], audio_embeddings.transpose(1, 2)).squeeze(1)# / aud_att.size(2)
        
        im_att = torch.bmm(audio_embeddings.transpose(1, 2), image_embedding.transpose(1, 2))
        im_att, _ = im_att.max(dim=1)
        im_att = im_att.unsqueeze(1)
        im_att = self.normalise(im_att)
        # print(im_att)
        im_context = (im_att * image_embedding.transpose(1, 2)).mean(dim=-1)

        # im_att = torch.sigmoid(im_att)
        # im_context = torch.bmm(im_att, image_embedding).squeeze(1)# / im_att.size(2)

        return aud_context, im_context

    def encode(self, image_embedding, audio_embeddings, audio_nframes): 
        
        aud_att = torch.bmm(image_embedding, audio_embeddings)
        aud_att, _ = aud_att.max(dim=1)
        aud_att = aud_att.unsqueeze(1)
        # aud_att = torch.sigmoid(aud_att)
        att = []
        aud_context = []
        for i in range(aud_att.size(0)):
            a = self.normalise(aud_att[i, :, :].unsqueeze(0), audio_nframes[i])
            c = (a[:, :, 0:audio_nframes[i]] * audio_embeddings[i, :, 0:audio_nframes[i]].unsqueeze(0)).mean(dim=-1)
            att.append(a)
            aud_context.append(c)
        aud_att = torch.cat(att, dim=0)
        aud_context = torch.cat(aud_context, dim=0)
        # aud_context = torch.bmm(aud_att, audio_embeddings.transpose(1, 2)).squeeze(1)# / aud_att.size(2)
        
        im_att = torch.bmm(audio_embeddings.transpose(1, 2), image_embedding.transpose(1, 2))
        im_att, _ = im_att.max(dim=1)
        im_att = im_att.unsqueeze(1)
        # im_att = torch.sigmoid(im_att)
        att = []
        im_context = []
        for i in range(im_att.size(0)):
            a = self.normalise(im_att[i, :, :].unsqueeze(0))
            c = (a * image_embedding[i, :, :].unsqueeze(0).transpose(1, 2)).mean(dim=-1)
            att.append(a)
            im_context.append(c)
        im_att = torch.cat(att, dim=0)
        im_context = torch.cat(im_context, dim=0)
        # im_context = torch.bmm(im_att, image_embedding).squeeze(1)# / im_att.size(2)

        score = self.cos(aud_context, im_context).unsqueeze(1)
        # print(im_context.size(), aud_context.size())
        # score = torch.bmm(aud_context.unsqueeze(1), im_context.unsqueeze(2))

        return aud_att, im_att, aud_context, im_context, score, audio_embeddings
        
class ContrastiveLoss(nn.Module):
    def __init__(self, args):
        super(ContrastiveLoss, self).__init__()

        self.embedding_dim = args["audio_model"]["embedding_dim"]
        self.margin = args["margin"]
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.criterion = nn.MSELoss()

    def forward(self, anchor, positives, negatives):

        # N = anchor.size(0)
        # loss = 0
        # sim = []
        # labels = []
        # for p in range(positives.size(1)):
        #     sim.append(self.cos(positives[:, p, :], anchor.squeeze(1)).unsqueeze(1))
        #     labels.append(torch.ones((anchor.size(0), 1), device=anchor.device))
        # for n in range(negatives.size(1)): 
        #     sim.append(self.cos(negatives[:, n, :], anchor.squeeze(1)).unsqueeze(1))
        #     labels.append(-1 * torch.ones((anchor.size(0), 1), device=anchor.device))
        #     # loss += (self.cos(negatives[:, n, :], anchor.squeeze(1)) - self.cos(positives[:, p, :], anchor.squeeze(1)) + 2.0).clamp(min=0).mean()
        #     # print(self.cos(negatives[:, n, :], anchor.squeeze(1)), self.cos(positives[:, p, :], anchor.squeeze(1)))
        # sim = torch.cat(sim, dim=1)
        # labels = torch.cat(labels, dim=1)
        # loss += self.criterion(sim, labels)

        N = anchor.size(0)
        loss = 0
        for p in range(positives.size(1)):
            sim = [self.cos(positives[:, p, :], anchor.squeeze(1)).unsqueeze(1)]
            labels = [torch.ones((anchor.size(0), 1), device=anchor.device)]
            for n in range(negatives.size(1)): 
                sim.append(self.cos(negatives[:, n, :], anchor.squeeze(1)).unsqueeze(1))
                labels.append(-1 * torch.ones((anchor.size(0), 1), device=anchor.device))
                # loss += (self.cos(negatives[:, n, :], anchor.squeeze(1)) - self.cos(positives[:, p, :], anchor.squeeze(1)) + 2.0).clamp(min=0).mean()
                # print(self.cos(negatives[:, n, :], anchor.squeeze(1)), self.cos(positives[:, p, :], anchor.squeeze(1)))
            sim = torch.cat(sim, dim=1)
            labels = torch.cat(labels, dim=1)
            loss += self.criterion(sim, labels) 

        return loss

    # def encode(self, anchor, positives, negatives):

    #     samples = torch.cat([positives, negatives], dim=1)
    #     # sim = torch.bmm(anchor, samples.transpose(1, 2)).squeeze(1)
    #     sim = []
    #     for i in range(anchor.size(0)):
    #         sim.append(self.cos(anchor[i, :, :].repeat(samples.size(1), 1), samples[i, :, :]).unsqueeze(0))
    #     sim = torch.cat(sim, dim=0)
    #     labels = torch.zeros(sim.size(0), dtype=torch.long, device=sim.device)

    #     return sim, labels