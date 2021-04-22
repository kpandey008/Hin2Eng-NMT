import torch
import torch.nn.functional as F
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim, num_embeddings=None, mode='sinusoid'):
        super(PositionalEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.mode = mode

        if self.mode == 'learned' and num_embeddings is None:
            raise ValueError('The num_embeddings param must be specified when using learned embeddings!')

        if self.mode == 'sinusoid':
            self.pos_embedding = SinusoidEmbedding(embedding_dim=self.embedding_dim)
        elif self.mode == 'learned':
            self.pos_embedding = LearnedEmbedding(embedding_dim=self.embedding_dim, num_embeddings=self.num_embeddings)
        else:
            raise ValueError(f'The embedding type {self.mode} is not supported yet!')

    def forward(self, x):
        S, B, D = x.shape
        assert self.embedding_dim == D
        positions = torch.arange(0, S, 1).to(x.device)
        embeddings = self.pos_embedding(positions)
        return embeddings


class SinusoidEmbedding:
    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim

    def __call__(self, positions):
        device = positions.device
        num_positions = positions.shape[-1]
        embedding = torch.zeros((self.embedding_dim, num_positions)).to(device)
        dim_range = torch.arange(0, self.embedding_dim // 2).to(device)
        positions = torch.unsqueeze(positions, 0)

        e_sin = 1 / (10000 ** (2 * dim_range / float(self.embedding_dim)))
        e_sin = torch.unsqueeze(e_sin, 1)
        e_sin = torch.sin(e_sin * positions).to(device)
        embedding[2 * dim_range] = e_sin

        e_cos = 1 / (10000 ** ((2 * dim_range + 1) / float(self.embedding_dim)))
        e_cos = torch.unsqueeze(e_cos, 1)
        e_cos = torch.cos(e_cos * positions).to(device)
        embedding[2 * dim_range + 1] = e_cos
        return embedding.detach()


class LearnedEmbedding(nn.Module):
    def __init__(self, embedding_dim, num_embeddings):
        super(LearnedEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=self.embedding_dim ** -0.5)

    def forward(self, positions):
        # Clip the position if it is greater than the max allowed embeddings
        positions = torch.clamp(positions, max=self.num_embeddings)
        # Retrieve the corresponding embeddings and permute
        return self.embedding(positions).permute(1, 0)
