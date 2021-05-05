import torch
import torch.nn.functional as F
import torch.nn as nn


class SEAttention(nn.Module):
    """
    Implements a Channel Attention Module as proposed in the report.
    Consists of two main stages:
    1) Squeeze Module: For computing initial weights
    2) Excitation Module: For learning transformed weights and attending the input
    """
    def __init__(self, d_model):
        super(SEAttention, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excite = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

    def forward(self, input, padding_mask=None):
        S, B, D = input.shape

        # Zero out the pad tokens to ignore there effect during
        # SE channel weight computation
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(2)
            B, S, D_ = padding_mask.shape
            padding_mask = padding_mask.contiguous().view(S, B, D_)
            input = input * (1 - padding_mask)

        # Squeeze
        input_ = input.view(B, D, S).contiguous()
        weights = self.squeeze(input_)
        weights = weights.permute(0, 2, 1).contiguous()

        # Excitation
        weights = self.excite(weights).permute(1, 0, 2)

        # Combine
        return input * weights


class PositionalEmbedding(nn.Module):
    """
    A PositionalEmbedding module which creates positional embeddings
    as proposed in https://arxiv.org/abs/1706.03762. Supports both
    sinusoid and learned positional embeddings.
    """
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
        # Compute position embeddings based on computed positions.
        embeddings = self.pos_embedding(positions)
        return embeddings


class SinusoidEmbedding:
    """
    Implementation of the Sinusoid position embedding module. Uses the
    same form as described in https://arxiv.org/abs/1706.03762.
    """
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
