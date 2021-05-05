import torch.nn as nn
import torch
import torch.nn.functional as F


class TransformerDecoder(nn.Module):
    """A container for the Decoder stacks in the Transformer model 
    """
    def __init__(self, decoder_layer, num_layers, norm=None, return_attn_weights=False):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
        self.return_attn_weights = return_attn_weights

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt
        attn_weights = []

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
            if len(output) == 2:
                output, weights = output[0], output[1]
                attn_weights.append(weights)

        if self.norm is not None:
            output = self.norm(output)

        if self.return_attn_weights:
            return output, torch.stack(attn_weights)
        return output


class TransformerDecoderLayer(nn.Module):
    """A single Decoder stack consisting of Self Multi-Head attention
    module followed by a Cross Multi-Head attention and a Feedforward module.
    Same as proposed in https://arxiv.org/abs/1706.03762
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5, batch_first=False, return_attn_weights=False):
        super(TransformerDecoderLayer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout
        self.return_attn_weights = return_attn_weights

        self.self_attn = nn.MultiheadAttention(self.d_model, self.nhead, dropout=self.dropout_rate)
        self.multihead_attn = nn.MultiheadAttention(self.d_model, self.nhead, dropout=self.dropout_rate)
        self.linear1 = nn.Linear(self.d_model, dim_feedforward)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.linear2 = nn.Linear(self.dim_feedforward, self.d_model)

        self.norm1 = nn.LayerNorm(self.d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(self.d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(self.d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.dropout2 = nn.Dropout(self.dropout_rate)
        self.dropout3 = nn.Dropout(self.dropout_rate)

        self.activation = nn.ReLU()

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, weights = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        if self.return_attn_weights:
            return tgt, weights
        return tgt

    def __repr__(self):
        return self.__class__.__name__ + f'(d_model={self.d_model}, nhead={self.nhead}, ff={self.dim_feedforward}, p={self.dropout_rate})'


class TransformerDecoderDualAttnLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5, batch_first=False):
        super(TransformerDecoderDualAttnLayer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.channel_attn = SEAttention(d_model)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # Compute Spatial attention
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        # Compute Channel-wise attention
        tgt2_channel = self.channel_attn(
            tgt, padding_mask=tgt_key_padding_mask
        )
        tgt = tgt + self.dropout1(tgt2 + tgt2_channel)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def __repr__(self):
        return self.__class__.__name__ + f'(d_model={self.d_model}, nhead={self.nhead}, ff={self.dim_feedforward}, p={self.dropout_rate})'


class TransformerDecoderFusedAttnLayer(nn.Module):
    def __init__(self, d_model, head_conf=[4, 8], dim_feedforward=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5, batch_first=False):
        super(TransformerDecoderFusedAttnLayer, self).__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout
        self.head_conf = head_conf

        self.self_attn_list = nn.ModuleList()
        for head in self.head_conf:
            self.self_attn_list.append(nn.MultiheadAttention(d_model, head, dropout=dropout))

        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2 = 0
        for attn_module in self.self_attn_list:
            tgt2 += attn_module(tgt, tgt, tgt, attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def __repr__(self):
        return self.__class__.__name__ + f'(d_model={self.d_model}, head_conf={self.head_conf}, ff={self.dim_feedforward}, p={self.dropout_rate})'
