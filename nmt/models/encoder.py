import torch.nn as nn
import torch
import torch.nn.functional as F


class TransformerEncoder(nn.Module):
    """A container for the Encoder stacks in the Transformer model
    """
    def __init__(self, encoder_layer, num_layers, norm=None, return_attn_weights=False):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
        self.return_attn_weights = return_attn_weights

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        attn_weights = []

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            if len(output) == 2:
                output, weights = output[0], output[1]
                attn_weights.append(weights)

        if self.norm is not None:
            output = self.norm(output)

        if self.return_attn_weights:
            return output, torch.stack(attn_weights)
        return output


class TransformerEncoderLayer(nn.Module):
    """A single encoder stack consisting of Multi-Head attention
    and Feedforward modules. Same as proposed in https://arxiv.org/abs/1706.03762
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5, return_attn_weights=False):
        super(TransformerEncoderLayer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout
        self.return_attn_weights = return_attn_weights

        self.self_attn = nn.MultiheadAttention(self.d_model, nhead, dropout=self.dropout_rate)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(self.d_model, self.dim_feedforward)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.linear2 = nn.Linear(self.dim_feedforward, self.d_model)

        self.norm1 = nn.LayerNorm(self.d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(self.d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.dropout2 = nn.Dropout(self.dropout_rate)

        self.activation = nn.ReLU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self-attention -> Dropout -> LayerNorm -> Feedforward -> Dropout -> LayerNorm
        src2, weights = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        if self.return_attn_weights:
            return src, weights
        return src

    def __repr__(self):
        return self.__class__.__name__ + f'(d_model={self.d_model}, nhead={self.nhead}, ff={self.dim_feedforward}, p={self.dropout_rate})'


class TransformerEncoderDualAttnLayer(nn.Module):
    """A single encoder stack consisting of Multi-Head attention + Channel Attention
    modules followed by a Feedforward module.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5, return_attn_weights=False):
        super(TransformerEncoderDualAttnLayer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout
        self.return_attn_weights = return_attn_weights

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.channel_attn = SEAttention(d_model)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Compute spatial attention
        src2_spatial, weights = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )

        # Compute Channel-wise attention
        src2_channel = self.channel_attn(
            src, padding_mask=src_key_padding_mask
        )
        src = src + self.dropout1(src2_spatial + src2_channel)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        if self.return_attn_weights:
            return src, weights
        return src

    def __repr__(self):
        return self.__class__.__name__ + f'(d_model={self.d_model}, nhead={self.nhead}, ff={self.dim_feedforward}, p={self.dropout_rate})'


class TransformerEncoderFusedAttnLayer(nn.Module):
    def __init__(self, d_model, head_conf=[4, 8], dim_feedforward=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5):
        """A single encoder stack consisting of Fused Multi-Head attention
        module followed by a Feedforward module. (Refer to the report
        for more technical details.)
        """
        super(TransformerEncoderFusedAttnLayer, self).__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout
        self.head_conf = head_conf

        self.self_attn_list = nn.ModuleList()
        for head in self.head_conf:
            self.self_attn_list.append(nn.MultiheadAttention(d_model, head, dropout=dropout))

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # TODO: This operation can also be concat (and needs to be experimented with)
        src2 = 0
        for attn_module in self.self_attn_list:
            src2 += attn_module(src, src, src, attn_mask=src_mask,
                                key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def __repr__(self):
        return self.__class__.__name__ + f'(d_model={self.d_model}, head_conf={self.head_conf}, ff={self.dim_feedforward}, p={self.dropout_rate})'
