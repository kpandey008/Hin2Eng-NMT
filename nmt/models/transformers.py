import torch.nn as nn
import torch
import torch.nn.functional as F


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5, batch_first=False):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
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

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
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


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class NMTModel(nn.Module):
    def __init__(
        self, d_model, nhead, de_vocab_size, en_vocab_size, n_encoder_layers=6, n_decoder_layers=6, pe_mode='learned'
    ):
        super(NMTModel, self).__init__()
        self.en_vocab_size = en_vocab_size
        self.de_vocab_size = de_vocab_size

        # Word embeddings
        self.en_embedding = nn.Embedding(self.en_vocab_size, embedding_dim=d_model)
        self.de_embedding = nn.Embedding(self.de_vocab_size, embedding_dim=d_model)

        # Positional Embeddings
        self.pe = PositionalEmbedding(d_model, mode=pe_mode, num_embeddings=128)

        # Transformer layers
        encoder_layer = TransformerEncoderLayer(d_model, nhead)
        decoder_layer = TransformerDecoderLayer(d_model, nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layers)

        # Clf
        self.clf = nn.Linear(d_model, self.en_vocab_size)

    def forward(self, src_ids, tgt_ids, src_key_padding_masks=None, tgt_key_padding_masks=None, src_attn_mask=None, tgt_attn_mask=None):
        src_embedding = self.de_embedding(src_ids)
        tgt_embedding = self.en_embedding(tgt_ids)

        # Reshape embedddings into (S, B, D)
        src_embedding = src_embedding.permute(1, 0, 2).contiguous()
        tgt_embedding = tgt_embedding.permute(1, 0, 2).contiguous()

        # Compute positional embeddings for the inputs
        src_pe = self.pe(src_embedding).unsqueeze(1).permute(2, 1, 0)
        tgt_pe = self.pe(tgt_embedding).unsqueeze(1).permute(2, 1, 0)

        # Encoder forward
        memory = self.encoder(src_embedding + src_pe, mask=src_attn_mask, src_key_padding_mask=src_key_padding_masks)

        # Decoder forward
        out = self.decoder(tgt_embedding + tgt_pe, memory, tgt_mask=tgt_attn_mask, tgt_key_padding_mask=tgt_key_padding_masks)
        return self.clf(out)

    def generate(self, src_ids, device, src_key_padding_masks=None, src_attn_mask=None, max_length=20):
        # Use the device from one of nn modules!
        sos_token_id = en_vocab.token2id[en_vocab.sos_token]
        eos_token_id = en_vocab.token2id[en_vocab.eos_token]

        # Iteratively generate output tokens
        output_batch = []
        for id, sentence in enumerate(src_ids):
            src = sentence.unsqueeze(0)  # BatchSize:1
            token_seq = [sos_token_id]
            for i in range(1, max_length):
                tgt = torch.Tensor(token_seq).long().unsqueeze(0).to(device)
                tgt_attn_mask = generate_square_subsequent_mask(i).to(device)
                out = model(src, tgt, src_key_padding_masks=src_key_padding_masks[id, :].unsqueeze(0), tgt_attn_mask=tgt_attn_mask).transpose(0, 1)

                # Get the predictions and extract the last word in the sequence
                out = torch.argmax(F.log_softmax(out, dim=-1), dim=-1).squeeze(0).cpu().numpy()[-1]
                token_seq.append(out)
                if out == eos_token_id:
                    break
            output_batch.append(token_seq)
        return output_batch


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask