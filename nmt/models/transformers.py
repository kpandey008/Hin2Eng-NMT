import torch.nn as nn
import torch
import torch.nn.functional as F


from models.encoder import *
from models.decoder import *
from models.util import *


class NMTModel(nn.Module):
    """
    The main NMT model which combines different types of encoder and
    decoder layers to create the overall model.
    """
    def __init__(
        self, d_model, nhead, de_vocab_size, en_vocab_size,
        n_encoder_layers=6, n_decoder_layers=6, pe_mode='sinusoid',
        return_attn_weights=False
    ):
        super(NMTModel, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.n_encoder_layers = n_encoder_layers
        self.n_decoder_layers = n_decoder_layers
        self.en_vocab_size = en_vocab_size
        self.de_vocab_size = de_vocab_size
        self.return_attn_weights = return_attn_weights

        # Word embeddings
        self.en_embedding = nn.Embedding(self.en_vocab_size, embedding_dim=self.d_model)
        self.de_embedding = nn.Embedding(self.de_vocab_size, embedding_dim=self.d_model)

        # Positional Embeddings
        self.pe = PositionalEmbedding(self.d_model, mode=pe_mode, num_embeddings=128)

        # Transformer layers
        encoder_layer = TransformerEncoderDualAttnLayer(self.d_model, self.nhead, return_attn_weights=return_attn_weights)
        decoder_layer = TransformerDecoderLayer(self.d_model, self.nhead, return_attn_weights=return_attn_weights)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=self.n_encoder_layers, return_attn_weights=return_attn_weights)
        self.decoder = TransformerDecoder(decoder_layer, num_layers=self.n_decoder_layers, return_attn_weights=return_attn_weights)

        # Clf
        self.clf = nn.Linear(self.d_model, self.en_vocab_size)

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
        if len(memory) == 2:
            memory, enc_weights = memory[0], memory[1]

        # Decoder forward
        out = self.decoder(tgt_embedding + tgt_pe, memory, tgt_mask=tgt_attn_mask, tgt_key_padding_mask=tgt_key_padding_masks)
        if len(out) == 2:
            out, dec_weights = out[0], out[1]

        if self.return_attn_weights:
            return self.clf(out), enc_weights, dec_weights
        return self.clf(out)

    def generate(self, src_ids, device, src_key_padding_masks=None, src_attn_mask=None, max_length=64):
        enc_attn_weights = []
        dec_attn_weights = []

        # Use the device from one of nn modules!
        sos_token_id = en_vocab.token2id[en_vocab.sos_token]
        eos_token_id = en_vocab.token2id[en_vocab.eos_token]

        # Iteratively generate output tokens
        output_batch = []
        for id, sentence in enumerate(src_ids):
            src = sentence.unsqueeze(0)  # BatchSize:1
            token_seq = [sos_token_id]
            dec_ = []
            for i in range(1, max_length):
                tgt = torch.Tensor(token_seq).long().unsqueeze(0).to(device)
                tgt_attn_mask = generate_square_subsequent_mask(i).to(device)
                out = self(
                    src,
                    tgt,
                    src_key_padding_masks=src_key_padding_masks[id, :].unsqueeze(0),
                    tgt_attn_mask=tgt_attn_mask
                )
                if len(out) == 3:
                    out, enc_weights, dec_weights = out
                
                # Exchange Batch and seq length dimensions
                out = out.transpose(0, 1)

                # Get the predictions and extract the last word in the sequence
                out = torch.argmax(F.log_softmax(out, dim=-1), dim=-1).squeeze(0).cpu().numpy()[-1]

                # Get the decoder attention weights for the last generated token
                # dec_weights: Shape: N x 1 x T, S
                w = dec_weights.squeeze(1)[:, -1, :]
                dec_.append(w)

                # Append the last token and re-iterate until <EOS>
                token_seq.append(out)
                if out == eos_token_id:
                    break

            # Add the generated seq to the batch
            output_batch.append(token_seq)

            # Add encoder and decoder attention weights
            if self.return_attn_weights:
                enc_attn_weights.append(enc_weights.squeeze(1))
                dec_attn_weights.append(torch.stack(dec_, dim=1))
                
                # Stack along the batch and return
                enc_attn_weights = torch.stack(enc_attn_weights, dim=0)
                dec_attn_weights = torch.stack(dec_attn_weights, dim=0)
                return output_batch, enc_attn_weights, dec_attn_weights
            return output_batch

    def __repr__(self):
        return self.__class__.__name__ + f'(d_model={self.d_model}, nhead={self.nhead}, encoder={self.encoder}, decoder={self.decoder})'


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


if __name__ == '__main__':
    encoder_layer = TransformerEncoderDualAttnLayer(512, 8)
    # S x B x D
    input = torch.randn(10, 1, 512)
    out = encoder_layer(input)
