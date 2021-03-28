import os
import pandas as pd
import torch

from torch.utils.data import Dataset


class Hin2EngDataset(Dataset):
    # NOTE: In the code, we use the prefix de to denote Devanagri and `en` to denote English
    def __init__(self, de_file_path, en_file_path, de_tokenizer, en_tokenizer, max_length=None):
        if not os.path.isfile(de_file_path):
            raise Exception(f'Path `{de_file_path}` is not a valid file')
        if not os.path.isfile(en_file_path):
            raise Exception(f'Path `{en_file_path}` is not a valid file')

        self.de_file_path = de_file_path
        self.en_file_path = en_file_path
        self.max_length = max_length

        self.de_text = pd.read_csv(self.de_file_path)
        self.en_text = pd.read_csv(self.en_file_path)

        self.de_tokenizer = de_tokenizer
        self.en_tokenizer = en_tokenizer

    def __getitem__(self, idx):
        de_text = self.de_text['Hindi'][idx]
        en_text = self.en_text['English'][idx]
        return de_text, en_text

    def collate_fn(self, batch):
        de_batch, en_batch = zip(*batch)

        # Enable padding when encoding
        self.de_tokenizer.enable_padding(
            pad_id=self.de_tokenizer.token_to_id('[PAD]'),
            pad_token="[PAD]"
        )
        self.en_tokenizer.enable_padding(
            pad_id=self.en_tokenizer.token_to_id('[PAD]'),
            pad_token="[PAD]"
        )

        # Enable truncation if max_length is set
        if self.max_length is not None:
            self.de_tokenizer.enable_truncation(self.max_length)
            self.en_tokenizer.enable_truncation(self.max_length)

        # Encode batches and compute token ids and attention masks
        de_batch_enc = self.de_tokenizer.encode_batch(de_batch)
        en_batch_enc = self.en_tokenizer.encode_batch(en_batch)

        de_batch_enc_ids = torch.Tensor([enc.ids for enc in de_batch_enc]).to(torch.int64)
        en_batch_enc_ids = torch.Tensor([enc.ids for enc in en_batch_enc]).to(torch.int64)

        de_batch_enc_mask = torch.Tensor([enc.attention_mask for enc in de_batch_enc]).to(torch.int64)
        en_batch_enc_mask = torch.Tensor([enc.attention_mask for enc in en_batch_enc]).to(torch.int64)

        return de_batch_enc_ids, de_batch_enc_mask, en_batch_enc_ids, en_batch_enc_mask

    def __len__(self):
        return self.de_text.shape[0]
