import os
import pandas as pd
import torch

from torch.utils.data import Dataset


class Hin2EngDataset(Dataset):
    # NOTE: In the code, we use the prefix de to denote Devanagri and `en` to denote English
    def __init__(self, de_file_path, en_file_path, tokenizer, max_length=None):
        if not os.path.isfile(de_file_path):
            raise Exception(f'Path `{de_file_path}` is not a valid file')
        if not os.path.isfile(en_file_path):
            raise Exception(f'Path `{en_file_path}` is not a valid file')

        self.de_file_path = de_file_path
        self.en_file_path = en_file_path
        self.max_length = max_length

        self.de_text = pd.read_csv(self.de_file_path)
        self.en_text = pd.read_csv(self.en_file_path)

        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        de_text = self.de_text['Hindi'][idx]
        en_text = self.en_text['English'][idx]
        return de_text, en_text

    def collate_fn(self, batch):
        de_batch, en_batch = zip(*batch)
        de_batch = list(de_batch)
        en_batch = list(en_batch)

        # Encode batches and compute token ids and attention masks
        de_batch_enc = self.tokenizer(de_batch, add_special_tokens=True, padding=True)
        en_batch_enc = self.tokenizer(en_batch, add_special_tokens=True, padding=True)

        de_batch_enc_ids = torch.Tensor(de_batch_enc['input_ids']).to(torch.int64)
        en_batch_enc_ids = torch.Tensor(en_batch_enc['input_ids']).to(torch.int64)

        de_batch_enc_mask = torch.Tensor(de_batch_enc['attention_mask']).to(torch.int64)
        en_batch_enc_mask = torch.Tensor(en_batch_enc['attention_mask']).to(torch.int64)

        return de_batch_enc_ids, de_batch_enc_mask, en_batch_enc_ids, en_batch_enc_mask

    def __len__(self):
        return self.de_text.shape[0]
