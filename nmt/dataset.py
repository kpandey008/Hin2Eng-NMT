import numpy as np
import os
import pandas as pd
import torch

from torch.utils.data import Dataset


class Hin2EngDataset(Dataset):
    # NOTE: In the code, we use the prefix `de` to denote Devanagri and `en` to denote English
    def __init__(self, root, mode='train', tokenizer, max_length=None):
        if not os.path.isdir(root):
            raise Exception(f'Path `{root}` does not exist')

        self.root = root
        self.mode = mode
        self.de_path = os.path.join(self.root, f'{self.mode}_hindi.csv')
        self.en_path = os.path.join(self.root, f'{self.mode}_english.csv')

        self.max_length = max_length
        self.de_text = None
        self.en_text = None

        if self.mode == 'train' or  self.mode == 'val':
            self.de_text = pd.read_csv(self.de_file_path)
            self.en_text = pd.read_csv(self.en_file_path)
        
        if self.mode == 'test':
            self.de_text = pd.read_csv(self.de_file_path)

        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        if self.mode == 'test':
            return self.de_text['hindi'][idx]

        de_text = self.de_text['Hindi'][idx]
        en_text = self.en_text['English'][idx]
        return de_text, en_text

    def collate_fn(self, batch):
        de_batch, en_batch = zip(*batch)
        de_batch = list(de_batch)
        en_batch = list(en_batch)

        # Encode batches and compute token ids and attention masks
        de_batch_enc = self.tokenizer(
            de_batch, add_special_tokens=True, padding=True,
            truncation=True, return_tensors='pt', max_length=self.max_length
        )
        en_batch_enc = self.tokenizer(
            en_batch, add_special_tokens=True, padding=True,
            truncation=True, return_tensors='pt', max_length=self.max_length
        )

        de_batch_ids = de_batch_enc['input_ids']
        en_batch_ids = en_batch_enc['input_ids']

        de_batch_mask = de_batch_enc['attention_mask']
        en_batch_mask = en_batch_enc['attention_mask']

        return de_batch_ids, de_batch_mask, en_batch_ids, en_batch_mask

    def decode_batch(self, batch):
        batch_ = None
        if isinstance(batch, list):
            batch_ = batch
        elif isinstance(batch, torch.Tensor):
            batch_ = list(batch.cpu().numpy())
        elif isinstance(batch, np.ndarray):
            batch_ = list(batch)
        else:
            raise ValueError('batch should be one of list, ndarray or Tensor')

        # TODO: skip_special_tokens does not work while decoding
        # so we perform decode here indirectly
        return [
            self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(b, skip_special_tokens=True))
            for b in batch_
        ]

    def __len__(self):
        return self.de_text.shape[0]
