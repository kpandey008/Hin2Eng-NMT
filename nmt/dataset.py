import csv
import numpy as np
import os
import torch

from torch.utils.data import Dataset


class Hin2EngDataset(Dataset):
    # NOTE: In the code, we use the prefix `de` to denote Devanagri and `en` to denote English
    # Also this dataset is written to work with cleaned data generated using preprocessing.
    def __init__(self, root, de_vocab, en_vocab, mode='train', max_length=None):
        if not os.path.isdir(root):
            raise Exception(f'Path `{root}` does not exist')

        self.root = root
        self.mode = mode
        self.de_vocab = de_vocab
        self.en_vocab = en_vocab
        self.de_path = os.path.join(self.root, f'{self.mode}_hindi.csv')
        self.en_path = os.path.join(self.root, f'{self.mode}_english.csv')

        self.max_length = max_length
        self.de_text = []
        self.en_text = []

        if self.mode == 'train' or  self.mode == 'val':
            with open(self.de_path, 'r') as de, open(self.en_path, 'r') as en:
                de_reader = csv.reader(de)
                en_reader = csv.reader(en)
                for de_row, en_row in zip(de_reader, en_reader):
                    self.de_text.append(de_row[-1])
                    self.en_text.append(en_row[-1])
        
        if self.mode == 'test':
            with open(self.de_path, 'r') as de:
                de_reader = csv.reader(de)
                for de_row in de_reader:
                    self.de_text.append(de_row[-1])

        self.de_text = self.de_text[1:]
        self.en_text = self.en_text[1:]

    def __getitem__(self, idx):
        if self.mode == 'test':
            return self.de_text[idx]

        de_text = self.de_text[idx]
        en_text = self.en_text[idx]
        return de_text, en_text

    def collate_fn(self, batch):
        if self.mode == 'test':
            de_batch = list(batch)
            de_batch_enc = self.de_vocab(
                de_batch, add_special_tokens=True, padding=True,
                truncation=True, max_length=self.max_length
            )
            de_batch_ids = de_batch_enc['input_ids']
            de_batch_mask = de_batch_enc['attention_mask']
            return de_batch_ids, de_batch_mask

        de_batch, en_batch = zip(*batch)
        de_batch = list(de_batch)
        en_batch = list(en_batch)

        # Encode batches and compute token ids and attention masks
        de_batch_enc = self.de_vocab(
            de_batch, add_special_tokens=True, padding=True,
            truncation=True, max_length=self.max_length
        )
        en_batch_enc = self.en_vocab(
            en_batch, add_special_tokens=True, padding=True,
            truncation=True, max_length=self.max_length
        )

        # Shuffle and encode batches
        de_batch_enc_shuffled = self.de_vocab(
            de_batch, add_special_tokens=True, padding=True,
            truncation=True, max_length=self.max_length, shuffle=True
        )
        en_batch_enc_shuffled = self.en_vocab(
            en_batch, add_special_tokens=True, padding=True,
            truncation=True, max_length=self.max_length, shuffle=True
        )

        de_batch_ids = de_batch_enc['input_ids']
        de_batch_ids_shuffled = de_batch_enc_shuffled['input_ids']
        en_batch_ids = en_batch_enc['input_ids']
        en_batch_ids_shuffled = en_batch_enc_shuffled['input_ids']

        de_batch_mask = de_batch_enc['attention_mask']
        de_batch_mask_shuffled = de_batch_enc_shuffled['attention_mask']
        en_batch_mask = en_batch_enc['attention_mask']
        en_batch_mask_shuffled = en_batch_enc_shuffled['attention_mask']

        return {
            'original': {
                'de_ids': de_batch_ids,
                'de_mask': de_batch_mask,
                'en_ids': en_batch_ids,
                'en_mask': en_batch_mask
            },
            'shuffled': {
                'de_ids': de_batch_ids_shuffled,
                'de_mask': de_batch_mask_shuffled,
                'en_ids': en_batch_ids_shuffled,
                'en_mask': en_batch_mask_shuffled
            }
        }

    def __len__(self):
        return len(self.de_text)
