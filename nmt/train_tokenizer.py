import csv
import json
import nltk
import numpy as np
import os
import random
import torch

from indicnlp.tokenize import indic_tokenize
from nltk.tokenize import word_tokenize
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


nltk.download('punkt')

class Vocab:
    def __init__(self, sos_token='[CLS]', eos_token='[SEP]', pad_token='[PAD]', unk_token='[UNK]', lang='hi'):
        if lang not in ['hi', 'en']:
            raise ValueError(f'Language {lang} is not supported yet!')

        self.lang = lang

        # Special symbols
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self._id = 0

        # Dicts to store token-id mapping
        self.token2id = {}
        self.id2token = {}

        # Stores unique tokens in a corpus
        self.unique_tokens = []
        self.special_tokens = [
            self.sos_token,
            self.eos_token,
            self.pad_token,
            self.unk_token
        ]

        # Add special tokens to unique tokens
        for token in self.special_tokens:
            self.add_token(token)

    def __call__(self, batch, add_special_tokens=True, padding=True, truncation=True, max_length=None, shuffle=False):
        return self.encode_batch(
            batch,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            shuffle=shuffle
        )

    def add_token(self, token):
        if token in self.unique_tokens:
            return
        self.token2id[token] = self._id
        self.id2token[self._id] = token
        self._id += 1
        self.unique_tokens.append(token)

    def fit(self, files, save_dir=None):
        # This would be language specific
        for f in files:
            print(f'Tokenizing file: {f}')
            with open(f, 'r') as fp:
                reader = csv.reader(fp)
                for row in tqdm(reader):
                    tokens = self.get_tokens(row[-1])
                    for token in tokens:
                        self.add_token(token)
        if save_dir is not None:
            self.save_pretrained(save_dir)

    def save_pretrained(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, 'vocab.json')
        with open(file_path, 'w') as fp:
            json.dump(self.token2id, fp)

    def from_pretrained(self, load_path):
        with open(load_path, 'r') as fp:
            self.token2id = json.load(fp)

        self.id2token = {v: k for k, v in self.token2id.items()}
        self.unique_tokens = list(self.token2id.keys())

        for token in self.special_tokens:
            if token not in self.unique_tokens:
                raise Warning(f'Special token {token} not found in the pretrained vocab!')

    def encode_batch(self, batch, add_special_tokens=True, padding=True, truncation=True, max_length=None, shuffle=False):
        if not isinstance(batch, list):
            raise TypeError(f'batch must be a list, found {type(batch)}')
        batch_size = len(batch)

        id_batch = []
        for sentence in batch:
            tokens = self.get_tokens(sentence)
            ids = [self.token2id.get(token, self.token2id[self.unk_token]) for token in tokens]
            
            if max_length is not None and truncation is True:
                ids = ids[:max_length]
            if shuffle is True:
                ids = random.sample(ids, k=len(ids))
            if add_special_tokens:
                ids  = [self.token2id[self.sos_token]] + ids + [self.token2id[self.eos_token]]
            id_batch.append(torch.Tensor(ids).long())

        if padding:
            id_batch = pad_sequence(id_batch, padding_value=self.token2id[self.pad_token], batch_first=True)

        # Generate attn masks
        attn_batch = (id_batch == self.token2id[self.pad_token]).long()
        return {
            'input_ids': id_batch,
            'attention_mask': attn_batch
        }

    def decode_batch(self, id_batch, skip_special_tokens=True):
        if isinstance(id_batch, torch.Tensor):
            id_batch = list(id_batch.cpu().numpy())
        if isinstance(id_batch, np.ndarray):
            id_batch = list(id_batch)

        unk_token_id = self.token2id[self.unk_token]
        sentence_batch = []
        for indices in id_batch:
            tokens = [self.id2token.get(id, unk_token_id) for id in indices]
            if skip_special_tokens:
                tokens = [token for token in tokens if token not in self.special_tokens]
            sentence_batch.append(' '.join(tokens))
        return sentence_batch

    def get_tokens(self, sentence):
        if self.lang == 'hi':
            return indic_tokenize.trivial_tokenize(sentence)
        if self.lang == 'en':
            # doc = self.nlp(sentence)
            return word_tokenize(sentence)


# NOTE: Deprecated!
def train_tokenizer(
    files, vocab_size=30522, backend='wordpiece', unk_token='[UNK]', sep_token='[SEP]', pad_token='[PAD]',
    cls_token='[CLS]', mask_token='[MASK]', save_path=None, **kwargs
):
    special_tokens = [unk_token, sep_token, pad_token, cls_token, mask_token]
    if backend == 'wordpiece':
        tokenizer = BertWordPieceTokenizer()
    elif backend == 'bytebpe':
        tokenizer = ByteLevelBPETokenizer()
    else:
        raise NotImplementedError(f'The tokenizer scheme {backend} is not yet supported!')
    tokenizer.train(files, vocab_size=vocab_size, special_tokens=special_tokens, **kwargs)
    print(f'Trained tokenizer')

    # Save the tokenizer
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        tokenizer.save_model(save_path)


if __name__ == '__main__':
    # de_train = '/home/lexent/Hin2Eng-NMT/nmt/data/cleaned/train_hindi.csv'
    en_train = '/home/lexent/Hin2Eng-NMT/nmt/data/cleaned/train_english.csv'
    # train_tokenizer([de_train, en_train], save_path='/home/lexent/Hin2Eng-NMT/nmt/data/tokenizers')

    en_vocab = Vocab(lang='en')
    # en_vocab.fit([en_train], save_dir='/home/lexent/test_nmt/')
    en_vocab.from_pretrained('/home/lexent/test_nmt/vocab.json')
    batch_ids = en_vocab.encode_batch(['in my dream i was watching', 'this is it'])
    print(batch_ids['input_ids'])
    print(batch_ids['attention_mask'])
    sentence = en_vocab.decode_batch(batch_ids['input_ids'])
    print(sentence)
