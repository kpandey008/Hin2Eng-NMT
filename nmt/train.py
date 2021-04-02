import numpy as np
import os
import random
import torch
import torch.nn as nn
import warnings

from tokenizers import Tokenizer
from transformers import EncoderDecoderModel, BertConfig, EncoderDecoderConfig, BertForMaskedLM, BertTokenizerFast

from dataset import Hin2EngDataset
from trainer import TransformersForNmtTrainer


def seed_everything(seed=0):
    # NOTE: Uses the implementation from Pytorch Lightning
    """Function that sets seed for pseudo-random number generators  in:
        pytorch, numpy, python.random
    """
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min

    if (seed > max_seed_value) or (seed < min_seed_value):
        raise ValueError(f"{seed} is not in bounds, \
            numpy accepts from {min_seed_value} to {max_seed_value}"
        )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_default_tensor_type('torch.FloatTensor')
    
    # Set a deterministic CuDNN backend
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def train_nmt(
    vocab_path, train_en_path, train_de_path, val_de_path=None, val_en_path=None,
    n_epochs=10, device='cpu', batch_size=32, random_state=0, save_path=os.getcwd(),
    **kwargs
):
    # Seed
    seed_everything(seed=random_state)

    # Load tokenizers
    tokenizer = BertTokenizerFast(vocab_path)

    # Dataset and loaders
    train_dataset = Hin2EngDataset(train_de_path, train_en_path, tokenizer, max_length=512)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_dataset.collate_fn, drop_last=True, shuffle=True)

    val_loader = None
    if (val_de_path is not None) and (val_en_path is not None):
        val_dataset = Hin2EngDataset(val_de_path, val_en_path, tokenizer, max_length=512)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, collate_fn=val_dataset.collate_fn, drop_last=False, shuffle=False)

    # Model
    model = EncoderDecoderModel.from_encoder_decoder_pretrained('google/bert_uncased_L-8_H-512_A-8', 'google/bert_uncased_L-8_H-512_A-8')

    # Trainer
    n_epochs = 10
    trainer = TransformersForNmtTrainer(
        train_loader,
        model,
        n_epochs,
        val_loader=val_loader,
        backend=device,
        results_dir=save_path,
        **kwargs
    )
    trainer.train()
