import numpy as np
import os
import random
import torch
import warnings

from tokenizers import Tokenizer
from transformers import EncoderDecoderModel
from trainer import NMTTrainer


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
    return seed


# NOTE: Keeping the model as a param here as it will change a lot during experimentation
# TODO: Complete this implementation after experimentation is done
def train_nmt(
    train_en_path, train_de_path, model
    en_tokenizer_path, de_tokenizer_path, n_epochs=100, code_size=10, device='cpu',
    batch_size=32, random_state=0, save_path=os.getcwd(), trainer_kwargs={}
):
    # Load tokenizers
    de_tokenizer = Tokenizer.from_file("de_tokenizer_path")
    en_tokenizer = Tokenizer.from_file("en_tokenizer_path")

    # Dataset
    train_dataset = Hin2EngDataset(train_de_path, train_en_path, de_tokenizer, en_tokenizer)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_dataset.collate_fn)

    # Train Loss
    train_loss = nn.CrossEntropyLoss(ignore_index=en_tokenizer.token_to_id('[PAD]'))

    # Trainer
    trainer = NMTTrainer(train_loader, model, train_loss, backend=device, random_state=random_state)
    trainer.train(n_metric_epochs, save_path)
