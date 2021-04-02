import copy
import gc
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader
from config import configure_device, get_lr_scheduler, get_optimizer
from metrics import MeteorScore, BLEUScore


class Trainer:
    def __init__(self, train_loader, model, num_epochs, val_loader=None, lr_scheduler='poly',
        lr=0.0001, log_step=50, n_train_steps_per_epoch=None, optimizer='Adam', backend='gpu',
        results_dir=None, n_val_steps=None, optimizer_kwargs={}, lr_scheduler_kwargs={}, **kwargs
    ):
        # Create the dataset
        self.train_loader = train_loader
        self.num_epochs = num_epochs
        self.lr = lr
        self.device = configure_device(backend)
        self.val_loader = val_loader
        self.log_step = log_step
        self.loss_profile = []
        self.n_train_steps_per_epoch = n_train_steps_per_epoch
        self.n_val_steps = n_val_steps
        self.train_progress_bar = None
        self.val_progress_bar = None
        self.results_dir = results_dir

        if (self.results_dir is not None) and (not os.path.isdir(self.results_dir)):
            os.makedirs(self.results_dir, exist_ok=True)

        self.model = model.to(self.device)

        self.optimizer = get_optimizer(optimizer, self.model, self.lr, **optimizer_kwargs)
        self.sched_type = lr_scheduler
        self.sched_kwargs = lr_scheduler_kwargs
        self.start_epoch = 0

        # Call some custom initialization here
        self.init()

    def init(self):
        pass

    def train(self, restore_path=None):
        # Configure lr scheduler
        self.lr_scheduler = get_lr_scheduler(self.optimizer, self.num_epochs, sched_type=self.sched_type, **self.sched_kwargs)

        # Restore checkpoint if available
        if restore_path is not None:
            # Load the model
            self.load(restore_path)

        best_eval = 0.0
        tk0 = range(self.start_epoch, self.num_epochs)

        self.epoch_idx = 0
        for _ in tk0:
            print(f'Training for epoch: {self.epoch_idx + 1}')
            avg_epoch_loss = self.train_one_epoch()
            print(f'Avg Loss for epoch:{avg_epoch_loss}')

            # LR scheduler step
            self.lr_scheduler.step()

            # Build loss profile
            self.loss_profile.append(avg_epoch_loss)

            # Evaluate the model
            if self.val_loader is not None:
                self.eval()

            self.epoch_idx += 1

    def train_one_epoch(self):
        self.model.train()
        epoch_loss = 0
        self.train_progress_bar = tqdm(self.train_loader)
        for idx, inputs in enumerate(self.train_progress_bar):
            if self.n_train_steps_per_epoch is not None and \
                (idx + 1) > self.n_train_steps_per_epoch:
                break
            step_loss = self.train_step(inputs)
            self.on_train_step_end()
            epoch_loss += step_loss
            if idx % self.log_step == 0:
                self.train_progress_bar.set_postfix_str(f'Avg Loss for step {idx + 1} : {step_loss}')

        self.on_train_epoch_end()
        return epoch_loss/ len(self.train_loader)

    def train_step(self):
        raise NotImplementedError()

    def on_train_epoch_end(self):
        pass

    def on_train_step_end(self):
        pass

    def eval(self):
        self.model.eval()
        self.val_progress_bar = tqdm(self.val_loader)
        with torch.no_grad():
            for idx, inputs in enumerate(self.val_progress_bar):
                if self.n_val_steps is not None and \
                    (idx + 1) > self.n_val_steps:
                    break
                self.val_step(inputs)
                self.on_val_step_end()
            self.on_val_epoch_end()

    def val_step(self):
        raise NotImplementedError()

    def on_val_epoch_end(self):
        pass

    def on_val_step_end(self):
        pass

    def save(self, path, name, prefix=None):
        checkpoint_name = f'{name}_{prefix}' if prefix is not None else name
        path = path if prefix is None else os.path.join(path, prefix)
        checkpoint_path = os.path.join(path, f'{checkpoint_name}.pt')
        state_dict = {}
        model_state = copy.deepcopy(self.model.state_dict())
        model_state = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model_state.items()}
        optim_state = copy.deepcopy(self.optimizer.state_dict())
        for state in optim_state['state'].values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cpu()

        state_dict['model'] = model_state
        state_dict['optimizer'] = optim_state
        state_dict['scheduler'] = self.lr_scheduler.state_dict()
        state_dict['epoch'] = self.epoch_idx + 1
        state_dict['loss_profile'] = self.loss_profile

        os.makedirs(path, exist_ok=True)
        for f in os.listdir(path):
            if f.endswith('.pt'):
                os.remove(os.path.join(path, f))
        torch.save(state_dict, checkpoint_path)
        del model_state, optim_state
        gc.collect()

    def load(self, load_path):
        state_dict = torch.load(load_path)
        iter_val = state_dict.get('epoch', 0)
        self.loss_profile = state_dict.get('loss_profile', [])
        if 'model' in state_dict:
            print('Restoring Model state')
            self.model.load_state_dict(state_dict['model'])

        if 'optimizer' in state_dict:
            print('Restoring Optimizer state')
            self.optimizer.load_state_dict(state_dict['optimizer'])
            # manually move the optimizer state vectors to device
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

        if 'scheduler' in state_dict:
            print('Restoring Learning Rate scheduler state')
            self.lr_scheduler.load_state_dict(state_dict['scheduler'])


class TransformersForNmtTrainer(Trainer):
    def init(self):
        self.tokenizer = self.train_loader.dataset.tokenizer
        self.meteor_score = MeteorScore()
        self.best_score = 0
        self.chkpt_name = 'nmt_chkpt'

    def train_step(self, inputs):
        self.optimizer.zero_grad()
        de, de_attn, en, en_attn = inputs
        de = de.to(self.device)
        de_attn = de_attn.to(self.device)
        en = en.to(self.device)
        en_attn = en_attn.to(self.device)
        lm_labels = en.clone()

        predictions_ = self.model(
            input_ids=de,
            attention_mask=de_attn,
            decoder_input_ids=en,
            decoder_attention_mask=en_attn,
        )
        predictions = predictions_.logits
        predictions = predictions[:, :-1, :].contiguous()
        targets = en[:, 1:]

        rearranged_output = predictions.view(predictions.shape[0]*predictions.shape[1], -1)
        rearranged_target = targets.contiguous().view(-1)

        loss = F.cross_entropy(rearranged_output, rearranged_target, ignore_index=self.tokenizer.convert_tokens_to_ids('[PAD]'))
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def val_step(self, inputs):
        meteor = MeteorScore()
        de, de_attn, en, en_attn = inputs
        de = de.to(self.device)
        de_attn = de_attn.to(self.device)
        en = en.to(self.device)
        en_attn = en_attn.to(self.device)

        predictions = self.model.generate(
            input_ids=de,
            attention_mask=de_attn,
            decoder_start_token_id=self.tokenizer.convert_tokens_to_ids('[CLS]'),
        )

        # Decode the indices using the tokenizer
        gt = self.val_loader.dataset.decode_batch(list(en.cpu().numpy()))
        preds = self.val_loader.dataset.decode_batch(list(predictions.cpu().numpy()))
        self.meteor_score.add(gt, preds)

    def on_val_epoch_end(self):
        avg_meteor = self.meteor_score.value()
        print(f'Average Meteor score: {avg_meteor}')

        if self.best_score > avg_meteor:
            self.best_score = avg_meteor
            if self.results_dir is not None:
                self.save(self.results_dir, self.chkpt_name, prefix='best')

        # Reset the metric
        self.meteor_score.reset()
