import copy
import gc
import numpy as np
import os
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader
from config import configure_device, get_lr_scheduler, get_optimizer
from metrics import MeteorScore


class Trainer:
    def __init__(self, train_loader, model, train_loss, val_loader=None, lr_scheduler='poly',
        lr=0.01, eval_loss=None, eval_key=None, log_step=10, optimizer='SGD', backend='gpu',
        random_state=0, optimizer_kwargs={}, lr_scheduler_kwargs={}, **kwargs
    ):
        # Create the dataset
        self.lr = lr
        self.random_state = random_state
        self.device = configure_device(backend)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.log_step = log_step
        self.loss_profile = []

        self.model = model.to(self.device)

        # The parameter train_loss must be a callable
        self.train_criterion = train_loss

        # The parameter eval_loss must be a callable
        self.val_criterion = eval_loss

        self.optimizer = get_optimizer(optimizer, self.model, self.lr, **optimizer_kwargs)
        self.sched_type = lr_scheduler
        self.sched_kwargs = lr_scheduler_kwargs

        # Some initialization code
        torch.manual_seed(self.random_state)
        torch.set_default_tensor_type('torch.FloatTensor')
        if self.device == 'gpu':
            # Set a deterministic CuDNN backend
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def train(self, num_epochs, save_path, restore_path=None, save_criterion='min',
        epoch_save_interval=1
    ):
        assert save_criterion in ['max', 'min']

        # Configure lr scheduler
        self.lr_scheduler = get_lr_scheduler(self.optimizer, num_epochs, sched_type=self.sched_type, **self.sched_kwargs)

        # Restore checkpoint if available
        if restore_path is not None:
            # Load the model
            self.load(restore_path)

        best_eval = 0.0
        start_epoch = 0
        tk0 = range(start_epoch, num_epochs)
        for epoch_idx in tk0:
            print(f'Training for epoch: {epoch_idx + 1}')
            avg_epoch_loss = self.train_one_epoch()

            # LR scheduler step
            self.lr_scheduler.step()

            # Build loss profile
            self.loss_profile.append(avg_epoch_loss)

            # Evaluate the model
            if self.val_loader is not None:
                val_eval = self.eval()
                print(f'Avg Loss for epoch: {avg_epoch_loss} Eval Metric: {val_eval}')
                if epoch_idx == 0:
                    best_eval = val_eval
                    self.save(save_path, epoch_idx, prefix='best')
                else:
                    is_save = (best_eval > val_eval) if save_criterion == 'max' else (best_eval < val_eval)
                    if is_save:
                        # Save checkpoint
                        self.save(save_path, epoch_idx, prefix='best')
                        best_eval = val_eval
            else:
                print(f'Avg Loss for epoch:{avg_epoch_loss}')

            # Save every provided interval anyways
            if epoch_idx % epoch_save_interval == 0:
                self.save(save_path, epoch_idx)

    def train_one_epoch(self):
        self.model.train()
        epoch_loss = 0
        tk0 = tqdm(self.train_loader)
        for idx, inputs in enumerate(tk0):
            step_loss = self.train_one_step(inputs)
            epoch_loss += step_loss
            tk0.set_postfix_str(f'Avg Loss for step:{step_loss}')
        return epoch_loss/ len(self.train_loader)

    def train_one_step(self):
        raise NotImplementedError()

    def eval(self):
        raise NotImplementedError()

    def save(self, path, epoch_id, prefix=''):
        checkpoint_name = f'chkpt_{epoch_id}'
        path = os.path.join(path, prefix)
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
        state_dict['epoch'] = epoch_id + 1
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
                        state[k] = v.to(device)

        if 'scheduler' in state_dict:
            print('Restoring Learning Rate scheduler state')
            self.lr_scheduler.load_state_dict(state_dict['scheduler'])


class NMTTrainer(Trainer):
    def train_one_step(self, inputs):
        self.optimizer.zero_grad()

        de, de_attn, en, en_attn = inputs
        de = de.to(self.device)
        de_attn = de_attn.to(self.device)
        en = en.to(self.device)
        en_attn = en_attn.to(self.device)

        predictions = self.model(
            input_ids=de,
            attention_mask=de_attn,
            decoder_input_ids=en,
            decoder_attention_mask=en_attn
        )
        preds = predictions.logits.permute(0, 2, 1).contiguous()
        loss = self.train_criterion(preds, en)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def eval(self, inputs):
        self.model.eval()
        tk0 = tqdm(self.val_loader)
        meteor = MeteorScore()
        with torch.no_grad():
            for idx, inputs in enumerate(tk0):
                de, de_attn, en, en_attn = inputs
                de = de.to(self.device)
                de_attn = de_attn.to(self.device)
                en = en.to(self.device)
                en_attn = en_attn.to(self.device)

                predictions = self.model(
                    input_ids=de,
                    attention_mask=de_attn,
                    decoder_input_ids=en,
                    decoder_attention_mask=en_attn
                )

                pred_indices = nn.Softmax(dim=2)(predictions.logits)
                pred_indices = torch.argmax(pred_indices, dim=2)
                # Decode the indices using the tokenizer

                gt = self.val_loader.dataset.en_tokenizer.decode_batch(list(en.numpy()))
                preds = self.val_loader.dataset.en_tokenizer.decode_batch(list(pred_indices.numpy()))
                meteor.add(gt, preds)
            return meteor.value()
