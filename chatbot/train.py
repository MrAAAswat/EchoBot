import os
import re
import math
import numpy as np
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_polynomial_decay_schedule_with_warmup

from data import DialoguesDataset
from utils import PadCollate


class Trainer:
    def __init__(self, model, args):
        print('Loading the optimizer...')
        self.optimizer = AdamW(model.parameters(), lr=args['lr'])
        self.best_loss = 1e+10
        self.last_epoch = 0
        self.last_batch = 0

        print('Loading train & valid data...')
        train_dataset = DialoguesDataset('train', args)
        valid_dataset = DialoguesDataset('valid', args)
        pad = PadCollate(args)

        self.train_loader = DataLoader(train_dataset,
                                       collate_fn=pad,
                                       shuffle=True,
                                       batch_size=args['batch_size'],
                                       num_workers=2,
                                       pin_memory=True)
        self.valid_loader = DataLoader(valid_dataset,
                                       collate_fn=pad,
                                       batch_size=args['batch_size'],
                                       num_workers=2,
                                       pin_memory=True)

        if not os.path.exists(args['models_dir']):
            os.makedirs(args['models_dir'])

        # Calculate total training steps
        num_batches = len(self.train_loader)
        total_train_steps = args['num_epochs'] * num_batches
        warmup_steps = int(args['warmup_ratio'] * total_train_steps)

        self.model = model
        self.args = args
        self.scheduler = get_polynomial_decay_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_train_steps,
            power=2
        )

        if args['checkpoint']:
            self._load_checkpoint()
        else:
            latest_checkpoint = self._find_latest_checkpoint()
            if latest_checkpoint:
                self._load_checkpoint(latest_checkpoint)

    def train(self):
        print('Launching training...')

        # Check if resuming from a checkpoint
        is_resuming = self.last_epoch > 0 or self.last_batch > 0

        if is_resuming:
            start_epoch = self.last_epoch
            print(f"Resuming training from epoch {start_epoch}, batch {self.last_batch + 1}...")
        else:
            start_epoch = 1  # Start from epoch 1 if there's no checkpoint
            print(f"Starting training from scratch, beginning at epoch {start_epoch}...")

        for epoch in range(start_epoch, self.args['num_epochs'] + 1):
            print('-' * 50 + f'\nEpoch: {epoch}\n' + '-' * 50)

            self.model.train()
            train_losses = []
            train_perplexity = []

            checkpoint_interval = 200  # Save checkpoint every 200 batches

            # Skip previously processed batches if resuming
            for i, batch in enumerate(tqdm(self.train_loader)):
                # Skip batches up to the last batch processed in this epoch
                if is_resuming and epoch == self.last_epoch and i <= self.last_batch:
                    continue

                # Turn off resuming logic after skipping batches
                is_resuming = False

                input_ids, token_type_ids, labels = batch
                input_ids = input_ids.to(self.args['device'])
                token_type_ids = token_type_ids.to(self.args['device'])
                labels = labels.to(self.args['device'])

                outputs = self.model(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    labels=labels
                )

                loss, logits = outputs[0], outputs[1]

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                train_losses.append(loss.detach())
                ppx = torch.exp(loss.detach())
                train_perplexity.append(ppx)

                # Update the last batch processed
                self.last_batch = i

                # Save the checkpoint every 'checkpoint_interval' batches
                if (i + 1) % checkpoint_interval == 0:
                    state_dict = {
                        'model_state_dict': self.model.state_dict(),
                        'optim_state_dict': self.optimizer.state_dict(),
                        'loss': np.mean([loss.item() for loss in train_losses]),
                        'epoch': epoch,  # Save the current epoch
                        'batch': i + 1  # Save the current batch
                    }
                    batch_checkpoint_filename = f"{self.args['models_dir']}/checkpoint_epoch_{epoch}_batch_{i + 1}.h5"
                    torch.save(state_dict, batch_checkpoint_filename)
                    print(f'Checkpoint for batch {i + 1} in epoch {epoch} saved: {batch_checkpoint_filename}')

            train_losses = [loss.item() for loss in train_losses]
            train_perplexity = [ppx.item() if not math.isinf(ppx.item()) else 1e+8 for ppx in train_perplexity]
            train_loss = np.mean(train_losses)
            train_ppx = np.mean(train_perplexity)
            print(f'Train loss: {train_loss} \nTrain perplexity: {train_ppx}')

            valid_loss, valid_ppx = self.validate()

            if valid_loss < self.best_loss:
                self.best_loss = valid_loss
                state_dict = {
                    'model_state_dict': self.model.state_dict(),
                    'optim_state_dict': self.optimizer.state_dict(),
                    'loss': self.best_loss,
                    'epoch': epoch
                }

                filename = f"{self.args['models_dir']}/model_best_{round(self.best_loss, 4)}.h5"
                torch.save(state_dict, filename)
                print(f'Checkpoint saved: {filename}')

            print(f'Best valid loss: {self.best_loss}')
            print(f'Valid loss: {valid_loss} \nValid perplexity: {valid_ppx}')

        print('Training completed')




    def validate(self):
        print('Launch validation...')
        self.model.eval()

        valid_losses = []
        valid_ppxs = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.valid_loader)):
                input_ids, token_type_ids, labels = batch
                input_ids = input_ids.to(self.args['device'])
                token_type_ids = token_type_ids.to(self.args['device'])
                labels = labels.to(self.args['device'])

                outputs = self.model(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    labels=labels
                )

                loss, logits = outputs[0], outputs[1]

                valid_losses.append(loss.detach())
                ppx = torch.exp(loss.detach())
                valid_ppxs.append(ppx)

            valid_losses = [loss.item() for loss in valid_losses]
            valid_ppxs = [ppx.item() if not math.isinf(ppx.item()) else 1e+8 for ppx in valid_ppxs]
            valid_loss = np.mean(valid_losses)
            valid_ppx = np.mean(valid_ppxs)

            if math.isnan(valid_ppx):
                valid_ppx = 1e+8

        return valid_loss, valid_ppx
    
    def _find_latest_checkpoint(self):
        # Find the latest checkpoint by scanning the models directory
        checkpoints = [f for f in os.listdir(self.args['models_dir']) if f.endswith(".h5")]
        if checkpoints:
            # Sort checkpoints by epoch (extract from filename)
            checkpoints.sort(key=lambda f: int(re.findall(r'epoch_(\d+)', f)[0]), reverse=True)
            latest_checkpoint = os.path.join(self.args['models_dir'], checkpoints[0])
            print(f"Found latest checkpoint: {latest_checkpoint}")
            return latest_checkpoint
        print("No checkpoints found.")
        return None
    

    def _load_checkpoint(self, path):
        # Corrected: accept path argument
        if os.path.exists(path):
            print(f'Loading checkpoint from {path}...')
            checkpoint = torch.load(path, map_location=self.args['device'])
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optim_state_dict'])
            self.best_loss = checkpoint['loss']
            self.last_epoch = checkpoint['epoch']
            self.last_batch = checkpoint.get('batch', 0)

            print(f'Resuming training from epoch {self.last_epoch}.')
        else:
            print(f"Checkpoint not found at {path}")