import torch
import torchmetrics
from datetime import datetime

import datasets
from torch import nn
from torch.optim import AdamW
from transformers import AutoConfig, AutoModel
from typing import Optional, Union

from lightning import LightningModule


class QAMLP(LightningModule):
    def __init__(
            self,
            num_choices : int,
            input_size : int,
            hidden_size : int,
            criterion: Union[str, nn.Module] = "cross_entropy",
            learning_rate: float = 2e-5,
            adam_epsilon: float = 1e-8,
            warmup_steps: int = 0,
            weight_decay: float = 0.0,
            **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.num_choices = num_choices
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
        )
        self.criterion = getattr(nn, criterion)() if type(criterion) == str else criterion
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_choices)
        self.valid_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_choices)

    def forward(self, x):
        
        num_choices = x.size(1)

        # flatten to shape (bs * num_choices, hidden_size)
        x = x.view(-1, x.size(-1))
       
        # calculate logits
        outputs = self.classifier(x)
        reshaped_out = outputs.view(-1, num_choices) # shape (bs, num_choices)
        return reshaped_out

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self(inputs)

        loss = self.criterion(logits, labels)
        self.log('loss/train', loss, on_step=True, on_epoch=False, prog_bar=True)
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, labels)
        self.log('accuracy/train', self.train_acc, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, labels = batch
        logits = self(inputs)
        loss = self.criterion(logits, labels)
        self.log('loss/validation', loss, on_step=False, on_epoch=True, prog_bar=True)
        preds = torch.argmax(logits, dim=1)
        self.valid_acc(preds, labels)
        self.log('accuracy/validation', self.valid_acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        return optimizer
