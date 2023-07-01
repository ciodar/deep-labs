import torch
import torchmetrics
from datetime import datetime

import datasets
from torch import nn
from torch.optim import AdamW
from transformers import AutoConfig, AutoModel
from typing import Optional, Union

from lightning import LightningModule


class MLP(LightningModule):
    def __init__(
            self,
            num_classes: int,
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

        self.num_classes = num_classes
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
        )
        self.criterion = getattr(nn, criterion)() if type(criterion) == str else criterion
        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.valid_acc = torchmetrics.Accuracy(task="binary")

    def forward(self, x):
       
        return self.classifier(x).squeeze()

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self(inputs)
        probs = torch.sigmoid(logits)
        loss = self.criterion(probs, labels.float())
        self.log('loss/train', loss, on_step=True, on_epoch=False, prog_bar=True)
        self.train_acc(labels, probs > 0.5)
        self.log('accuracy/train', self.train_acc, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, labels = batch
        logits = self(inputs)
        probs = torch.sigmoid(logits)
        loss = self.criterion(probs, labels.float())
        self.log('loss/validation', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.valid_acc(labels, probs > 0.5)
        self.log('accuracy/validation', self.valid_acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        return optimizer
