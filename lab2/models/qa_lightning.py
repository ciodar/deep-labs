import torch
import torchmetrics
from datetime import datetime

import datasets
from torch import nn
from torch.optim import AdamW
from transformers import AutoConfig, AutoModel
from typing import Optional

from lightning import LightningModule


class QATransformer(LightningModule):
    def __init__(
            self,
            model_name_or_path: str,
            num_choices: int,
            criterion: str,
            learning_rate: float = 2e-5,
            adam_epsilon: float = 1e-8,
            warmup_steps: int = 0,
            weight_decay: float = 0.0,
            **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path, config=self.config)
        self.num_choices = num_choices
        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size, 1),
        )
        self.criterion = getattr(nn, criterion)()
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_choices)
        self.valid_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_choices)

    def forward(self, **inputs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        # flatten to shape (bs * num_choices, max_len)
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        attention_mask = attention_mask.view(-1,
                                             attention_mask.shape[-1])
        # calculates hidden states with frozen Transformer model
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 output_attentions=None,
                                 output_hidden_states=None,
                                 return_dict=None)
        hidden_states = outputs[0]  # shape (bs * num_choices, max_len, hidden_size)
        pooled_output = hidden_states[:, 0]  # shape (bs * num_choices, hidden_size)
        logits = self.classifier(pooled_output)  # shape (bs * num_choices, num_labels)
        reshaped_logits = logits.view(-1, self.num_choices)  # shape (bs, num_choices, num_labels)
        return reshaped_logits

    def training_step(self, batch, batch_idx):
        labels = batch.pop("label")
        logits = self(**batch)
        loss = self.criterion(logits, labels)
        self.log('loss/train', loss, on_step=True, on_epoch=False, prog_bar=True)
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, labels)
        self.log('accuracy/train', self.train_acc, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        labels = batch.pop("label")
        logits = self(**batch)
        loss = self.criterion(logits, labels)
        self.log('loss/validation', loss, on_step=False, on_epoch=True, prog_bar=True)
        preds = torch.argmax(logits, dim=1)
        self.valid_acc(preds, labels)
        self.log('accuracy/validation', self.valid_acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        classifier = self.classifier
        optimizer = AdamW(classifier.parameters(), lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        return optimizer
