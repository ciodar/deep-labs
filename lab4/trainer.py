import numpy as np
import torchmetrics
import torch.nn.functional as F
import torch
from matplotlib import pyplot as plt
from torch import nn
from torchmetrics import Accuracy
from tqdm import tqdm

from lab4.fgsm import fgsm


class Trainer:
    def __init__(self, optimizer, writer, lr_scheduler=None, epochs=100, device='cuda'):
        self.optimizer = optimizer
        self.writer = writer
        self.lr_scheduler = lr_scheduler
        self.criterion = F.cross_entropy
        self.epochs = epochs
        self.device = device
        # initialize metrics (for now it's just top-1 accuracy)
        self.train_metrics = nn.ModuleDict({'train_accuracy': Accuracy(task="multiclass", num_classes=10),
                                            'adv_train_accuracy': Accuracy(task="multiclass", num_classes=10)
                                            })
        self.valid_metrics = nn.ModuleDict({'val_accuracy': Accuracy(task="multiclass", num_classes=10),
                                            'adv_val_accuracy': Accuracy(task="multiclass", num_classes=10)
                                            })

    def train(self, model, train_loader, valid_loader=None):
        self.log_freq = int(np.sqrt(train_loader.batch_size))
        self.writer.watch(model, self.log_freq)
        train_results, val_results = [], []
        for epoch in range(self.epochs + 1):
            train_log = self._train_epoch(model, train_loader, epoch)
            # print(f"Epoch: {epoch} | loss: {train_log[0]} | accuracy: {train_log[1]}")
            train_results.append(train_log)
            if valid_loader is not None:
                val_log = self._valid_epoch(model, valid_loader, epoch)
                val_results.append(val_log)
        self.writer.finish()
        return train_results, val_results

    def _get_adv_examples(self, model, input, target, eps):
        # Watch out! Uses original image
        input.requires_grad = True
        logits = model(input)
        init_pred = torch.argmax(logits, dim=1)
        # If the initial prediction is wrong, attack and gradient could be wrong?
        loss = F.cross_entropy(logits, target)
        # zeroes gradients of all parameters
        self.optimizer.zero_grad()
        loss.backward()
        input_grad = input.grad.data
        # mask = torch.nonzero((init_pred != ys).int())
        # image_grad[mask] = torch.zeros(image_grad[mask].shape[1:])
        # print(image_grad.shape)
        adv_input = fgsm(input, input_grad, eps)
        # Xs_adv = transforms.Normalize(NORM_MEAN, NORM_STD)(Xs_adv)
        # Re-classify the perturbed image
        output = model(adv_input)

        # Check for success
        final_pred = torch.argmax(output, dim=1) # get the index of the max log-probability
        return adv_input.detach(), final_pred

    def _train_epoch(self, model, data_loader, epoch):
        model.train()
        losses = []
        train_logs = {
            'train_loss': [],
            'train_accuracy': [],
            'adv_train_accuracy': []
        }
        self.train_metric.reset()
        pbar = self._get_pbar()
        pbar.reset(total=len(data_loader))
        pbar.initial = 0
        pbar.set_description(f"Epoch {epoch}")
        for batch_idx, (input, target) in enumerate(data_loader):
            input, target = input.to(self.device), target.to(self.device)
            output = self._get_adv_examples(input)

            loss = self.criterion(output, target, )
            loss.backward()
            self.optimizer.step()

            preds = torch.argmax(output, dim=1)
            acc = self.train_metric(preds, target)
            if batch_idx % self.log_freq == 0:
                pbar.set_postfix(step_loss=loss.item(), step_accuracy=acc.item())
                self.writer.log({"train_loss": loss.item(), "train_accuracy": acc.item()})
            losses.append(loss.item())
            pbar.update()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        train_logs['train_loss'] = np.mean(losses)
        train_acc = self.train_metric.compute().item()
        pbar.set_postfix(train_loss=train_loss, train_acc=train_acc)
        pbar.refresh()
        return train_loss, train_acc

    def _valid_epoch(self, model, data_loader, epoch):
        model.eval()

        self.valid_metric.reset()
        losses = []
        pbar = self._get_pbar()
        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(data_loader):
                input, target = input.to(self.device), target.to(self.device)
                output = model(input)

                loss = self.criterion(output, target)

                preds = torch.argmax(output, dim=1)
                acc = self.valid_metric(preds, target)
                losses.append(loss.item())
        val_loss = np.mean(losses)
        val_acc = self.valid_metric.compute().item()
        self.writer.log({"val_loss": val_loss, "val_acc": val_acc})
        pbar.set_postfix(valid_loss=val_loss, valid_acc=val_acc)
        pbar.refresh()
        # print(f"Epoch: {epoch} | valid loss: {val_loss} | valid accuracy: {val_acc}")
        return val_loss, val_acc

    def _get_pbar(self):
        if not hasattr(self, 'pbar'):
            self.pbar = tqdm(leave=True)
        return self.pbar


# Simple function to plot the loss curve and validation accuracy.
def plot_validation_curves(losses_and_accs):
    losses = [x for (x, _) in losses_and_accs]
    accs = [x for (_, x) in losses_and_accs]
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Average Training Loss per Epoch')
    plt.subplot(1, 2, 2)
    plt.plot(accs)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title(f'Best Accuracy = {np.max(accs)} @ epoch {np.argmax(accs)}')
