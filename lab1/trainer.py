import os

import numpy as np
import torchmetrics
import torch.nn.functional as F
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from wandb.beta.workflows import log_model, link_model

import wandb

USE_BETA_APIS = False


class Trainer:
    def __init__(self, optimizer, writer, lr_scheduler=None, epochs=100, device='cuda', checkpoints=False,
                 dataset_name="cifar10", model_collection_name="CIFAR-10 ResNet", **kwargs):
        self.optimizer = optimizer
        self.writer = writer
        self.lr_scheduler = lr_scheduler
        self.criterion = F.cross_entropy
        self.epochs = epochs
        self.device = device
        # initialize metrics (for now it's just top-1 accuracy)
        self.train_metric = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)
        self.valid_metric = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)
        # placeholder for pbar
        self._train_acc, self._train_loss, self._val_acc, self._val_loss = None, None, None, None
        # model checkpoint saving
        self.checkpoints = checkpoints
        self.dataset_name = dataset_name
        self.model_collection_name = model_collection_name
        if self.checkpoints:
            self.checkpoint_dir = writer.dir
            self.best_loss, self.best_model = np.inf, None

    def train(self, model, train_loader, valid_loader=None):
        # reset best stats
        self.best_loss = np.inf
        self.best_model = None

        self.log_freq = int(np.sqrt(train_loader.batch_size))
        if self.writer is not None:
            self.writer.watch(model, self.log_freq)
        train_results, val_results = [], []
        for epoch in range(1, self.epochs + 1):
            self._train_loss, self._train_acc = self._train_epoch(model, train_loader, epoch)
            train_results.append((self._train_loss, self._train_acc))
            # validate model
            if valid_loader is not None:
                self._val_loss, self._val_acc = self._valid_epoch(model, valid_loader, epoch)
                val_results.append((self._val_loss, self._val_acc))
            # evaluate model, save best checkpoint
            if self.checkpoints and self._val_loss < self.best_loss:
                self.best_loss = self._val_loss
                self._save_checkpoint(model, epoch, True)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
        if self.writer is not None:
            if self.checkpoints:
                if USE_BETA_APIS:
                    link_model(self.best_model, self.model_collection_name)
                else:
                    wandb.run.link_artifact(self.best_model, self.model_collection_name, ["latest"])
            self.writer.finish()
        return train_results, val_results

    def _save_checkpoint(self, model, epoch, best=False):
        arch = type(model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': dict(self.writer.config)
        }

        # print(f"Saving into {filename} ...")
        if USE_BETA_APIS:
            model_version = log_model(model, self.dataset_name, ["best"] if best else None)
            if best:
                self.best_model = model_version
        else:
            art = wandb.Artifact(f"{self.dataset_name}-{self.writer.id}", "model")
            filename = str(os.path.join(self.checkpoint_dir, 'ckpt-best.pth'.format(epoch)))
            torch.save(state, filename)
            art.add_file(filename)
            wandb.log_artifact(art, aliases=["best", "latest"] if best else None)
            if best:
                self.best_model = art

    def _train_epoch(self, model, data_loader, epoch):
        model.train()

        losses = []
        self.train_metric.reset()
        pbar = self._get_pbar()
        pbar.reset(total=len(data_loader))
        pbar.initial = 0
        pbar.set_description(f"Epoch {epoch}")
        for batch_idx, (input, target) in enumerate(data_loader):
            input, target = input.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = model(input)

            loss = self.criterion(output, target, )
            loss.backward()
            self.optimizer.step()

            preds = torch.argmax(output, dim=1)
            acc = self.train_metric(preds, target)
            step_loss, step_acc = loss.item(), acc.item()
            if batch_idx % self.log_freq == 0:
                pbar.set_postfix(step_loss=step_loss, step_acc=step_acc, train_loss=self._train_loss,
                                 train_acc=self._train_acc, valid_loss=self._val_loss, valid_acc=self._val_acc)
                if self.writer:
                    self.writer.log({"train_loss": loss.item(), "train_accuracy": acc.item()})
            losses.append(step_loss)
            pbar.update()
        train_loss = np.mean(losses)
        train_acc = self.train_metric.compute().item()
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
                self.valid_metric.update(preds, target)
                losses.append(loss.item())
        val_loss = np.mean(losses)
        val_acc = self.valid_metric.compute().item()
        if self.writer:
            self.writer.log({"val_loss": val_loss, "val_acc": val_acc})
        # print(f"Epoch: {epoch} | valid loss: {val_loss} | valid accuracy: {val_acc}")
        return val_loss, val_acc

    def _get_pbar(self):
        if not hasattr(self, 'pbar'):
            self.pbar = tqdm(leave=True)
        return self.pbar

    # Return accuracy score and classification report.
    # return (accuracy_score(np.hstack(gts), np.hstack(predictions)),
    #         classification_report(np.hstack(gts), np.hstack(predictions), zero_division=0, digits=3))


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
