import numpy as np
import torchmetrics
import torch.nn.functional as F
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm


class Trainer:
    def __init__(self, optimizer, writer, lr_scheduler=None, epochs=100, device='cuda'):
        self.optimizer = optimizer
        self.writer = writer
        self.lr_scheduler = lr_scheduler
        self.criterion = F.cross_entropy
        self.epochs = epochs
        self.device = device
        # initialize metrics (for now it's just top-1 accuracy)
        self.train_metric = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)
        self.valid_metric = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)

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
            if batch_idx % self.log_freq == 0:
                pbar.set_postfix(step_loss=loss.item(), step_accuracy=acc.item())
                self.writer.log({"train_loss": loss.item(), "train_accuracy": acc.item()})
            losses.append(loss.item())
            pbar.update()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        train_loss = np.mean(losses)
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
