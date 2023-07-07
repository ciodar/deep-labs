import wandb
import wandb.beta.workflows as wf 

import torch
from torch import nn
import torch.nn.functional as F 
import torchvision
import torchmetrics as tm

import os
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np

from fgsm import fgsm
from utils import get_pil_image

# define label convention
real_label = 1.
fake_label = 0.

USE_BETA_APIS = False

class Trainer:
    def __init__(
                self, 
                optimizers: dict, 
                writer, 
                epochs: int = 10,
                update_d_every_n_steps: int = 1, 
                device: str = 'cpu',
                at1: bool = False,
                add_noise: bool = True,
                epsilon: float = 0.1,
                lambda_adv: float = 1.0
                ):
        self.optimizerG = optimizers['generator']
        self.optimizerD = optimizers['discriminator']
        self.writer = writer
        self.checkpoint_dir = writer.dir
        self.epochs = epochs
        self.device = device
        # Discriminator hyperparameters
        self.criterion_d = nn.BCELoss()
        self.update_d_every_n_steps = update_d_every_n_steps
        # Generator hyperparameters
        self.epsilon = epsilon
        self.at1 = at1
        self.lambda_ = lambda_adv
        self.add_noise = add_noise
        self.dataset_name = "cifar-10"
        
        # initialize metrics (for now it's just top-1 accuracy)
        self.train_metrics = nn.ModuleDict({'train_accuracy_cls': tm.Accuracy(task="multiclass", num_classes=10),
                                            'train_accuracy_adv': tm.Accuracy(task="multiclass", num_classes=10)
                                            }).to(self.device)
        self.valid_metrics = nn.ModuleDict({'val_accuracy_cls': tm.Accuracy(task="multiclass", num_classes=10),
                                            'val_accuracy_adv': tm.Accuracy(task="multiclass", num_classes=10)
                                            }).to(self.device)

    def train(self, model, train_loader, valid_loader=None):
        self.log_freq = int(np.sqrt(train_loader.batch_size))
        self.writer.watch(model[0], self.log_freq)
        train_results, val_results = [], []
        for epoch in range(1, self.epochs + 1):
            train_log = self._train_epoch(model, train_loader, epoch)
            # print(f"Epoch: {epoch} | loss: {train_log[0]} | accuracy: {train_log[1]}")
            train_results.append(train_log)
            if valid_loader is not None:
                val_log = self._valid_epoch(model, valid_loader, epoch)
                val_results.append(val_log)
                self.writer.log(val_log)
        self._save_checkpoint(model, epoch, False)
        self.writer.finish()
        return train_results, val_results

    def _train_epoch(self, model, data_loader, epoch):
        netG, netD = model
        g_losses, d_losses = [],[]
        train_logs = {
            'train_loss_g': [],
            'train_loss_d': [],
            'train_accuracy_cls': [],
            'train_accuracy_adv': [],
        }
        for m in self.train_metrics.values():
            m.reset()

        # initialize progress bar
        pbar = self._get_pbar()
        pbar.reset(total=len(data_loader))
        pbar.initial = 0
        pbar.set_description(f"Epoch {epoch}")

        for i, (x, target) in enumerate(data_loader):
            x,target = x.to(self.device),target.to(self.device)
            batch_size = x.size(0)
            if self.add_noise:
                x = x + torch.rand_like(x) * self.epsilon # for simplicity it's the same epsilon as fgsm
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            label_r = torch.full((x.size(0),), real_label, dtype=torch.float, device=self.device)
            # generate fake image batch with G
            fake, (J, errG_cls, preds_cls) = netG.generate(x,target)
            label_f = torch.full((batch_size,), fake_label, dtype=torch.float, device=self.device)
            labels =  torch.cat((label_r, label_f))

            if i % self.update_d_every_n_steps == 0:
                netD.zero_grad()
                # build batch
                batch = torch.cat((x, fake.detach()))
                # calculate Discriminator output
                output = netD(batch)
                errD = self.criterion_d(output, labels)
                errD.backward()
                D_x = output[0:batch_size].mean().item()
                D_G_z1 = output[batch_size:].mean().item()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()

            # adversarial training with fgsm
            if self.at1:
                x_perturbed, _ = fgsm(x, J, self.epsilon)
                # Re-classify the perturbed image
                output = netG(x_perturbed)
                # Calculate the loss
                errG_fgsm = F.cross_entropy(output, target)
                errG_fgsm.backward()
                preds_fgsm = output.argmax(dim=1)
                adv_acc = self.train_metrics['train_accuracy_adv'](preds_fgsm, target)
            
            # label fake batch
            label_f.fill_(real_label) # fake labels are real for generator cost
            # since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake)
            # calculate G's loss based on this output
            errG_adv = self.criterion_d(output, label_f)
            # calculate gradients for G
            errG = errG_cls + self.lambda_ * errG_adv
            errG.backward()
            D_G_z2 = output.mean().item() # mean of the discriminator output for fake images (after G update)
            # update G
            self.optimizerG.step()

            acc = self.train_metrics['train_accuracy_cls'](preds_cls, target)
            
            if i % self.log_freq == 0:
                pbar.set_postfix({"Loss_D":errD.item(), 
                                 "Loss_G":errG.item(),
                                 "Loss_Adv":errG_adv.item(), 
                                 "D(x)": D_x, 
                                 "D(G(z1))": D_G_z1, 
                                 "D(G(z2))": D_G_z2}) 
                self.writer.log({"loss/errD": errD.item(),"loss/errG": errG.item(),"loss/errG_adv":errG_adv.item(), "accuracy/train": acc.item(), "accuracy/train_adversarial": adv_acc.item() if self.at1 else 0.0})
            g_losses.append(errG.item())
            d_losses.append(errD.item())
            pbar.update()
        log_n_images = int(np.sqrt(x.size(0)))
        J_cpu = J.detach().cpu()
        N, C, H, W = J_cpu.shape
        normalized_J = F.layer_norm(J_cpu,[C,H,W])
        self.writer.log({"J_prime": wandb.Image(torchvision.utils.make_grid(fake[:log_n_images].detach().cpu())),
                         "J": wandb.Image(torchvision.utils.make_grid(normalized_J[:log_n_images])),
                         "original": wandb.Image(torchvision.utils.make_grid(x[:log_n_images].detach().cpu())),
                         })
        train_logs['train_loss_g'] = np.mean(g_losses)
        train_logs['train_loss_d'] = np.mean(d_losses)
        train_logs['train_accuracy_cls'] = self.train_metrics['train_accuracy_cls'].compute().item()
        train_logs['train_accuracy_adv'] = self.train_metrics['train_accuracy_adv'].compute().item()
        return train_logs

    def _valid_epoch(self, model, data_loader, epoch):
        netG, netD = model
        netG.eval()
        netD.eval()
        for m in self.valid_metrics.values():
            m.reset()
        valid_logs = {
            'valid_loss_g': [],
            'valid_accuracy_cls': [],
            'valid_accuracy_adv': [],
        }
        losses = []
        for i, (x, target) in enumerate(data_loader, 0):
            x = x.to(self.device)
            target = target.to(self.device)
            # forward pass real batch through G
            x.requires_grad = True
            output = netG(x)
            loss = F.cross_entropy(output, target)
            loss.backward()
            preds_cls = output.argmax(dim=1)
            # add the gradients from the all-real and all-fake batches
            J = x.grad.detach()
            x_perturbed, _ = fgsm(x, J, self.epsilon)
            # Re-classify the perturbed image
            with torch.no_grad():
                output = netG(x_perturbed)
            preds_fgsm = output.argmax(dim=1)
            acc = self.valid_metrics['val_accuracy_cls'](preds_cls, target)
            adv_acc = self.valid_metrics['val_accuracy_adv'](preds_fgsm, target)
            losses.append(loss.item())
        valid_logs['valid_loss_g'] = np.mean(losses)
        valid_logs['valid_accuracy_cls'] = self.valid_metrics['val_accuracy_cls'].compute().item()
        valid_logs['valid_accuracy_adv'] = self.valid_metrics['val_accuracy_adv'].compute().item()
        # print(f"Epoch: {epoch} | valid loss: {val_loss} | valid accuracy: {val_acc}")
        return valid_logs

    def _get_pbar(self):
        if not hasattr(self, 'pbar'):
            self.pbar = tqdm(leave=True)
        return self.pbar

    def _save_checkpoint(self, model, epoch, best=False):
        netG, netD = model
        arch = type(netG).__name__
        
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': netG.state_dict(),
            'config': dict(self.writer.config)
        }
        # print(f"Saving into {filename} ...")

        if USE_BETA_APIS:
            model_version = wandb.log_model(netG, self.dataset_name, ["best"] if best else None)
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
