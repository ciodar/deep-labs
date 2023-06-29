# Start with some standard imports.
import argparse
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.data import Subset

import wandb
from data_loader import SubsetDataset
from models.resnet import ResNetForClassification
from trainer import Trainer


def main(args):
    # Configuration
    DEVICE = args.device
    BATCH_SIZE = args.batch_size

    # Load CIFAR train
    dataset = datasets.CIFAR10(root=args.data, train=True, download=True)
    # normalize data based on train split
    normalize = transforms.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))
    # Data augmentation and normalization for training
    # Just normalization for validation
    transform = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            normalize
        ]),
    }

    # Split train into train and validation.
    val_size = int(0.1 * len(dataset))
    I = np.random.permutation(len(dataset))
    ds_train = SubsetDataset(Subset(dataset, I[val_size:]), transform['train'])
    ds_val = SubsetDataset(Subset(dataset, I[:val_size]), transform['val'])

    train_data_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
    valid_data_loader = DataLoader(ds_val, batch_size=BATCH_SIZE)
    # log in to WandB
    wandb.login()
    wandb_config = {
        "project": "lab-1",
        "job_type": "model_trainer",
        "dataset_name": "cifar10",
        "model_collection_name": "CIFAR Fully-conv ResNet"
    }

    # Training hyperparameters.
    trainer_hparams = {
        "optimizer": args.optim,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "momentum": args.momentum
    }

    # Architecture hyperparameters.
    arch_hparams = {
        "layers": [args.num_layers] * 3,
        "residual": args.residual,
        "batchnorm": args.batchnorm,
        "num_classes": 10,
        "num_channels": args.num_channels,
        "residual_type": args.residual_type
    }

    config = {
        **trainer_hparams,
        **arch_hparams
    }

    writer = wandb.init(project=wandb_config['project'], config=config)
    config = wandb.config
    model = ResNetForClassification(**arch_hparams).to(DEVICE)

    tot_params = sum(p.numel() for p in model.parameters())

    print(model)
    print('Total number of parameters: {:,}'.format(tot_params))

    # Get optimizer from torch.optim
    opt = getattr(torch.optim, config.optimizer)(params=model.parameters(), lr=config.lr,
                                                 weight_decay=config.weight_decay, momentum=config.momentum)

    scheduler = MultiStepLR(opt, milestones=[50, 75], gamma=0.1)

    # Begin training
    trainer = Trainer(opt, writer, epochs=config.epochs, device=DEVICE, lr_scheduler=scheduler,
                      checkpoints=args.checkpoints)
    trainer.train(model, train_data_loader, valid_data_loader)


parser = argparse.ArgumentParser(description='Lab1 - Resnets training')
# Configuration
parser.add_argument('--data', default='data/', help='path to CIFAR-10 dataset root (default: ./data/)')
parser.add_argument('--batch_size', type=int,
                    default=128, help='input batch size')
parser.add_argument('--checkpoints', action=argparse.BooleanOptionalAction,
                    help='Enable checkpoints saving')
parser.add_argument('--device', type=str, default='cuda',
                    help='ID of GPUs to use, eg. cuda:0,cuda:1')
# Optimizer hyperparameters
parser.add_argument('--epochs', default=85, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--optim', type=str, default='SGD',
                    help='optim for training, Adam / SGD (default)')
parser.add_argument('--lr', default=0.1, type=float,
                    help='learning rate for training')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum for SGD')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help='weight_decay for SGD')
# Architecture hyperparameters
parser.add_argument('--num_layers', default=1, type=int,
                    help='Number of layers for each convolutional block')
parser.add_argument('--num_channels', default=16, type=int,
                    help='Number of channels of first conv layer')
parser.add_argument('--residual', action=argparse.BooleanOptionalAction,
                    help='Enable residual connections')
parser.add_argument('--batchnorm', action=argparse.BooleanOptionalAction,
                    help='Enable batch normalization')
parser.add_argument('--residual_type', default='a', type=str,
                    help='Residual connection type (Available values: a,b,c)')
args = parser.parse_args()

main(args)
