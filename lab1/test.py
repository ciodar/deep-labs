# Start with some standard imports.
from pprint import pprint

import argparse

import numpy as np
import torch
import torchvision.datasets as datasets
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Subset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from wandb.beta.workflows import use_model

import wandb
from data_loader import SubsetDataset
from models.resnet import ResNetForClassification, FullyConvResNet
from trainer import Trainer


def main(args):
    # Configuration
    DEVICE = args.device
    BATCH_SIZE = args.batch_size
    USE_BETA_APIS = False

    # normalize data based on train split
    normalize = transforms.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))
    # Data augmentation and normalization for training
    # Just normalization for validation
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    # Load CIFAR test
    ds_test = datasets.CIFAR10(root=args.data, train=False, download=True, transform=transform)
    test_data_loader = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=True)

    # log in to WandB
    wandb.login()

    # Architecture hyperparameters.
    arch_hparams = {
        "layers": [args.num_layers] * 3,
        "residual": args.residual,
        "batchnorm": args.batchnorm,
        "num_classes": 10,
        "num_channels": args.num_channels,
        "residual_type": args.residual_type
    }

    # "[[entity/]project/]collectionName:latest"
    ##### W&B MODEL MANAGEMENT SPECIFIC CALLS ######


    if args.checkpoint_url:
        run = wandb.init()
        if USE_BETA_APIS:
            model_art = use_model(f"{args.checkpoint_url}:latest")
            ckpt = model_art.model_obj()
        else:
            model_art = wandb.use_artifact(f"{args.checkpoint_url}")
            model_path = model_art.get_path("ckpt-best.pth").download()
            ckpt = torch.load(model_path, map_location=DEVICE)
        # this ignores console params
        config_dict = ckpt['config']
        arch_hparams.update((k, config_dict[k]) for k in config_dict.keys() & arch_hparams.keys())
        model = ResNetForClassification(**arch_hparams).to(DEVICE)
        state_dict = ckpt['state_dict']
        model.load_state_dict(state_dict)
    else:
        model = ResNetForClassification(**arch_hparams).to(DEVICE)

    tot_params = sum(p.numel() for p in model.parameters())
    print(model)
    print("Total number of parameters: ", tot_params)

    # Begin evaluation
    trainer = Trainer()
    acc, results = trainer.test(model, test_data_loader)
    print(f'Accuracy report on TEST:\n Accuracy: {acc}')
    print(results)


parser = argparse.ArgumentParser(description='Lab1 - Resnets evaluation')
# Configuration
parser.add_argument('--checkpoint_url', required=True, help='url to a model collection on wandb')
parser.add_argument('--data', default='data/', help='path to CIFAR-10 dataset root (default: ./data/)')
parser.add_argument('--batch_size', type=int,
                    default=128, help='input batch size')
parser.add_argument('--device', type=str, default='cuda',
                    help='ID of GPUs to use, eg. cuda:0,cuda:1')
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
