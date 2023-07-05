import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import CIFAR10

import numpy as np
from torch.utils.data import Subset, DataLoader
import wandb

from models.gan import Generator, Discriminator
from models.cnn import CNN
from trainer import Trainer

def main(args):
    # hyperparameters
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    DEVICE = args.device

    # We will use CIFAR-10 as our in-distribution dataset.
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Load the datasets and setup the DataLoaders.
    cifar_train = CIFAR10(root=args.data, train=True, download=True, transform=transform)
    # Split train into train and validation.
    val_size = int(0.1 * len(cifar_train))
    I = np.random.permutation(len(cifar_train))
    ds_train = Subset(cifar_train, I[val_size:])
    ds_val = Subset(cifar_train, I[:val_size])

    ds_test = CIFAR10(root=args.data, train=False, download=True, transform=transform)

    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # log in to WandB
    wandb.login()
    wandb_config = {
        "project": "lab-4-adversarial",
        "job_type": "model_trainer",
        "dataset_name": "cifar10",
        "model_collection_name": "CIFAR JARN CNN"
    }

    writer = wandb.init(project=wandb_config['project'],config=vars(args))

    # define backbones
    g_backbone = CNN(num_classes=10,num_conv_layers=args.g_conv_layers,num_fc_layers=args.g_fc_layers,activation=args.g_activation)
    d_backbone = CNN(num_classes=1,num_conv_layers=args.d_conv_layers,num_fc_layers=args.d_fc_layers,activation=args.d_activation)

    # define GAN
    netG = Generator(
                    backbone=g_backbone,
                    criterion=nn.CrossEntropyLoss(),
                    adapter_type=args.adapter_type
                    ).to(DEVICE)
    netD = Discriminator(d_backbone).to(DEVICE)
    model = netG, netD

    optimG = torch.optim.Adam(netG.parameters(), lr=args.g_lr)
    optimD = torch.optim.Adam(netD.parameters(), lr=args.d_lr)


    trainer = Trainer(
        optimizers={'generator': optimG, 'discriminator': optimD},
        writer=writer,
        epochs=EPOCHS,
        update_d_every_n_steps=args.update_d_every,
        device=DEVICE,
        at1=args.at1,
        add_noise=args.add_noise,
        epsilon=args.epsilon,
        lambda_adv=args.lambda_adv
    )

    # Train the model.
    train_log, val_log = trainer.train(model, dl_train, dl_val)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Lab1 - Resnets training')
    # Configuration
    parser.add_argument('--data', default='./data/', help='path to CIFAR-10 dataset root (default: ./data/)')
    parser.add_argument('--batch_size', type=int,
                        default=128, help='input batch size')
    parser.add_argument('--checkpoints', action=argparse.BooleanOptionalAction,
                        help='Enable checkpoints saving')
    parser.add_argument('--device', type=str, default='cuda',
                        help='ID of GPUs to use, eg. cuda:0,cuda:1')
    # Optimizer hyperparameters
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--optim', type=str, default='SGD',
                        help='optim for training, Adam / SGD (default)')
    parser.add_argument('--g_lr', default=1e-3, type=float,
                        help='learning rate for generator')
    parser.add_argument('--d_lr', default=1e-3, type=float,
                        help='learning rate for discriminator')
    parser.add_argument('--update_d_every', default=1, type=int,
                    help='updates discriminator every n generator iterations')
    # Architecture hyperparameters
    parser.add_argument('--g_conv_layers', default=4, type=int,
                        help='Number of convolutional layers for generator')
    parser.add_argument('--g_fc_layers', default=1, type=int,
                        help='Number of fully connected layers for generator')
    parser.add_argument('--g_activation', default="ReLU", type=str,
                        help='Activation for generator')
    parser.add_argument('--d_conv_layers', default=5, type=int,
                        help='Number of convolutional layers for generator')
    parser.add_argument('--d_fc_layers', default=0, type=int,
                        help='Number of convolutional layers for generator')
    parser.add_argument('--d_activation', default="ReLU", type=str,
                        help='ctivation for discriminator')
    parser.add_argument('--adapter_type', default="conv", type=str,
                        help='Adapter type for generator (conv, conv-noactivation, normalize)')
    # Training hyperparameters
    parser.add_argument('--at1', action=argparse.BooleanOptionalAction,
                        help='Enable adversarial training with FGSM')
    parser.add_argument('--add_noise', action=argparse.BooleanOptionalAction,
                        help='Enable adding noise to the input')
    parser.add_argument('--epsilon', default=0.1, type=float,
                        help='Epsilon for adversarial training')
    parser.add_argument('--lambda_adv', default=1.0, type=float,
                        help='Lambda for adversarial training')

    args = parser.parse_args()

    main(args)
