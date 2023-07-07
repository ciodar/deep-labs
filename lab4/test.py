import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import CIFAR10

import numpy as np
from torch.utils.data import Subset, DataLoader
import wandb
import wandb.beta.workflows as wf

from models.gan import Generator, Discriminator
from models.cnn import CNN
from trainer import Trainer

from fgsm import get_adversarial_examples

USE_BETA_APIS = False

def test_adversarial( model, device, test_loader, epsilon ):
    # Accuracy counter
    correct = 0
    adv_examples = None

    # Loop over all examples in test set
    for i,(Xs, ys) in enumerate(test_loader):
        Xs, ys = Xs.to(device), ys.to(device)
        Xs_adversarial_normalized =  get_adversarial_examples(model, Xs, ys, epsilon)
        final_out = model(Xs_adversarial_normalized)
        final_pred = final_out.argmax(1) # get the index of the max log-probability
        correct += (final_pred.squeeze() == ys).sum().item()
        # add first batch of adversarial examples
        if i==0:
            adv_ex = Xs_adversarial_normalized.detach().cpu().numpy()
            adv_examples = (adv_ex, ys, final_pred)
    # Calculate final accuracy for this epsilon
    acc = correct/len(test_loader.dataset)
    print(f"Epsilon: {epsilon:.2f}\tTest Accuracy = {correct} / {len(test_loader.dataset)} = {acc}")

    # Return the accuracy and an adversarial example
    return acc, adv_examples

def main(args):
    # hyperparameters
    BATCH_SIZE = args.batch_size
    DEVICE = args.device

    # We will use CIFAR-10 as our in-distribution dataset.
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    ds_test = CIFAR10(root=args.data, train=False, download=True, transform=transform)
    dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # log in to WandB
    wandb.login()
    wandb_config = {
        "project": "lab-4-validation",
        "job_type": "model_trainer",
        "dataset_name": "cifar10",
        "model_collection_name": "CIFAR JARN CNN"
    }

    # "[[entity/]project/]collectionName:latest"
    ##### W&B MODEL MANAGEMENT SPECIFIC CALLS ######

    arch_hparams = {
        "num_classes": 10,
        "num_conv_layers": args.g_conv_layers,
        "num_fc_layers": args.g_fc_layers,
        "activation": args.g_activation
    }

    # define backbones
    g_backbone = CNN(num_classes=10,num_conv_layers=args.g_conv_layers,num_fc_layers=args.g_fc_layers,activation=args.g_activation)

    if args.checkpoint_url:
        run = wandb.init()
        if USE_BETA_APIS:
            model_art = wf.use_model(f"{args.checkpoint_url}:latest")
            ckpt = model_art.model_obj()
        else:
            model_art = wandb.use_artifact(f"{args.checkpoint_url}")
            model_path = model_art.get_path("ckpt-best.pth").download()
            ckpt = torch.load(model_path, map_location=DEVICE)
        # this ignores console params
        config_dict = ckpt['config']
        arch_hparams.update((k, config_dict[k]) for k in config_dict.keys() & arch_hparams.keys())
        backbone = CNN(**arch_hparams).to(DEVICE)
        model = Generator(backbone, criterion=nn.CrossEntropyLoss,adapter_type=args.adapter_type).to(DEVICE)
        state_dict = ckpt['state_dict']
        model.load_state_dict(state_dict)
    else:
        backbone = CNN(**arch_hparams).to(DEVICE)
        model = Generator(backbone, adapter_type=args.adapter_type).to(DEVICE)
        run = wandb.init(config=arch_hparams, **wandb_config)

    tot_params = sum(p.numel() for p in model.parameters())
    print(model)

    epsilons = np.arange(0, 0.3, 0.05)

    # Run test for each epsilon
    for eps in epsilons:
        acc, ex = test_adversarial(model, DEVICE, dl_test, eps)
        run.log({"epsilon": eps, "accuracy": acc, "examples": ex})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Lab1 - Resnets training')
    # Configuration
    parser.add_argument('--data', default='./data/', help='path to CIFAR-10 dataset root (default: ./data/)')
    parser.add_argument('--checkpoint_url', required=True, help='url to a model collection on wandb')
    parser.add_argument('--batch_size', type=int,
                        default=128, help='input batch size')
    parser.add_argument('--checkpoints', action=argparse.BooleanOptionalAction,
                        help='Enable checkpoints saving')
    parser.add_argument('--device', type=str, default='cuda',
                        help='ID of GPUs to use, eg. cuda:0,cuda:1')
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
    parser.add_argument('--epsilon', default=0.1, type=float,
                        help='Epsilon for adversarial training')

    args = parser.parse_args()

    main(args)
