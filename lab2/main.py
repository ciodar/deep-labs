from lightning.pytorch.cli import LightningCLI

from data_loader import *
from models.qa_mlp import QAMLP


def main():
    cli = LightningCLI(QAMLP, save_config_callback=None)


if __name__ == '__main__':
    main()
