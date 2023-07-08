from lightning.pytorch.cli import LightningCLI

from data_loader import *
from models import *


def main():
    cli = LightningCLI(save_config_callback=None)


if __name__ == '__main__':
    main()
