from lightning.pytorch.cli import LightningCLI

from data_loader import *
from models.qa_lightning import QATransformer


def main():
    cli = LightningCLI(QATransformer, save_config_callback=None)


if __name__ == '__main__':
    main()
