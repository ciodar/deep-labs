from lightning.pytorch.cli import LightningCLI

from data_loader.race_datamodule import RACEDataModule
from models.qa_lightning import QATransformer


def main():
    cli = LightningCLI(QATransformer, RACEDataModule, save_config_callback=None)


if __name__ == '__main__':
    main()
