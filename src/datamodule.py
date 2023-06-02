from pathlib import Path

# Lightning imports
import lightning as pl

# Torch imports
import torch
from torch.utils.data import DataLoader

# Dataset imports
from torchvision import datasets

from brands_dataset import BrandsDataset


class BrandsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        batch_size: int = 64 if torch.cuda.is_available() else 32,
        transform=None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transform

    def setup(self, stage: str):
        if stage == "test":
            self.brands_dataset_test = BrandsDataset(
                root_dir=self.data_dir.joinpath("processed").resolve(),
                transform=None,
            )
        if stage == "fit":
            self.brands_dataset_full = BrandsDataset(
                root_dir=self.data_dir.joinpath("processed").resolve(),
                transform=self.transform,
            )

    # Dataloader initialisation
    def train_dataloader(self):
        return DataLoader(
            self.brands_dataset_full, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.brands_dataset_full, batch_size=self.batch_size, shuffle=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.brands_dataset_test, batch_size=self.batch_size, shuffle=True
        )
