from pathlib import Path

# Lightning imports
import lightning as pl

# Torch imports
import torch
from torch.utils.data import DataLoader

# Dataset imports
from brands_dataset import BrandsDataset


class BrandsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        batch_size: int = 64 if torch.cuda.is_available() else 32,
        t_transform=None,
        v_transform=None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.t_transform = t_transform
        self.v_transform = v_transform

    def setup(self, stage: str):
        if stage == "test":
            self.brands_dataset_test = BrandsDataset(
                root_dir=self.data_dir.joinpath("processed/test").resolve(),
                transform=self.v_transform,
            )
        if stage == "fit":
            self.brands_dataset_train = BrandsDataset(
                root_dir=self.data_dir.joinpath("processed/train").resolve(),
                transform=self.t_transform,
            )

            self.brands_dataset_val = BrandsDataset(
                root_dir=self.data_dir.joinpath("processed/val").resolve(),
                transform=self.v_transform,
            )

    # Dataloader initialisation
    def train_dataloader(self):
        return DataLoader(
            self.brands_dataset_train, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.brands_dataset_val, batch_size=self.batch_size, shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.brands_dataset_test, batch_size=self.batch_size, shuffle=False
        )
