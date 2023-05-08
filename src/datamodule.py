import os
from pathlib import Path

# Lightning imports
import lightning as pl

# Torch imports
import torch
from torch.utils.data import DataLoader, random_split

# Dataset imports
from torchvision import transforms, datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2


class BrandsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        batch_size: int = 256 if torch.cuda.is_available() else 64,
        valid_ratio=0.2,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.valid_ratio = valid_ratio
        self.train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
            ]
        )
        self.val_transform = A.Compose(
            [
                ToTensorV2(),
            ]
        )

    def setup(self, stage: str):
        if stage == "test":
            self.brands_dataset_test = datasets.ImageFolder(
                root=self.data_dir.joinpath("test").resolve(),
                transform=self.val_transform,
            )
        if stage == "fit":
            brands_dataset_full = datasets.ImageFolder(
                root=self.data_dir.joinpath("fit").resolve(),
                transform=self.train_transform,
            )
            nb_train = int((1.0 - self.valid_ratio) * len(brands_dataset_full))
            nb_valid = int(self.valid_ratio * len(brands_dataset_full))
            self.brands_dataset_train, self.mnisbrands_dataset_val = random_split(
                brands_dataset_full, [nb_train, nb_valid]
            )

    # Dataloader initialisation
    def train_dataloader(self):
        return DataLoader(
            self.brands_dataset_train, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnisbrands_dataset_val, batch_size=self.batch_size, shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.brands_dataset_test, batch_size=self.batch_size, shuffle=True
        )
