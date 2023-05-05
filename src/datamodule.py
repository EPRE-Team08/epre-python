import os

# Lightning imports
import lightning as pl

# Torch imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Dataset imports
from torchvision import transforms
from torchvision.datasets import MNIST
import torchvision


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = os.environ.get("PATH_DATASETS", "."),
        batch_size: int = 256 if torch.cuda.is_available() else 64,
        valid_ratio=0.2,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.valid_ratio = valid_ratio
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
            ]
        )

    def setup(self, stage: str):
        if stage == "test":
            self.mnist_test = MNIST(
                self.data_dir, train=False, transform=self.transform
            )
        if stage == "fit":
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            nb_train = int((1.0 - self.valid_ratio) * len(mnist_full))
            nb_valid = int(self.valid_ratio * len(mnist_full))
            self.mnist_train, self.mnist_val = random_split(
                mnist_full, [nb_train, nb_valid]
            )

    def prepare_data(self):
        # download data and cache it
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    # Dataloader initialisation
    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, shuffle=True)
