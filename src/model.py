import os

# Lightning imports
import lightning as pl
from torchmetrics import Accuracy

# Torch imports
import torch
import torch.nn as nn


class NumbersDetector(pl.LightningModule):
    def __init__(self, hidden_size=128, learning_rate=1e-3):
        super().__init__()

        # Set our init args as class attributes
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.loss = nn.CrossEntropyLoss()  # loss function

        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.input_size = 28 * 28  # Image dimensions

        # Define PyTorch model
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_classes),
        )

        # torchmetris accuracy
        self.val_accuracy = Accuracy(task="multiclass", num_classes=10)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss(preds, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        preds = self(x)
        loss = self.loss(preds, y)
        pred = torch.argmax(preds, dim=1)
        self.val_accuracy.update(pred, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss(preds, y)
        pred = torch.argmax(preds, dim=1)
        self.test_accuracy.update(pred, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
