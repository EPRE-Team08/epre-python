import os

# Lightning imports
import lightning as pl
from torchmetrics import Accuracy
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# Torch imports
import torch
import torch.nn as nn


class BrandsDetector(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()

        # Set our init args as class attributes
        self.learning_rate = learning_rate
        self.loss = nn.CrossEntropyLoss()  # loss function

        # Hardcode some dataset specific attributes
        self.num_classes = 3
        self.input_size = 224 * 224  # Image dimensions

        # Define PyTorch model

        self.model = mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )

        # Freeze all layers except the last one
        for param in self.model.parameters():
            param.requires_grad = False

        # Change the last layer
        self.model.classifier[3] = nn.Linear(
            in_features=1024, out_features=3, bias=True
        )

        # torchmetris accuracy
        self.val_accuracy = Accuracy(task="multiclass", num_classes=3)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=3)

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
