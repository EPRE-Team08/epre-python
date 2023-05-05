from model import NumbersDetector
from datamodule import MNISTDataModule

import lightning as pl
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


mnist = MNISTDataModule()
model = NumbersDetector()
trainer = pl.Trainer(
    accelerator="auto",
    devices=1,
    max_epochs=25,
    logger=CSVLogger(save_dir="logs/"),
    callbacks=[EarlyStopping(monitor="train_loss", mode="min")],
)
trainer.fit(model, mnist)
