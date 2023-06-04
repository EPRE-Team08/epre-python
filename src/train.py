from model import BrandsDetector
from datamodule import BrandsDataModule
from pathlib import Path

import torch
import lightning as pl
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


DEBUG = True

train_transforms = A.Compose(
    [
        # Pixel-level transforms
        A.RGBShift(always_apply=True),
        A.RandomFog(p=0.2),
        A.RandomRain(p=0.2),
        A.RandomSnow(p=0.2),
        A.InvertImg(),
        # Spatial-level transforms
        A.Rotate(180, always_apply=True),
        A.HorizontalFlip(),
        A.Perspective(p=0.8),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

val_transfroms = A.Compose(
    [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

brands_dm = BrandsDataModule(Path(r"data"), t_transform=train_transforms, v_transform = val_transfroms)
model = BrandsDetector()
trainer = pl.Trainer(
    accelerator="auto",
    devices=1,
    max_epochs=200,
    logger=CSVLogger(save_dir="logs/"),
)
trainer.fit(model, brands_dm)

# save model as pt file
torch.save(model.state_dict(), "model.pt")
