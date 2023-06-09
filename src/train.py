from model import BrandsDetector
from datamodule import BrandsDataModule
from pathlib import Path

import torch
import lightning as pl
from lightning.pytorch.loggers import CSVLogger

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


DEBUG = True

train_transforms = A.Compose(
    [
        # Pixel-level transforms
        A.RGBShift(always_apply=True, r_shift_limit=127, g_shift_limit=127, b_shift_limit=127),
        A.RandomFog(p=0.2),
        A.RandomRain(p=0.2),
        A.RandomSnow(p=0.2),
        A.RandomSunFlare(p=0.2),
        # Spatial-level transforms
        A.Rotate(180, always_apply=True),
        A.HorizontalFlip(),
        # A.RandomGridShuffle(grid=(2, 2), always_apply=True),        
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
        
    ]
)

val_transfroms = A.Compose(
    [
        A.RGBShift(always_apply=True, r_shift_limit=127, g_shift_limit=127, b_shift_limit=127),
        A.Rotate(180, always_apply=True),        
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

trainer.test(model, datamodule=brands_dm)



# save model as pt file
torch.save(model.state_dict(), "model.pt")
