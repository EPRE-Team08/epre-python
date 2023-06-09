from model import BrandsDetector
from brands_dataset import BrandsDataset
from pathlib import Path
from albumentations.pytorch.transforms import ToTensorV2
import cv2
import numpy as np

import torch
import albumentations as A

transforms = A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

CKPT = r"logs\lightning_logs\version_0\checkpoints\epoch=199-step=200-v2.ckpt"


model = BrandsDetector.load_from_checkpoint(CKPT)
model.eval()
model.to("cpu")

brands_dm = BrandsDataset(Path(r"data/processed/test"), transforms)

for batch in brands_dm:
    images, labels = batch
    
    preds = model(images.unsqueeze(0).to("cpu"))
    preds = int(torch.argmax(preds, dim=1))
    if preds == labels:
        print(f"{brands_dm.numbers[preds]} is CORRECT")
    else:
        print(f"{brands_dm.numbers[preds]} should be {brands_dm.numbers[labels]}")
    print("----")
    