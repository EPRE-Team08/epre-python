import cv2
from pathlib import Path
import os

PATH = Path(r"data\raw").resolve()

for i, path in enumerate(PATH.glob("**/*")):
    if not path.suffix in [".png", ".webm", ".jpg", ".jpeg", ".webp"]:
        continue
    brand = path.parent.name
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.shape[0] > img.shape[1]:
        img = cv2.copyMakeBorder(
            img, 0, 0, 0, img.shape[0] - img.shape[1], cv2.BORDER_CONSTANT
        )
    if img.shape[0] < img.shape[1]:
        img = cv2.copyMakeBorder(
            img, 0, img.shape[1] - img.shape[0], 0, 0, cv2.BORDER_CONSTANT
        )

    img = cv2.resize(img, (320, 320))
    save_path = PATH.parent.joinpath(f"processed/{brand}/{str(i).zfill(4)}.png")
    if not save_path.parent.exists():
        os.makedirs(save_path.parent)
    cv2.imwrite(str(save_path), img)
