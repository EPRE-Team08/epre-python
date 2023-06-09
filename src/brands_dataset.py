import cv2
from torch.utils.data import Dataset
from pathlib import Path
import albumentations as A
import random

class BrandsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transforms = transform
        self.pathlist = list(root_dir.glob("**/*.png"))
        self.labels = {"fila": 0, "asics": 1, "puma": 2}
        self.numbers = {0: "fila", 1: "asics", 2: "puma"}

    def __len__(self):
        return len(self.pathlist)

    def __getitem__(self, index):
        img = cv2.imread(str(self.pathlist[index]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transforms:
            img = self.transforms(image=img)["image"]
        label = self.pathlist[index].parent.name
        return img, self.labels[label]


if __name__ == "__main__":
    #Test
    transforms = A.Compose(
        [
            # Pixel-level transforms
            A.RGBShift(always_apply=True, r_shift_limit=127, g_shift_limit=127, b_shift_limit=127),

            # Spatial-level transforms
            A.Rotate(180, always_apply=True),
            A.HorizontalFlip(),
            # A.RandomGridShuffle(grid=(4, 4), always_apply=True),
        ]
    )

    bd = BrandsDataset(Path("data/processed/test"), transform=transforms)
    for data in bd:
        print(data[1])
        cv2.imshow("img", data[0])
        cv2.waitKey(0)
        # cv2.imwrite(f"data/{random.randint(100_000,999_999)}.png", data[0])
