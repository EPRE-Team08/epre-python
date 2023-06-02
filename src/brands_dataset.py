import cv2
from torch.utils.data import Dataset
from pathlib import Path
import albumentations as A


class BrandsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transforms = transform
        self.pathlist = list(root_dir.glob("**/*.png"))
        self.labels = {"fila": 0, "asics": 1, "puma": 2}

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
    transforms = A.Compose(
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
        ]
    )

    bd = BrandsDataset(Path("data/processed"), transform=transforms)
    print(len(bd))
    for data in bd:
        print(data[1])
        cv2.imshow("img", data[0])
        cv2.waitKey(0)
