from pathlib import Path
import cv2

# Reloads all pngs in the data folder to prevent 4 channel pngs errors

for path in Path('data').glob('**/*.png'):
    img = cv2.imread(str(path))
    cv2.imwrite(str(path), img)
    print(f"Reloaded {path}")