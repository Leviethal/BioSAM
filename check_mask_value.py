import os
import cv2
import numpy as np

MASK_DIR = "flare_split/test/masks"

files = os.listdir(MASK_DIR)

unique_values = set()

for f in files[:50]:  # check first 50 masks
    mask = cv2.imread(os.path.join(MASK_DIR, f), cv2.IMREAD_GRAYSCALE)
    vals = np.unique(mask)
    unique_values.update(vals.tolist())

print("Unique values found:", sorted(unique_values))