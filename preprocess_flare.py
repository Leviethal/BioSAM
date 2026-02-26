import os
import nibabel as nib
import numpy as np
import cv2
from tqdm import tqdm

# ========== CONFIG ==========
IMAGE_DIR = "FLARE22_LabeledCase50/images"
LABEL_DIR = "FLARE22_LabeledCase50/labels"

OUTPUT_IMAGE_DIR = "flare_slices/images"
OUTPUT_MASK_DIR = "flare_slices/masks"

TARGET_SIZE = 512
LIVER_LABEL = 1
# ============================


os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)


def normalize_ct(volume):
    # Simple normalization
    volume = volume.astype(np.float32)
    volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
    return volume


image_files = sorted(os.listdir(IMAGE_DIR))

slice_counter = 0

for file in tqdm(image_files):
    if not file.endswith(".nii.gz"):
        continue

    image_path = os.path.join(IMAGE_DIR, file)
    label_name = file.replace("_0000.nii.gz", ".nii.gz")
    label_path = os.path.join(LABEL_DIR, label_name)

    if not os.path.exists(label_path):
        continue

    # Load volumes
    image_vol = nib.load(image_path).get_fdata()
    label_vol = nib.load(label_path).get_fdata()

    image_vol = normalize_ct(image_vol)

    depth = image_vol.shape[2]

    for z in range(depth):
        img_slice = image_vol[:, :, z]
        mask_slice = label_vol[:, :, z]

        # Keep only slices that contain liver
        if np.sum(mask_slice == LIVER_LABEL) == 0:
            continue

        # Binary liver mask
        mask_slice = (mask_slice == LIVER_LABEL).astype(np.uint8)

        # Resize
        img_slice = cv2.resize(img_slice, (TARGET_SIZE, TARGET_SIZE))
        mask_slice = cv2.resize(mask_slice, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_NEAREST)

        # Convert to 0-255
        img_slice = (img_slice * 255).astype(np.uint8)
        mask_slice = (mask_slice * 255).astype(np.uint8)

        # Save
        cv2.imwrite(os.path.join(OUTPUT_IMAGE_DIR, f"{slice_counter}.png"), img_slice)
        cv2.imwrite(os.path.join(OUTPUT_MASK_DIR, f"{slice_counter}.png"), mask_slice)

        slice_counter += 1

print(f"Done. Total slices saved: {slice_counter}")