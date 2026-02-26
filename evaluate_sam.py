import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
from segment_anything import sam_model_registry, SamPredictor

# ================= CONFIG =================
TEST_IMAGE_DIR = "flare_split/test/images"
TEST_MASK_DIR = "flare_split/test/masks"
SAM_CHECKPOINT = "sam_vit_l_0b3195.pth"
DEVICE = "cuda"
# ==========================================


def dice_score(pred, target, smooth=1e-5):
    pred = (pred > 0.5).astype(np.float32)
    intersection = (pred * target).sum()
    return (2 * intersection + smooth) / (
        pred.sum() + target.sum() + smooth
    )


def get_bounding_box(mask):
    coords = np.where(mask > 0)
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    return np.array([x_min, y_min, x_max, y_max])


def main():

    sam = sam_model_registry["vit_l"](checkpoint=SAM_CHECKPOINT)
    sam.to(DEVICE)
    predictor = SamPredictor(sam)

    image_files = sorted(os.listdir(TEST_IMAGE_DIR))

    total_dice = 0

    for f in tqdm(image_files):

        img = cv2.imread(os.path.join(TEST_IMAGE_DIR, f), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(TEST_MASK_DIR, f), cv2.IMREAD_GRAYSCALE)

        mask = (mask > 127).astype(np.float32)

        # Convert to 3-channel
        img_rgb = np.stack([img, img, img], axis=-1)

        predictor.set_image(img_rgb)

        box = get_bounding_box(mask)

        masks, scores, logits = predictor.predict(
            box=box,
            multimask_output=False
        )

        pred_mask = masks[0].astype(np.float32)

        total_dice += dice_score(pred_mask, mask)

    print("Pure SAM (GT Box) Dice:", total_dice / len(image_files))


if __name__ == "__main__":
    main()