import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
from segment_anything import sam_model_registry, SamPredictor

# ================= CONFIG =================
TEST_IMAGE_DIR = "flare_split/test/images"
TEST_MASK_DIR = "flare_split/test/masks"
SAM_CHECKPOINT = "sam_vit_b_01ec64.pth"
MEDSAM_CHECKPOINT = "medsam_vit_b.pth"
DEVICE = "cuda"
ALPHAS = [0.75, 0.50, 0.25]
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


def build_merged_model(alpha):
    # Load SAM
    sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT)

    # Load MedSAM
    medsam = sam_model_registry["vit_b"](checkpoint=None)
    state_dict_medsam = torch.load(MEDSAM_CHECKPOINT, map_location="cpu")
    medsam.load_state_dict(state_dict_medsam)

    # Merge only image encoder weights
    for (name_s, param_s), (name_m, param_m) in zip(
        sam.image_encoder.named_parameters(),
        medsam.image_encoder.named_parameters()
    ):
        param_s.data = alpha * param_s.data + (1 - alpha) * param_m.data

    return sam


def evaluate_model(model):
    model.to(DEVICE)
    predictor = SamPredictor(model)

    image_files = sorted(os.listdir(TEST_IMAGE_DIR))
    total_dice = 0

    for f in tqdm(image_files, leave=False):

        img = cv2.imread(os.path.join(TEST_IMAGE_DIR, f), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(TEST_MASK_DIR, f), cv2.IMREAD_GRAYSCALE)

        mask = (mask > 127).astype(np.float32)
        img_rgb = np.stack([img, img, img], axis=-1)

        predictor.set_image(img_rgb)
        box = get_bounding_box(mask)

        masks, scores, logits = predictor.predict(
            box=box,
            multimask_output=False
        )

        pred_mask = masks[0].astype(np.float32)
        total_dice += dice_score(pred_mask, mask)

    return total_dice / len(image_files)


def main():

    print("\n=== Uniform Merging Experiments ===\n")

    results = {}

    for alpha in ALPHAS:
        print(f"\nEvaluating alpha = {alpha}")
        merged_model = build_merged_model(alpha)
        dice = evaluate_model(merged_model)
        results[alpha] = dice
        print(f"Alpha {alpha} → Dice: {dice:.6f}")

    print("\n=== Final Results ===")
    for alpha, dice in results.items():
        print(f"Alpha {alpha} → Dice: {dice:.6f}")


if __name__ == "__main__":
    main()