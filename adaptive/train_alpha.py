import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from adaptive_model import AdaptiveSAM

# ================= CONFIG =================
CALIB_IMAGE_DIR = "flare_split/calib/images"
CALIB_MASK_DIR = "flare_split/calib/masks"
SAM_CKPT = "sam_vit_b_01ec64.pth"
MEDSAM_CKPT = "medsam_vit_b.pth"
DEVICE = "cuda"
EPOCHS = 10
LR = 0.001
CHECKPOINT_DIR = "adaptive/checkpoints"
# ==========================================

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


class CalibDataset(Dataset):
    def __init__(self):
        self.files = sorted(os.listdir(CALIB_IMAGE_DIR))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]

        img = cv2.imread(os.path.join(CALIB_IMAGE_DIR, f), 0)
        mask = cv2.imread(os.path.join(CALIB_MASK_DIR, f), 0)

        # DO NOT resize image
        # DO NOT normalize
        # DO NOT apply pixel mean/std

        img = img.astype(np.float32)

        # Convert grayscale → 3-channel
        img = np.stack([img, img, img], axis=0)  # (3, H, W)

        mask = (mask > 127).astype(np.float32)

        return torch.tensor(img, dtype=torch.float32), \
            torch.tensor(mask, dtype=torch.float32).unsqueeze(0)


def dice_loss(pred, target, smooth=1e-5):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def get_box(mask):
    coords = torch.nonzero(mask[0])
    y_min = coords[:, 0].min()
    y_max = coords[:, 0].max()
    x_min = coords[:, 1].min()
    x_max = coords[:, 1].max()
    return torch.stack([x_min, y_min, x_max, y_max]).unsqueeze(0).float()


def save_checkpoint(model, optimizer, epoch, best_loss):
    torch.save({
        "epoch": epoch,
        "alpha": model.alpha.detach().cpu(),
        "optimizer": optimizer.state_dict(),
        "best_loss": best_loss
    }, os.path.join(CHECKPOINT_DIR, "latest.pt"))


def load_checkpoint(model, optimizer):
    ckpt_path = os.path.join(CHECKPOINT_DIR, "latest.pt")
    if not os.path.exists(ckpt_path):
        return 0, float("inf")

    checkpoint = torch.load(ckpt_path)
    model.alpha.data = checkpoint["alpha"].to(DEVICE)
    optimizer.load_state_dict(checkpoint["optimizer"])
    print(f"\nResuming from epoch {checkpoint['epoch']+1}")
    return checkpoint["epoch"] + 1, checkpoint["best_loss"]


def main():

    dataset = CalibDataset()
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)

    model = AdaptiveSAM(SAM_CKPT, MEDSAM_CKPT, DEVICE).to(DEVICE)
    optimizer = torch.optim.Adam([model.alpha], lr=LR)

    start_epoch, best_loss = load_checkpoint(model, optimizer)

    print("\n=== Training Adaptive Alpha (Resume Enabled) ===\n")

    for epoch in range(start_epoch, EPOCHS):

        epoch_loss = 0
        progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=True)

        for img, mask in progress_bar:
            img = img.to(DEVICE)
            mask = mask.to(DEVICE)

            boxes = get_box(mask)

            optimizer.zero_grad()

            pred = model(img, boxes)

            mask = torch.nn.functional.interpolate(
                mask,
                size=pred.shape[-2:], 
                mode="nearest"
            )

            loss = dice_loss(pred, mask)

            smooth = ((model.alpha[1:] - model.alpha[:-1]) ** 2).mean()
            loss += 0.01 * smooth

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            progress_bar.set_postfix({
                "batch_loss": f"{loss.item():.4f}",
                "avg_loss": f"{epoch_loss / (progress_bar.n + 1):.4f}"
            })

        epoch_loss /= len(loader)

        print(f"\nEpoch {epoch+1} Summary Loss: {epoch_loss:.4f}")

        print("Current Alpha (sigmoid):")
        print(torch.sigmoid(model.alpha).detach().cpu().numpy())

        # Save latest checkpoint
        save_checkpoint(model, optimizer, epoch, best_loss)

        # Save best model separately
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.alpha.detach().cpu(),
                       os.path.join(CHECKPOINT_DIR, "best_alpha.pt"))
            print("Best alpha updated.")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()