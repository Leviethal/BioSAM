import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

from MedSAM.MedSAM_Inference import MedSAMInfer

# ------------------ FILE PICKER ------------------
root = Tk()
root.withdraw()

image_path = filedialog.askopenfilename(
    title="Select medical image",
    filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")]
)

if not image_path:
    print("No image selected.")
    exit()

# ------------------ LOAD IMAGE ------------------
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

if len(image.shape) == 2:
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
else:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

H, W, _ = image.shape

# ------------------ SIMPLE DEFAULT BOX ------------------
# Center box (you can later make this interactive)
box = [
    W * 0.25,
    H * 0.25,
    W * 0.75,
    H * 0.75,
]

# ------------------ LOAD MEDSAM ------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
medsam = MedSAMInfer(
    checkpoint="medsam_vit_b.pth",
    device=device,
)

# ------------------ INFERENCE ------------------
mask = medsam.segment(image, box)

# ------------------ DISPLAY ------------------
plt.figure(figsize=(8, 8))
plt.imshow(image)
plt.imshow(mask, alpha=0.5, cmap="jet")
plt.axis("off")
plt.title("MedSAM Segmentation")
plt.show()
