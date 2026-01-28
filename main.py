import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
from segment_anything import sam_model_registry, SamPredictor

# ------------------ FILE PICKER ------------------
root = Tk()
root.withdraw()  # hide main tkinter window

image_path = filedialog.askopenfilename(
    title="Select an image",
    filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
)

if not image_path:
    print("No image selected. Exiting.")
    exit()

# ------------------ CONFIG ------------------
checkpoint_path = "sam_vit_l_0b3195.pth"
model_type = "vit_l"
device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------ LOAD MODEL ------------------
sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
sam.to(device)
predictor = SamPredictor(sam)

# ------------------ LOAD IMAGE ------------------
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

predictor.set_image(image)

# ------------------ PROMPT ------------------
# Single foreground click (center of image)
h, w, _ = image.shape
input_point = np.array([[w // 2, h // 2]])
input_label = np.array([1])

# ------------------ PREDICT ------------------
with torch.no_grad():
    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True
    )

best_mask = masks[scores.argmax()]

# ------------------ DISPLAY ------------------
plt.figure(figsize=(8, 8))
plt.imshow(image)
plt.imshow(best_mask, alpha=0.5)
plt.axis("off")
plt.title("SAM Segmentation Result")
plt.show()
