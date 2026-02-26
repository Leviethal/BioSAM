import os
import random
import shutil

IMAGE_DIR = "flare_slices/images"
MASK_DIR = "flare_slices/masks"

OUTPUT_DIR = "flare_split"

CALIB_RATIO = 0.2

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "calib/images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "calib/masks"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "test/images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "test/masks"), exist_ok=True)

files = sorted(os.listdir(IMAGE_DIR))
random.seed(42)
random.shuffle(files)

split_idx = int(len(files) * CALIB_RATIO)

calib_files = files[:split_idx]
test_files = files[split_idx:]

def move_files(file_list, split_type):
    for f in file_list:
        shutil.copy(os.path.join(IMAGE_DIR, f),
                    os.path.join(OUTPUT_DIR, split_type, "images", f))
        shutil.copy(os.path.join(MASK_DIR, f),
                    os.path.join(OUTPUT_DIR, split_type, "masks", f))

move_files(calib_files, "calib")
move_files(test_files, "test")

print("Split complete.")
print("Calibration:", len(calib_files))
print("Test:", len(test_files))