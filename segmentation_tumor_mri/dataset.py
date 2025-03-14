import os
import nibabel as nib
import numpy as np

# Path Configurations
DATA_DIR = "/home/cap5516.student2/cap5516_spring25/segmentation_tumor_mri/Task01_BrainTumour"
IMG_DIR = os.path.join(DATA_DIR, "imagesTr")
MASK_DIR = os.path.join(DATA_DIR, "labelsTr")

# Get valid files (skip hidden/system files)
image_files = sorted([f for f in os.listdir(IMG_DIR) if not f.startswith("._")])[:5]  # Load first 5 valid samples

for file_name in image_files:
    img_path = os.path.join(IMG_DIR, file_name)
    mask_path = os.path.join(MASK_DIR, file_name)  # Mask should have same filename

    try:
        # Load the NIfTI files
        img = nib.load(img_path).get_fdata().astype(np.float32)  # (H, W, D, C)
        mask = nib.load(mask_path).get_fdata().astype(np.int64)  # (H, W, D)

        # Print shapes and value ranges
        print(f"🔹 File: {file_name}")
        print(f"   Image Shape: {img.shape} (H, W, D, C)")
        print(f"   Mask Shape: {mask.shape} (H, W, D)")
        print(f"   Image Min/Max: {img.min()} / {img.max()}")
        print(f"   Mask Unique Labels: {np.unique(mask)}")  # Check valid classes (0,1,2,3)

        # Check divisibility by 16
        h, w, d, _ = img.shape  # Assuming last dim is channels
        print(f"   Image divisible by 16? {h%16==0 and w%16==0 and d%16==0}")
        print("-" * 50)

    except Exception as e:
        print(f"❌ Error loading {file_name}: {e}")
