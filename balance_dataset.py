import os
import shutil
import random
import pandas as pd
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image

# Set paths
root_dir = "~/.cache/kagglehub/datasets/paultimothymooney/chest-xray-pneumonia/versions/2/chest_xray"
temp_balanced_dir = "./balanced_pneumonia_temp"
final_balanced_dir = "./balanced_pneumonia"

# Ensure temporary and final directories exist
for folder in [temp_balanced_dir, final_balanced_dir]:
    for category in ["NORMAL", "PNEUMONIA"]:
        os.makedirs(os.path.join(folder, category), exist_ok=True)

# Load original dataset
dataset = ImageFolder(os.path.join(root_dir, "train"))
normal_images = [img[0] for img in dataset.imgs if img[1] == dataset.class_to_idx["NORMAL"]]
pneumonia_images = [img[0] for img in dataset.imgs if img[1] == dataset.class_to_idx["PNEUMONIA"]]

# Balance the classes
augmentation_needed = len(pneumonia_images) - len(normal_images)
print(f"Augmenting {augmentation_needed} NORMAL images to balance the dataset...")

augmentations = {
    "rot": transforms.RandomRotation(10),
    "flip": transforms.RandomHorizontalFlip(p=1.0),
    "affine": transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    "jitter": transforms.ColorJitter(brightness=0.2, contrast=0.2),
    "blur": transforms.GaussianBlur(kernel_size=3),
}

augmentation_log = []
augmented_count = 0

# Augment NORMAL images
for img_path in normal_images:
    if augmented_count >= augmentation_needed:
        break

    img = Image.open(img_path)
    original_filename = os.path.basename(img_path)

    for key, transform in augmentations.items():
        if augmented_count >= augmentation_needed:
            break

        # Apply transformation and save augmented image
        aug_img = transform(img)
        aug_filename = f"aug_{key}_{augmented_count}.jpg"
        aug_img.save(os.path.join(temp_balanced_dir, "NORMAL", aug_filename))

        # Log augmentation details
        augmentation_log.append({
            "original_image": original_filename,
            "augmentation": key,
            "augmented_image": aug_filename
        })

        augmented_count += 1

# Copy original images to the temporary balanced directory
for img_path in normal_images:
    shutil.copy(img_path, os.path.join(temp_balanced_dir, "NORMAL", os.path.basename(img_path)))

for img_path in pneumonia_images:
    shutil.copy(img_path, os.path.join(temp_balanced_dir, "PNEUMONIA", os.path.basename(img_path)))

# Save augmentation log
log_df = pd.DataFrame(augmentation_log)
log_df.to_csv(os.path.join(temp_balanced_dir, "augmentation_log.csv"), index=False)
print("Class balancing complete!")

# Split balanced dataset into 70/20/10 train/val/test
def split_data(image_paths, train_ratio=0.7, val_ratio=0.2):
    random.shuffle(image_paths)
    train_size = int(len(image_paths) * train_ratio)
    val_size = int(len(image_paths) * val_ratio)
    train = image_paths[:train_size]
    val = image_paths[train_size:train_size + val_size]
    test = image_paths[train_size + val_size:]
    return train, val, test

# Get balanced image paths
balanced_dataset = ImageFolder(temp_balanced_dir)
normal_balanced = [img[0] for img in balanced_dataset.imgs if img[1] == balanced_dataset.class_to_idx["NORMAL"]]
pneumonia_balanced = [img[0] for img in balanced_dataset.imgs if img[1] == balanced_dataset.class_to_idx["PNEUMONIA"]]

# Split images
normal_train, normal_val, normal_test = split_data(normal_balanced)
pneumonia_train, pneumonia_val, pneumonia_test = split_data(pneumonia_balanced)

# Move images to final balanced directories
def move_images(image_list, target_dir, class_name):
    # Ensure the target directory exists
    os.makedirs(os.path.join(target_dir, class_name), exist_ok=True)
    
    # Copy images to the target directory
    for img_path in image_list:
        shutil.copy(img_path, os.path.join(target_dir, class_name, os.path.basename(img_path)))


for category, images in [("NORMAL", normal_train), ("PNEUMONIA", pneumonia_train)]:
    move_images(images, os.path.join(final_balanced_dir, "train"), category)

for category, images in [("NORMAL", normal_val), ("PNEUMONIA", pneumonia_val)]:
    move_images(images, os.path.join(final_balanced_dir, "val"), category)

for category, images in [("NORMAL", normal_test), ("PNEUMONIA", pneumonia_test)]:
    move_images(images, os.path.join(final_balanced_dir, "test"), category)

print("Dataset split into 70% train, 20% val, 10% test!")
print("Balanced and split dataset saved to:", final_balanced_dir)
