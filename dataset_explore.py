# %%
import os

# %%
# DATASET_PATH = "/home/cap5516.student2/.cache/kagglehub/datasets/paultimothymooney/chest-xray-pneumonia/versions/2/chest_xray"
DATASET_PATH = BALANCED_DATASET_PATH = "/home/cap5516.student2/cap5516_spring25/balanced_pneumonia"

# How many NORMAL/PNEUMONIA in train/val/test?
# %%
def count_items(DATASET_PATH):
    categories = ["NORMAL", "PNEUMONIA"]
    splits = ["train", "val", "test"]
    for split in splits:
        split_path = os.path.join(DATASET_PATH, split)
        
        for cat in categories:
            cat_path = os.path.join(split_path, cat)
            count = len(os.listdir(cat_path))
            print(f'Number of images in {split}/{cat}: {count}')
            
# %%
count_items(DATASET_PATH)
# %%
