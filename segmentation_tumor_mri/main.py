from medpy.metric import binary
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import nibabel as nib
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
from monai.networks.nets import UNet
from monai.metrics import DiceMetric
import torch.nn.functional as F
from tqdm import tqdm


# Configurations
DATA_DIR = "/home/cap5516.student2/cap5516_spring25/segmentation_tumor_mri/Task01_BrainTumour"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 20
EPOCHS = 50
KFOLDS = 5
NUM_CLASSES = 4
CHECKPOINT_DIR = "./checkpoints"
RESULTS_DIR = "./results"
PATIENCE = 10  # Early stopping patience
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Dataset Loader
class BrainTumorDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.file_names = sorted([f for f in os.listdir(img_dir) if not f.startswith("._")])

    def __len__(self):
        return len(self.file_names)

    def pad_to_multiple(self, tensor, target_shape=(256, 256, 160)):
        """Pads a 3D tensor (H, W, D) to match `target_shape`."""
        h, w, d = tensor.shape[-3:]
        pad_h = target_shape[0] - h
        pad_w = target_shape[1] - w
        pad_d = target_shape[2] - d
        return F.pad(tensor, (0, pad_d, 0, pad_w, 0, pad_h), mode="constant", value=0)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.file_names[idx])
        mask_path = os.path.join(self.mask_dir, self.file_names[idx])

        img = nib.load(img_path).get_fdata().astype(np.float32)  # (H, W, D, C)
        mask = nib.load(mask_path).get_fdata().astype(np.int64)  # (H, W, D)

        # Normalize image
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        img_tensor = torch.tensor(img).permute(3, 0, 1, 2)  # (C, H, W, D)
        mask_tensor = torch.tensor(mask)  # (H, W, D)

        # Apply padding
        img_tensor = self.pad_to_multiple(img_tensor)
        mask_tensor = self.pad_to_multiple(mask_tensor)

        return img_tensor, mask_tensor

dataset = BrainTumorDataset(
    os.path.join(DATA_DIR, "imagesTr"),
    os.path.join(DATA_DIR, "labelsTr")
)
kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=42)

# UNet Model
class SimpleUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=NUM_CLASSES):
        super(SimpleUNet, self).__init__()
        self.unet = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2
        )

    def forward(self, x):
        return self.unet(x)

# Loss Function (Dice + CrossEntropy)
class DiceLoss(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = F.softmax(preds, dim=1)  # Convert logits to probabilities
        targets_one_hot = F.one_hot(targets.long(), num_classes=self.num_classes).permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)

        intersection = (preds * targets_one_hot).sum(dim=(2, 3, 4))
        union = preds.sum(dim=(2, 3, 4)) + targets_one_hot.sum(dim=(2, 3, 4))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class CombinedLoss(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(num_classes)

    def forward(self, preds, targets):
        ce = self.ce_loss(preds, targets)
        dice = self.dice_loss(preds, targets)
        return self.alpha * ce + (1 - self.alpha) * dice

# Training & Evaluation
fold_results = []  # Store final Dice & HD for all folds

for fold, (train_idx, val_idx) in tqdm(enumerate(kf.split(dataset), start=1), desc="Processing folds..."):
    print(f"\nTraining Fold {fold}/{KFOLDS}")

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=1, shuffle=False)

    model = SimpleUNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = CombinedLoss()
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    best_dice = 0
    no_improve_epochs = 0
    history = {"train_loss": [], "val_dice": [], "val_hausdorff": []}

    for epoch in tqdm(range(EPOCHS), desc="Processing epochs..."):
        model.train()
        epoch_loss = 0

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}") as tbar:
            for img, mask in tbar:
                img, mask = img.to(DEVICE), mask.to(DEVICE)
                optimizer.zero_grad()
                output = model(img)
                loss = criterion(output, mask)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                tbar.set_postfix(loss=loss.item())

        history["train_loss"].append(epoch_loss / len(train_loader))

    # Validation Loop
    model.eval()
    dice_scores, hausdorff_distances = [], []

    with torch.no_grad():
        for img, mask in tqdm(val_loader, desc="Validating"):
            img, mask = img.to(DEVICE), mask.to(DEVICE)
            output = torch.argmax(model(img), dim=1)  # Convert logits to class labels

            # Convert to one-hot encoding for DiceMetric
            output_one_hot = F.one_hot(output, num_classes=NUM_CLASSES).permute(0, 4, 1, 2, 3)
            mask_one_hot = F.one_hot(mask, num_classes=NUM_CLASSES).permute(0, 4, 1, 2, 3)

            dice_metric.reset()
            dice_metric(y_pred=output_one_hot.float(), y=mask_one_hot.float())
            dice = dice_metric.aggregate().mean().item()
            dice_scores.append(dice)

            # Compute Hausdorff Distance
            mask_np = mask.cpu().numpy().astype(bool)
            output_np = output.cpu().numpy().astype(bool)

            if np.any(mask_np) and np.any(output_np):  # Ensure non-empty masks
                hausdorff_distance = binary.hd(output_np, mask_np, voxelspacing=None, connectivity=1)
            else:
                hausdorff_distance = np.nan  # No foreground, skip computation

            hausdorff_distances.append(hausdorff_distance)

    avg_dice = np.mean(dice_scores)
    avg_hausdorff = np.nanmean(hausdorff_distances)

    history["val_dice"].append(avg_dice)
    history["val_hausdorff"].append(avg_hausdorff)

    print(f"\nFold {fold} Results - Dice: {avg_dice:.4f}, Hausdorff: {avg_hausdorff:.4f}")

    if avg_dice > best_dice:
        best_dice = avg_dice
        no_improve_epochs = 0
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"unet_fold{fold}_best.pth")
        torch.save(model.state_dict(), checkpoint_path)
    else:
        no_improve_epochs += 1

    if no_improve_epochs >= PATIENCE:
        print(f"⏳ Early stopping triggered at epoch {epoch+1}")
        break

    # Save history for visualization
    np.save(os.path.join(RESULTS_DIR, f"history_fold{fold}.npy"), history)
    fold_results.append({"fold": fold, "dice": avg_dice, "hausdorff": avg_hausdorff})

# Save final results
np.save(os.path.join(RESULTS_DIR, "final_results.npy"), fold_results)
for fold in fold_results:
    print(f"Fold {fold['fold']} - Dice: {fold['dice']:.4f}, Hausdorff: {fold['hausdorff']:.4f}")
print("\nTraining Complete! All results saved.")
