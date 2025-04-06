import os
import math
import glob
import tifffile
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from EfficientSAM.efficient_sam.efficient_sam import build_efficient_sam
from monai.metrics import DiceMetric
from monai.metrics import MeanIoU
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import json
from datetime import datetime
import logging
import sys
from torch.optim.lr_scheduler import CosineAnnealingLR
from monai.metrics import PanopticQualityMetric
from skimage.measure import label

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s', 
                   handlers=[logging.StreamHandler(sys.stdout)])

logger = logging.getLogger()
logger.info("Starting training and validation process") 

def build_efficient_sam_vitt():
    return build_efficient_sam(
        encoder_patch_embed_dim=192,
        encoder_num_heads=3,
        checkpoint="EfficientSAM/weights/efficient_sam_vitt.pt",
    ).eval()

class _LoRA_qkv(nn.Module):
    """Handles the qkv projection in attention layers for EfficientSAM"""
    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
        r: int,
        alpha: int
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.r = r
        self.alpha = alpha

    def forward(self, x):
        # Original QKV computation
        qkv = self.qkv(x)  # Shape: B,N,3*C
        
        # Get original tensor dimensions
        B, N, three_C = qkv.shape
        C = three_C // 3
        
        # LoRA computations
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        
        # Add LoRA contributions to q and v parts of qkv
        qkv_split = qkv.reshape(B, N, 3, C)
        qkv_split[:, :, 0, :] += (self.alpha / self.r) * new_q
        qkv_split[:, :, 2, :] += (self.alpha / self.r) * new_v
        
        # Return the reshaped tensor in the original format
        return qkv_split.reshape(B, N, three_C)
    
class LoRA_EfficientSAM(nn.Module):
    """Applies low-rank adaptation to EfficientSAM's image encoder.
    
    Args:
        sam_model: an EfficientSAM model
        r: rank of LoRA
        alpha: scaling factor
        lora_layer_indices: which layers to apply LoRA (default: all)
    """
    
    def __init__(self, sam_model, r=4, alpha=4, lora_layer_indices=None):
        super(LoRA_EfficientSAM, self).__init__()
        
        self.sam = sam_model
        self.image_encoder = self.sam.image_encoder
        
        self.w_As = nn.ModuleList()
        self.w_Bs = nn.ModuleList()
        
        for param in self.sam.parameters():
            param.requires_grad_(False)
            
        if lora_layer_indices is None:
            lora_layer_indices = list(range(len(self.image_encoder.blocks)))
            
        for block_idx, block in enumerate(self.image_encoder.blocks):
            if block_idx not in lora_layer_indices:
                continue
                
            # Get the qkv projection layer
            w_qkv_linear = block.attn.qkv
            dim = w_qkv_linear.in_features
            
            # Create LoRA projections for q and v
            w_a_linear_q = nn.Linear(dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, dim, bias=False)
            w_a_linear_v = nn.Linear(dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, dim, bias=False)
            
            # Make sure these new LoRA layers are trainable
            w_a_linear_q.requires_grad_(True)
            w_b_linear_q.requires_grad_(True)
            w_a_linear_v.requires_grad_(True)
            w_b_linear_v.requires_grad_(True)
            
            # Save the LoRA projections
            self.w_As.extend([w_a_linear_q, w_a_linear_v])
            self.w_Bs.extend([w_b_linear_q, w_b_linear_v])
            
            # Replace the QKV projection with a LoRA version
            block.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
                r,
                alpha
            )
            
        # Initialize the LoRA parameters
        self.reset_parameters()
        
        # Double-check that only LoRA parameters are trainable
        for name, param in self.named_parameters():
            if any(f"w_As.{i}" in name or f"w_Bs.{i}" in name for i in range(len(self.w_As))):
                param.requires_grad_(True)
            elif "linear_a_" in name or "linear_b_" in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
    
    def reset_parameters(self):
        # Initialize A with normal distribution and B with zeros
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)
    
    def forward(self, batched_images, batched_points, batched_point_labels, scale_to_original_image_size=True):
        """Forward pass of LoRA-EfficientSAM"""
        return self.sam(batched_images, batched_points, batched_point_labels, scale_to_original_image_size)
    
    def get_image_embeddings(self, batched_images):
        """Get image embeddings from the EfficientSAM model"""
        return self.sam.get_image_embeddings(batched_images)
    
    def predict_masks(self, image_embeddings, batched_points, batched_point_labels, multimask_output, 
                      input_h, input_w, output_h=-1, output_w=-1):
        """Predict masks from image embeddings and prompts"""
        return self.sam.predict_masks(image_embeddings, batched_points, batched_point_labels, 
                                     multimask_output, input_h, input_w, output_h, output_w)
    
    def save_lora_parameters(self, filename):
        """Save only the LoRA parameters to a file"""
        lora_state_dict = {}
        
        # Save weights for all LoRA layers
        for i, w_A in enumerate(self.w_As):
            lora_state_dict[f"w_a_{i}"] = w_A.weight
        
        for i, w_B in enumerate(self.w_Bs):
            lora_state_dict[f"w_b_{i}"] = w_B.weight
            
        torch.save(lora_state_dict, filename)
    
    def load_lora_parameters(self, filename):
        """Load only the LoRA parameters from a file"""
        lora_state_dict = torch.load(filename, map_location="cpu")
        
        for i, w_A in enumerate(self.w_As):
            if f"w_a_{i}" in lora_state_dict:
                w_A.weight.data = lora_state_dict[f"w_a_{i}"]
        
        for i, w_B in enumerate(self.w_Bs):
            if f"w_b_{i}" in lora_state_dict:
                w_B.weight.data = lora_state_dict[f"w_b_{i}"]
                

class SegmentationLoss(nn.Module):
    def __init__(self, lambda_focal=1.0, lambda_dice=1.0, alpha=0.25, gamma=2.0, smooth=1e-6):
        """
        Combined segmentation loss: L_mask = λ_focal * L_focal + λ_dice * L_dice
        
        Args:
            lambda_focal (float): Weight for focal loss
            lambda_dice (float): Weight for dice loss
            alpha (float): Weighting factor in focal loss (for handling class imbalance)
            gamma (float): Focusing parameter in focal loss (for hard examples)
            smooth (float): Smoothing factor to prevent division by zero in dice loss
        """
        super(SegmentationLoss, self).__init__()
        self.lambda_focal = lambda_focal
        self.lambda_dice = lambda_dice
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        
    def forward(self, predictions, targets):
        """
        Calculate the combined segmentation loss.
        
        Args:
            predictions (torch.Tensor): Model predictions 
            targets (torch.Tensor): Ground truth binary masks
            
        Returns:
            torch.Tensor: Scalar loss value
        """
        # Ensure predictions have channel dimension if missing
        if predictions.dim() == 2:
            predictions = predictions.unsqueeze(0)
        if targets.dim() == 2:
            targets = targets.unsqueeze(0)
        
        # Ensure 4D tensor for interpolation (B, C, H, W)
        if predictions.dim() == 3:
            predictions = predictions.unsqueeze(1)
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        
        # Ensure predictions and targets have the same spatial dimensions
        if predictions.shape[-2:] != targets.shape[-2:]:
            predictions = F.interpolate(
                predictions, 
                size=targets.shape[-2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        predictions = predictions.squeeze()
        targets = targets.squeeze()
        
        # Ensure values are in [0, 1] range
        predictions = torch.sigmoid(predictions)
        predictions = torch.clamp(predictions, min=self.smooth, max=1.0-self.smooth)
        
        focal_loss = self.focal_loss(predictions, targets)
        dice_loss = self.dice_loss(predictions, targets)
        combined_loss = self.lambda_focal * focal_loss + self.lambda_dice * dice_loss
        return combined_loss
    
    def focal_loss(self, predictions, targets):
        """
        Calculate the focal loss.
        """
        predictions = predictions.float()
        targets = targets.float()
        
        # Ensure consistent dimensions
        if predictions.dim() == 3:
            predictions = predictions.unsqueeze(0)
        if targets.dim() == 3:
            targets = targets.unsqueeze(0)
        
        bce_loss = F.binary_cross_entropy(
            predictions.squeeze(), 
            targets.squeeze(), 
            reduction='none'
        )
        
        # Calculate focal weights
        if self.gamma > 0:
            pt = torch.where(targets == 1, predictions, 1 - predictions)
            focal_weight = (1 - pt) ** self.gamma
            
            # Apply alpha weighting
            if self.alpha is not None:
                alpha_weight = torch.where(
                    targets == 1, 
                    torch.full_like(predictions, self.alpha),
                    torch.full_like(predictions, 1 - self.alpha)
                )
                focal_weight = alpha_weight * focal_weight
                
            loss = focal_weight * bce_loss
        else:
            loss = bce_loss
            
        return loss.mean()
    
    def dice_loss(self, predictions, targets):
        """
        Calculate the soft dice loss.
        """
        # Flatten the tensors
        predictions_flat = predictions.view(-1)
        targets_flat = targets.view(-1)
        
        # Calculate intersection and union
        intersection = (predictions_flat * targets_flat).sum()
        sum_pred = predictions_flat.sum()
        sum_targets = targets_flat.sum()
        
        # Calculate dice coefficient and dice loss
        dice_coefficient = (2.0 * intersection + self.smooth) / (sum_pred + sum_targets + self.smooth)
        dice_loss = 1.0 - dice_coefficient
        return dice_loss
    
    
class NuInsSegDataset(Dataset):
    def __init__(self, root_dir, tissue_types=None, transform=None, num_points=370):
        """
        Args:
            root_dir: Root directory of the NuInsSeg dataset
            tissue_types: List of tissue types to include (e.g., ['human bladder', 'human brain'])
                         If None, use all available tissues
            transform: Optional transforms to apply
            num_points: Fixed number of prompt points per image (will be padded/truncated to this number)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.num_points = num_points
        
        # Get all available tissue types if none specified
        if tissue_types is None:
            tissue_types = [d for d in os.listdir(root_dir) 
                           if os.path.isdir(os.path.join(root_dir, d))]
        
        self.samples = []
        
        # Collect all samples from specified tissue types
        for tissue in tissue_types:
            tissue_dir = os.path.join(root_dir, tissue)
            
            # Get image paths
            image_dir = os.path.join(tissue_dir, "tissue images")
            if not os.path.exists(image_dir):
                continue
                
            # Get corresponding label mask paths (TIF files)
            label_mask_dir = os.path.join(tissue_dir, "label masks")
            
            if not os.path.exists(label_mask_dir):
                continue
            
            # Find all image files
            image_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))
            
            for img_path in image_files:
                img_filename = os.path.basename(img_path)
                img_name = os.path.splitext(img_filename)[0]
                
                # Label mask has .tif extension
                label_mask_path = os.path.join(label_mask_dir, f"{img_name}.tif")
                
                if os.path.exists(label_mask_path):
                    self.samples.append((img_path, label_mask_path))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label_mask_path = self.samples[idx]
        image = np.array(Image.open(img_path).convert("RGB"))
        instance_mask = tifffile.imread(label_mask_path)
        
        # Create binary mask from instance mask
        binary_mask = (instance_mask > 0).astype(np.float32)
        
        panoptic_mask = np.zeros((2, instance_mask.shape[0], instance_mask.shape[1]), dtype=np.float32)
        panoptic_mask[0] = instance_mask  # Instance IDs
        panoptic_mask[1] = binary_mask
        
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        panoptic_mask_tensor = torch.from_numpy(panoptic_mask)
        
        # Generate prompt points from nuclei masks
        instance_ids = np.unique(instance_mask)[1:]  # Skip background (0)
        
        # Prepare arrays for prompt points and labels
        prompt_points = []
        prompt_labels = []
        
        # Get valid instance points
        for instance_id in instance_ids:
            y, x = np.where(instance_mask == instance_id)
            if len(y) > 0:
                index = np.random.randint(0, len(y))
                prompt_points.append([x[index], y[index]])
                prompt_labels.append(1)
        
        # Determine how many points we have
        num_valid_points = len(prompt_points)
        
        # Handle truncation if we have more than num_points
        if num_valid_points > self.num_points:
            # Randomly select points to keep
            indices = np.random.choice(num_valid_points, self.num_points, replace=False)
            prompt_points = [prompt_points[i] for i in indices]
            prompt_labels = [prompt_labels[i] for i in indices]
        
        # Handle padding if we have fewer than num_points
        elif num_valid_points < self.num_points:
            # Number of padding points needed
            num_padding = self.num_points - num_valid_points
            
            if num_valid_points > 0:
                # Pad with invalid points at (0,0) with label -1
                padding_points = [[0, 0]] * num_padding
                padding_labels = [-1] * num_padding  # Invalid point indicator
                
                prompt_points.extend(padding_points)
                prompt_labels.extend(padding_labels)
            else:
                # If no valid nuclei found, create all padding points
                prompt_points = [[0, 0]] * self.num_points
                prompt_labels = [-1] * self.num_points 
                
        prompt_points_tensor = torch.tensor(prompt_points, dtype=torch.float32)
        prompt_labels_tensor = torch.tensor(prompt_labels, dtype=torch.float32)
        
        # Apply transforms if provided
        if self.transform:
            # Apply transforms that can handle both the image and points
            # This would need a custom transform that modifies both consistently
            pass
        
        return {
            'image': image_tensor,
            'panoptic_mask': panoptic_mask_tensor,  
            'binary_mask': torch.from_numpy(binary_mask),
            'instance_mask': torch.from_numpy(instance_mask).float(), 
            'prompt_points': prompt_points_tensor,
            'prompt_labels': prompt_labels_tensor,
            'image_path': img_path,
            'mask_path': label_mask_path
        }

        
                
class EarlyStopping:
    """Early stopping to prevent overfitting and save the best model.
    
    Args:
        patience (int): How many epochs to wait after last improvement.
        min_delta (float): Minimum change to qualify as an improvement.
        verbose (bool): If True, prints a message for each improvement.
    """
    def __init__(self, patience=5, min_delta=0.0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        
    def __call__(self, val_loss, model, path='best_model.pth'):
        """
        Call after each validation.
        
        Args:
            val_loss (float): Current validation loss
            model (nn.Module): The model being trained
            path (str): Path to save the best model checkpoint
        
        Returns:
            bool: True if model improved, False otherwise
        """
        # Check if the current loss is better than best loss
        if val_loss < self.best_loss - self.min_delta:
            improved = True
            self._save_checkpoint(val_loss, model, path)
            self.best_loss = val_loss
            self.counter = 0
        else:
            improved = False
            self.counter += 1
            if self.verbose:
                logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logger.info('Early stopping triggered')
        
        return improved
    
    def _save_checkpoint(self, val_loss, model, path):
        """Save model checkpoint when validation loss decreases."""
        if self.verbose:
            logger.info(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}). Saving model...')
        
        if isinstance(model, LoRA_EfficientSAM):
            # For LoRA models, save only the LoRA parameters
            model.save_lora_parameters(path)
        else:
            # For regular models, save the entire model
            torch.save(model.state_dict(), path)
            
            
def save_results(results, filename):
    """Save the training results to a JSON file"""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)


# Update the panoptic quality calculation in the validate function

def validate(model, val_loader, criterion, device, max_batches=None):
    """Evaluate the model on validation data"""
    model.eval()
    val_loss = 0.0
    val_batches = 0
    
    # Reset metrics
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    jaccard_metric = MeanIoU()
    pq_metric = PanopticQualityMetric(num_classes=1)
    
    dice_scores = []
    jaccard_scores = []
    panoptic_quality_scores = []
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            # For quick testing, limit the number of batches
            if max_batches is not None and i >= max_batches:
                break
            
            gt_panoptic = batch['panoptic_mask'].to(device)
                
            # Extract data from the batch and move to device
            images = batch['image'].to(device)
            gt_masks = batch['binary_mask'].to(device)
            instance_masks = batch['instance_mask'].to(device)
            
            # Process prompt points and labels
            prompt_points = batch['prompt_points'].to(device)
            prompt_labels = batch['prompt_labels'].to(device)
            
            # Reshape for EfficientSAM
            if len(prompt_points.shape) == 2:
                prompt_points = prompt_points.unsqueeze(0)
            prompt_points = prompt_points.unsqueeze(1)
            
            if len(prompt_labels.shape) == 1:
                prompt_labels = prompt_labels.unsqueeze(0)
            prompt_labels = prompt_labels.unsqueeze(1)
            
            try:
                # Forward pass
                predicted_masks, iou_predictions = model(images, prompt_points, prompt_labels)
                
                # Process masks to the right shape
                if iou_predictions is not None:
                    best_mask_idx = iou_predictions.argmax(dim=2)
                    B, Q = best_mask_idx.shape
                    
                    predicted_mask = torch.zeros(B, Q, predicted_masks.shape[3], predicted_masks.shape[4], device=device)
                    
                    for b in range(B):
                        for q in range(Q):
                            predicted_mask[b, q] = predicted_masks[b, q, best_mask_idx[b, q]]
                    
                    if Q == 1:
                        predicted_mask = predicted_mask.squeeze(1)
                else:
                    predicted_mask = predicted_masks[:, 0, 0]
                
                # Resize if needed
                if predicted_mask.shape[-2:] != gt_masks.shape[-2:]:
                    predicted_mask = F.interpolate(
                        predicted_mask.unsqueeze(1),
                        size=gt_masks.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(1)
                
                predicted_mask_binary = (predicted_mask > 0.5).float()

                batch_size = predicted_mask_binary.shape[0]
                h, w = predicted_mask_binary.shape[-2:]
                pred_panoptic = torch.zeros((batch_size, 2, h, w), device=device)

                for b in range(batch_size):
                    # For predictions, label connected components to get instance IDs
                    pred_binary = predicted_mask_binary[b].cpu().numpy()
                    pred_instances = torch.from_numpy(label(pred_binary)).float().to(device)
                    
                    # Fill panoptic prediction tensor
                    pred_panoptic[b, 0] = pred_instances  # Instance IDs 
                    pred_panoptic[b, 1] = (pred_instances > 0).float()  # Class IDs
                
                # Calculate PQ metric
                pq_metric.reset()
                pq_metric(y_pred=pred_panoptic, y=gt_panoptic)
                pq = pq_metric.aggregate().mean().item()
                panoptic_quality_scores.append(pq)
                
                # Calculate loss
                loss = criterion(predicted_mask, gt_masks)
                
                output_one_hot = predicted_mask_binary.unsqueeze(0)
                mask_one_hot = gt_masks.unsqueeze(0)
                
                dice_metric.reset()
                dice_metric(y_pred=output_one_hot.float(), y=mask_one_hot.float())
                dice = dice_metric.aggregate().mean().item()
                dice_scores.append(dice)
                
                jaccard_metric.reset()
                jaccard_metric(y_pred=output_one_hot.float(), y=mask_one_hot.float())
                jaccard = jaccard_metric.aggregate().mean().item()
                jaccard_scores.append(jaccard)
                                     
                val_loss += loss.item()
                val_batches += 1
                
            except Exception as e:
                logger.info(f"Error in validation batch: {e}")
                import traceback
                traceback.print_exc()
                import sys; sys.exit(1)
                # continue
    
    avg_val_loss = val_loss / max(1, val_batches)
    avg_dice = np.mean(dice_scores) if dice_scores else 0
    avg_jaccard = np.mean(jaccard_scores) if jaccard_scores else 0
    avg_panoptic_quality = np.mean(panoptic_quality_scores) if panoptic_quality_scores else 0
    
    return avg_val_loss, avg_dice, avg_jaccard, avg_panoptic_quality


def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, max_batches=None):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0.0
    batch_count = 0
    
    # For quick testing, limit the number of batches
    if max_batches is not None:
        logger.info(f"QUICK TEST MODE: Limiting to {max_batches} batches per epoch")
    
    for i, batch in enumerate(train_loader):
        # For quick testing, limit the number of batches
        if max_batches is not None and i >= max_batches:
            break
            
        # Extract data from the batch and move to device
        images = batch['image'].to(device)
        gt_masks = batch['binary_mask'].to(device)
        
        # Ensure images are in the right format (channels first, normalized)
        if images.dim() == 3:
            images = images.unsqueeze(0)
        
        # Ensure gt_masks are in the right format
        if gt_masks.dim() == 2:
            gt_masks = gt_masks.unsqueeze(0)
        
        # Process prompt points and labels
        prompt_points = batch['prompt_points'].to(device)
        prompt_labels = batch['prompt_labels'].to(device)
        
        # Reshape prompt points and labels
        if len(prompt_points.shape) == 2:
            prompt_points = prompt_points.unsqueeze(0)
        prompt_points = prompt_points.unsqueeze(1)
        
        if len(prompt_labels.shape) == 1:
            prompt_labels = prompt_labels.unsqueeze(0)
        prompt_labels = prompt_labels.unsqueeze(1)
        
        try:
            # Forward pass
            predicted_masks, iou_predictions = model(images, prompt_points, prompt_labels)

            if iou_predictions is not None:
                best_mask_idx = iou_predictions.argmax(dim=2)
                B, Q = best_mask_idx.shape
                
                predicted_mask = torch.zeros(B, Q, predicted_masks.shape[3], predicted_masks.shape[4], device=device)
                
                for b in range(B):
                    for q in range(Q):
                        predicted_mask[b, q] = predicted_masks[b, q, best_mask_idx[b, q]]
                
                if Q == 1:
                    predicted_mask = predicted_mask.squeeze(1)
            else:
                predicted_mask = predicted_masks[:, 0, 0]
            
            # Resize if needed
            if predicted_mask.shape[-2:] != gt_masks.shape[-2:]:
                predicted_mask = F.interpolate(
                    predicted_mask.unsqueeze(1),
                    size=gt_masks.shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(1)
            
            loss = criterion(predicted_mask, gt_masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            
        except Exception as e:
            logger.info(f"Error in training batch: {e}")
            import traceback
            traceback.print_exc()
            # continue
    
    return epoch_loss / max(1, batch_count)

def plot_metrics(results, save_dir):
    """Plot training and validation loss curves with all folds on the same plot"""
    
    plt.figure(figsize=(12, 6))
    
    # Plot Training Loss
    plt.subplot(1, 2, 1)
    for fold, fold_results in results['folds'].items():
        fold_epochs = list(range(1, len(fold_results['train_losses']) + 1))
        plt.plot(
            fold_epochs, 
            fold_results['train_losses'], 
            label=f'Fold {fold}'
        )
    
    # plot the average training loss
    if results['avg_train_losses']:
        plt.plot(
            list(range(1, len(results['avg_train_losses']) + 1)), 
            results['avg_train_losses'], 
            'k--', 
            linewidth=2,
            label='Average'
        )
    
    plt.title('Training Loss Across Folds')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for fold, fold_results in results['folds'].items():
        fold_epochs = list(range(1, len(fold_results['val_losses']) + 1))
        plt.plot(
            fold_epochs, 
            fold_results['val_losses'], 
            label=f'Fold {fold}'
        )
    
    if results['avg_val_losses']:
        plt.plot(
            list(range(1, len(results['avg_val_losses']) + 1)), 
            results['avg_val_losses'], 
            'k--', 
            linewidth=2,
            label='Average'
        )
    
    plt.title('Validation Loss Across Folds')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'))
    plt.close()
    
    
def main():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    root_dir = "/home/cap5516.student2/cap5516_spring25/segmentation_efficientsam_histological/NuInsSeg"
    seed = seed
    num_epochs = 100
    batch_size = 24
    num_folds = 5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    patience = 10
    
    rank = 4
    alpha = 4 * rank
    
    learning_rate = 1e-4
    max_scheduler_iter = 10
    min_learning_rate = 1e-6
    lambda_focal = 1.0
    lambda_dice = 1.0
    loss_alpha = 0.25
    loss_gamma = 2.0
    early_stopping_min_delta = 0.001
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_dir = f"results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    checkpoint_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    dataset = NuInsSegDataset(root_dir=root_dir)
    logger.info(f"Dataset loaded with {len(dataset)} samples")
    
    # QUICK TEST MODE
    QUICK_TEST = False  # Set to False for full training

    if QUICK_TEST:
        # Create a tiny dataset with just one sample
        dataset = torch.utils.data.Subset(dataset, [0, 0, 0, 0])
        num_folds = 2  
        num_epochs = 10 
        logger.info("QUICK TEST MODE: Using duplicated samples")
        
    logger.info(f"Seed: {seed}")
    logger.info(f'Using device: {device}')
    logger.info(f'Batch size: {batch_size}')
    logger.info(f'Number of epochs: {num_epochs}')
    logger.info(f'Number of folds: {num_folds}')
    logger.info(f'LoRA Rank: {rank}')
    logger.info(f'LoRA Alpha: {alpha}')
    logger.info(f'Patience: {patience}')
    logger.info(f'Root directory: {root_dir}')
    logger.info(f'Checkpoint directory: {checkpoint_dir}')
    logger.info(f"learning rate: {learning_rate}")
    logger.info(f"max scheduler iter: {max_scheduler_iter}")
    logger.info(f"min learning rate: {min_learning_rate}")
    logger.info(f"lambda focal: {lambda_focal}")
    logger.info(f"lambda dice: {lambda_dice}")
    logger.info(f"loss alpha: {loss_alpha}")
    logger.info(f"loss gamma: {loss_gamma}")
    logger.info(f"early stopping min delta: {early_stopping_min_delta}")
    
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    
    results = {
        'folds': {},
        'avg_train_losses': [],
        'avg_val_losses': [],
        'avg_dice_scores': [],
        'avg_jaccard_scores': [],
        'avg_panoptic_quality_scores': [],
        'best_val_losses': [],
        'best_dice_scores': [],
        'best_jaccard_scores': [],
        'best_panoptic_quality_scores': []
    }
    
    fold_train_losses = [[] for _ in range(num_epochs)]
    fold_val_losses = [[] for _ in range(num_epochs)]
    fold_dice_scores = [[] for _ in range(num_epochs)]
    fold_jaccard_scores = [[] for _ in range(num_epochs)]
    fold_panoptic_quality_scores = [[] for _ in range(num_epochs)]
    
    for fold, (train_indices, val_indices) in enumerate(kfold.split(dataset)):
        logger.info(f"{'='*20} Fold {fold+1}/{num_folds} {'='*20}")
        
        results['folds'][str(fold)] = {
            'train_losses': [],
            'val_losses': [],
            'dice_scores': [],
            'jaccard_scores': [],
            'panoptic_quality_scores': [],
            'best_val_loss': float('inf'),
            'best_epoch': 0,
            'best_dice': 0.0,
            'best_jaccard': 0.0,
            'best_panoptic_quality': 0.0
        }
        
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=4
        )
        
        val_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=4
        )
        
        sam_model = build_efficient_sam_vitt().to(device)
        
        total_params_sam = sum(p.numel() for p in sam_model.parameters())
        logger.info(f"EfficientSAM-Tiny total parameters: {total_params_sam:,}")

        # get trainable parameter count for sam_model before LoRA
        trainable_params_sam = sum(p.numel() for p in sam_model.parameters() if p.requires_grad)
        logger.info(f"EfficientSAM-Tiny trainable parameters before LoRA: {trainable_params_sam:,}")
        
        # Freeze base model parameters
        sam_model.eval()  
        for param in sam_model.parameters():
            param.requires_grad_(False)
            
        frozen_params_sam = sum(p.numel() for p in sam_model.parameters() if p.requires_grad)
        logger.info(f"EfficientSAM-Tiny trainable parameters after freezing: {frozen_params_sam:,}")
        
        # Create LoRA model k fold
        lora_sam = LoRA_EfficientSAM(sam_model, r=rank, alpha=alpha).to(device)
        
        # get trainable and total parameter counts for the LoRA model
        trainable_params_lora = sum(p.numel() for p in lora_sam.parameters() if p.requires_grad)
        total_params_lora = sum(p.numel() for p in lora_sam.parameters())
        logger.info(f"LoRA-EfficientSAM trainable parameters: {trainable_params_lora:,}")
        logger.info(f"LoRA-EfficientSAM total parameters: {total_params_lora:,}")
        
        optimizer = torch.optim.Adam([p for p in lora_sam.parameters() if p.requires_grad], lr=learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=max_scheduler_iter, eta_min=min_learning_rate)
        criterion = SegmentationLoss(lambda_focal=lambda_focal, lambda_dice=lambda_dice, alpha=loss_alpha, gamma=loss_gamma)
        
        early_stopping = EarlyStopping(patience=patience, min_delta=early_stopping_min_delta, verbose=True)
        
        fold_checkpoint_path = os.path.join(checkpoint_dir, f"fold_{fold}_best_model.pth")
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            # Train for one epoch (with batch limiting in quick test mode)
            max_batches = 2 if QUICK_TEST else None
            train_loss = train_epoch(lora_sam, train_loader, criterion, optimizer, scheduler, device, max_batches)
            logger.info(f"  Training Loss: {train_loss:.4f}")
            
            # Validate (with batch limiting in quick test mode)
            val_loss, dice_score, jaccard_score, panoptic_quality = validate(lora_sam, val_loader, criterion, device, max_batches)
            logger.info(f"  Validation Loss: {val_loss:.4f}, Dice: {dice_score:.4f}, Jaccard: {jaccard_score:.4f}, PQ: {panoptic_quality:.4f}")
            
            results['folds'][str(fold)]['train_losses'].append(train_loss)
            results['folds'][str(fold)]['val_losses'].append(val_loss)
            results['folds'][str(fold)]['dice_scores'].append(dice_score)
            results['folds'][str(fold)]['jaccard_scores'].append(jaccard_score)
            results['folds'][str(fold)]['panoptic_quality_scores'].append(panoptic_quality)
            
            # Collect metrics across folds for averaging
            fold_train_losses[epoch].append(train_loss)
            fold_val_losses[epoch].append(val_loss)
            fold_dice_scores[epoch].append(dice_score)
            fold_jaccard_scores[epoch].append(jaccard_score)
            fold_panoptic_quality_scores[epoch].append(panoptic_quality)
            
            improved = early_stopping(val_loss, lora_sam, path=fold_checkpoint_path)
            if improved:
                results['folds'][str(fold)]['best_val_loss'] = val_loss
                results['folds'][str(fold)]['best_epoch'] = epoch + 1
                results['folds'][str(fold)]['best_dice'] = dice_score
                results['folds'][str(fold)]['best_jaccard'] = jaccard_score
                results['folds'][str(fold)]['best_panoptic_quality'] = panoptic_quality
                logger.info(f"  New best model saved with validation loss: {val_loss:.4f}")
            
            # Check if early stopping should be triggered
            if early_stopping.early_stop:
                logger.info("  Early stopping triggered")
                break
        
        # Record best metrics for this fold
        results['best_val_losses'].append(results['folds'][str(fold)]['best_val_loss'])
        results['best_dice_scores'].append(results['folds'][str(fold)]['best_dice'])
        results['best_jaccard_scores'].append(results['folds'][str(fold)]['best_jaccard'])
        results['best_panoptic_quality_scores'].append(results['folds'][str(fold)]['best_panoptic_quality'])
        
        # Clean up to free memory for next fold
        del sam_model, lora_sam, optimizer, criterion, early_stopping
        torch.cuda.empty_cache()
    
    # Calculate average metrics across all folds for each epoch
    for epoch in range(num_epochs):
        if fold_train_losses[epoch]: 
            results['avg_train_losses'].append(np.mean(fold_train_losses[epoch]))
            results['avg_val_losses'].append(np.mean(fold_val_losses[epoch]))
            results['avg_dice_scores'].append(np.mean(fold_dice_scores[epoch]))
            results['avg_jaccard_scores'].append(np.mean(fold_jaccard_scores[epoch]))
            results['avg_panoptic_quality_scores'].append(np.mean(fold_panoptic_quality_scores[epoch]))
    
    # Save final results
    results['overall'] = {
        'avg_best_val_loss': np.mean(results['best_val_losses']),
        'avg_best_dice': np.mean(results['best_dice_scores']),
        'avg_best_jaccard': np.mean(results['best_jaccard_scores']),
        'avg_best_panoptic_quality': np.mean(results['best_panoptic_quality_scores']),
        'std_val_loss': np.std(results['best_val_losses']),
        'std_dice': np.std(results['best_dice_scores']),
        'std_jaccard': np.std(results['best_jaccard_scores']),
        'std_panoptic_quality': np.std(results['best_panoptic_quality_scores'])
    }
    
    results_file = os.path.join(results_dir, 'training_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    plot_metrics(results, results_dir)
    
    for fold, fold_results in results['folds'].items():
        logger.info(f"{'='*20} Fold {fold} Results {'='*20}")
        logger.info(f"  Best Validation Loss: {fold_results['best_val_loss']:.4f}")
        logger.info(f"  Best Dice Score: {fold_results['best_dice']:.4f}")
        logger.info(f"  Best Jaccard Score: {fold_results['best_jaccard']:.4f}")
        logger.info(f"  Best Panoptic Quality: {fold_results['best_panoptic_quality']:.4f}")
    
    logger.info("\nCross-validation completed.")
    logger.info(f"Average best validation loss: {results['overall']['avg_best_val_loss']:.4f} ± {results['overall']['std_val_loss']:.4f}")
    logger.info(f"Average best Dice coefficient: {results['overall']['avg_best_dice']:.4f} ± {results['overall']['std_dice']:.4f}")
    logger.info(f"Average best Jaccard index: {results['overall']['avg_best_jaccard']:.4f} ± {results['overall']['std_jaccard']:.4f}")
    logger.info(f"Average best Panoptic Quality: {results['overall']['avg_best_panoptic_quality']:.4f} ± {results['overall']['std_panoptic_quality']:.4f}")
    logger.info(f"Results saved to {results_file}")
    logger.info(f"Metric plots saved to {results_dir}")
    
    
if __name__ == "__main__":
    main()