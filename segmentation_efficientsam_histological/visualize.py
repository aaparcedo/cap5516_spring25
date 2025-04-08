import random
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from skimage.measure import label
from skimage.segmentation import find_boundaries
from monai.metrics import DiceMetric
from monai.metrics import MeanIoU




def visualize_augmentations(dataset, output_dir, num_samples=5):
    """
    Visualize the effect of different augmentations on random samples.
    
    Args:
        dataset: The dataset to sample from
        output_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
    """    
    os.makedirs(output_dir, exist_ok=True)
    dataset_size = len(dataset)
    sample_indices = random.sample(range(dataset_size), min(num_samples, dataset_size))
    
    augmentation_types = ['color']
    
    for idx, sample_idx in enumerate(sample_indices):
        sample = dataset[sample_idx]
        
        # Get original image and mask
        orig_image = sample['image'].numpy().transpose(1, 2, 0)
        
        # Denormalize image for visualization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        orig_image = (orig_image * std + mean)
        orig_image = np.clip(orig_image, 0, 1)
        
        orig_binary_mask = sample['binary_mask'].numpy()
        orig_instance_mask = sample['instance_mask'].numpy()
        orig_points = sample['prompt_points'].numpy()
        orig_labels = sample['prompt_labels'].numpy()
        
        # Create a figure with rows for each augmentation type
        fig, axs = plt.subplots(1 + len(augmentation_types), 3, figsize=(15, 5 * (1 + len(augmentation_types))))
        
        # Plot original image, mask, and points
        axs[0, 0].imshow(orig_image)
        axs[0, 0].set_title('Original Image')
        axs[0, 0].axis('off')
        
        # Plot original binary mask
        axs[0, 1].imshow(orig_image)
        axs[0, 1].imshow(orig_binary_mask, alpha=0.5, cmap='Reds')
        axs[0, 1].set_title('Original Binary Mask')
        axs[0, 1].axis('off')
        
        # Plot original instance mask with different colors
        axs[0, 2].imshow(orig_image)
        # Create custom colormap for instances
        cmap = plt.cm.get_cmap('tab20', np.max(orig_instance_mask) + 1)
        instance_colored = cmap(orig_instance_mask)
        instance_colored[orig_instance_mask == 0] = [0, 0, 0, 0]  # Make background transparent
        axs[0, 2].imshow(instance_colored, alpha=0.7)
        axs[0, 2].set_title('Original Instance Mask')
        axs[0, 2].axis('off')
        
        # Plot prompt points on top of all original images
        valid_points = orig_points[orig_labels == 1]
        for pt in valid_points:
            for ax in axs[0]:
                ax.plot(pt[0], pt[1], 'yo', markersize=5)
        
        # Apply and visualize each augmentation type
        for aug_idx, aug_type in enumerate(augmentation_types):
            # Create a deep copy of the sample for augmentation
            aug_sample = {k: v.clone() if isinstance(v, torch.Tensor) else v 
                         for k, v in sample.items()}
            
            # Apply augmentation
            aug_sample = dataset.apply_single_augmentation(aug_sample, augmentation_type=aug_type)
            
            # Get augmented image and mask
            aug_image = aug_sample['image'].numpy().transpose(1, 2, 0)
            aug_image = (aug_image * std + mean)
            aug_image = np.clip(aug_image, 0, 1)
            
            aug_binary_mask = aug_sample['binary_mask'].numpy()
            aug_instance_mask = aug_sample['instance_mask'].numpy()
            aug_points = aug_sample['prompt_points'].numpy()
            aug_labels = aug_sample['prompt_labels'].numpy()
            
            # Plot augmented image
            axs[aug_idx+1, 0].imshow(aug_image)
            axs[aug_idx+1, 0].set_title(f'{aug_type.capitalize()} - Image')
            axs[aug_idx+1, 0].axis('off')
            
            # Plot augmented binary mask
            axs[aug_idx+1, 1].imshow(aug_image)
            axs[aug_idx+1, 1].imshow(aug_binary_mask, alpha=0.5, cmap='Reds')
            axs[aug_idx+1, 1].set_title(f'{aug_type.capitalize()} - Binary Mask')
            axs[aug_idx+1, 1].axis('off')
            
            # Plot augmented instance mask
            axs[aug_idx+1, 2].imshow(aug_image)
            # Color the instances
            aug_instance_colored = cmap(aug_instance_mask)
            aug_instance_colored[aug_instance_mask == 0] = [0, 0, 0, 0]  # Make background transparent
            axs[aug_idx+1, 2].imshow(aug_instance_colored, alpha=0.7)
            axs[aug_idx+1, 2].set_title(f'{aug_type.capitalize()} - Instance Mask')
            axs[aug_idx+1, 2].axis('off')
            
            valid_aug_points = aug_points[aug_labels == 1]
            for pt in valid_aug_points:
                for ax in axs[aug_idx+1]:
                    ax.plot(pt[0], pt[1], 'yo', markersize=5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'augmentation_sample_{sample_idx}.png'), dpi=150)
        plt.close(fig)
        
    print(f"Augmentation visualizations saved to {output_dir}")


def visualize_predictions(model, dataset, device, output_dir, num_samples=5, fold_idx=0):
    """
    Visualize model predictions vs ground truth masks.
    
    Args:
        model: The trained model
        dataset: The full dataset
        device: Device to run inference on ('cuda' or 'cpu')
        output_dir: Directory to save visualization images
        num_samples: Number of samples to visualize
        fold_idx: Current fold index for naming
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    original_indices = []
    for i, sample_data in enumerate(dataset.samples):
        img_path, mask_path, is_augmented, _ = sample_data
        if not is_augmented:  # Only select non-augmented samples
            original_indices.append(i)
    
    # Select from original samples only
    indices = np.random.choice(
        original_indices, 
        min(num_samples, len(original_indices)), 
        replace=False
    )
    
    vis_subset = Subset(dataset, indices)
    vis_loader = DataLoader(vis_subset, batch_size=1, shuffle=False)
    
    with torch.no_grad():
        for i, batch in enumerate(vis_loader):
            images = batch['image'].to(device).float()
            gt_masks = batch['binary_mask'].to(device)
            prompt_points = batch['prompt_points'].to(device).unsqueeze(1).float()
            prompt_labels = batch['prompt_labels'].to(device).unsqueeze(1).float()
            
            image_path = batch['image_path'][0]
            sample_name = os.path.basename(image_path).split('.')[0]
            
            predicted_masks, iou_predictions = model(images, prompt_points, prompt_labels)
            
            # Get best mask based on IOU predictions
            if iou_predictions is not None:
                best_mask_idx = iou_predictions.argmax(dim=2)
                predicted_mask = predicted_masks[0, 0, best_mask_idx[0, 0]]
            else:
                predicted_mask = predicted_masks[0, 0, 0]
            
            if predicted_mask.shape != gt_masks.shape[-2:]:
                predicted_mask = F.interpolate(
                    predicted_mask.unsqueeze(0).unsqueeze(0),
                    size=gt_masks.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                ).squeeze()
            
            # Convert to binary mask
            pred_binary = (predicted_mask > 0.5).float()
            
            # Get instance masks
            gt_instance_mask = batch['instance_mask'][0].cpu().numpy()
            pred_binary_np = pred_binary.cpu().numpy()
            pred_instance_mask = label(pred_binary_np)
            
            # Convert back to original image for visualization
            # De-normalize the image
            img = images[0].cpu().numpy().transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1) * 255
            img = img.astype(np.uint8)
            
            # Load original image for better visualization
            try:
                orig_img = np.array(Image.open(image_path).convert("RGB"))
            except:
                orig_img = img
            
            # Create visualization with 6 subplots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # Original image
            axes[0, 0].imshow(orig_img)
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
            
            # Ground truth binary mask
            axes[0, 1].imshow(orig_img)
            gt_mask_np = gt_masks[0].cpu().numpy()
            axes[0, 1].imshow(gt_mask_np, alpha=0.5, cmap='Reds')
            axes[0, 1].set_title('Ground Truth Mask')
            axes[0, 1].axis('off')
            
            # Predicted binary mask
            axes[0, 2].imshow(orig_img)
            axes[0, 2].imshow(pred_binary_np, alpha=0.5, cmap='Blues')
            axes[0, 2].set_title('Predicted Mask')
            axes[0, 2].axis('off')
            
            # Ground truth instance mask (colored)
            axes[1, 0].imshow(orig_img)
            cmap = plt.cm.tab20
            
            # For ground truth instances
            gt_instance_colored = np.zeros((*gt_instance_mask.shape, 4))
            unique_instances = np.unique(gt_instance_mask)
            unique_instances = unique_instances[unique_instances != 0]  # Remove background

            for idx, inst_id in enumerate(unique_instances):
                color = cmap(idx % 20)  # Use the enumeration index instead of instance ID
                mask = gt_instance_mask == inst_id
                gt_instance_colored[mask] = color
            
            axes[1, 0].imshow(gt_instance_colored, alpha=0.7)
            axes[1, 0].set_title('Ground Truth Instances')
            axes[1, 0].axis('off')
            
            # Predicted instance mask (colored)
            axes[1, 1].imshow(orig_img)
            
            
            # Predicted instance mask (colored)
            pred_instance_colored = np.zeros((*pred_instance_mask.shape, 4))
            unique_pred_instances = np.unique(pred_instance_mask)
            unique_pred_instances = unique_pred_instances[unique_pred_instances != 0]  # Remove background

            for idx, inst_id in enumerate(unique_pred_instances):
                color = cmap(idx % 20)  # Use the enumeration index instead of instance ID
                mask = pred_instance_mask == inst_id
                pred_instance_colored[mask] = color
            
            axes[1, 1].imshow(pred_instance_colored, alpha=0.7)
            axes[1, 1].set_title('Predicted Instances')
            axes[1, 1].axis('off')
            
            axes[1, 2].imshow(orig_img)
            
            tp_mask = np.logical_and(pred_binary_np > 0.5, gt_mask_np > 0.5)
            fp_mask = np.logical_and(pred_binary_np > 0.5, gt_mask_np <= 0.5)
            fn_mask = np.logical_and(pred_binary_np <= 0.5, gt_mask_np > 0.5)
            
            error_vis = np.zeros((*gt_mask_np.shape, 3))
            error_vis[tp_mask, 1] = 1.0  # Green for true positives
            error_vis[fp_mask, 0] = 1.0  # Red for false positives
            error_vis[fn_mask, 2] = 1.0  # Blue for false negatives
            
            axes[1, 2].imshow(error_vis, alpha=0.5)
            axes[1, 2].set_title('Error Analysis (TP=green, FP=red, FN=blue)')
            axes[1, 2].axis('off')
            
            # Add prompt points visualization to the original image
            for pt_idx in range(prompt_points.shape[2]):
                if prompt_labels[0, 0, pt_idx] > 0:  # Only visualize valid points
                    x, y = prompt_points[0, 0, pt_idx].cpu().numpy()
                    # Draw circles on all plots in the first row
                    for ax in axes[0, :]:
                        ax.plot(x, y, 'yo', markersize=8, alpha=0.7)
            
            dice_metric = DiceMetric(include_background=False, reduction="mean")
            jaccard_metric = MeanIoU()
            
            # Prepare tensors for metrics
            pred_tensor = pred_binary.unsqueeze(0).unsqueeze(0)
            gt_tensor = gt_masks.unsqueeze(0)
            
            dice_metric.reset()
            dice_metric(y_pred=pred_tensor, y=gt_tensor)
            dice_score = dice_metric.aggregate().item()
            
            jaccard_metric.reset()
            jaccard_metric(y_pred=pred_tensor, y=gt_tensor)
            iou_score = jaccard_metric.aggregate().item()
            
            # Calculate instance-level statistics
            pred_instances = len(np.unique(pred_instance_mask)) - 1  # Subtract background
            gt_instances = len(np.unique(gt_instance_mask)) - 1
            
            # Add metrics to the plot title
            plt.suptitle(f"Sample: {sample_name} | Dice: {dice_score:.4f} | IoU: {iou_score:.4f}\n"
                         f"GT Instances: {gt_instances} | Pred Instances: {pred_instances}", 
                         fontsize=16)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.90)  # Make room for the suptitle
            
            save_path = os.path.join(output_dir, f"sample_{sample_name}_visualization.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Create a detailed error map with instance boundaries
            fig, ax = plt.subplots(figsize=(12, 12))
            ax.imshow(orig_img)
            
            # Create a composite error visualization
            error_vis = np.zeros((*gt_mask_np.shape, 4))  # RGBA
            
            error_vis[tp_mask, 1] = 1.0  # Green for true positives
            error_vis[tp_mask, 3] = 0.3  # Alpha
            
            error_vis[fp_mask, 0] = 1.0  # Red for false positives
            error_vis[fp_mask, 3] = 0.5  # Alpha
            
            error_vis[fn_mask, 2] = 1.0  # Blue for false negatives
            error_vis[fn_mask, 3] = 0.5  # Alpha
            
            ax.imshow(error_vis)
            
            gt_boundaries = find_boundaries(gt_instance_mask.astype(np.int32), mode='outer')
            pred_boundaries = find_boundaries(pred_instance_mask.astype(np.int32), mode='outer')
            
            ax.contour(gt_boundaries, colors='yellow', linewidths=1.0, alpha=0.7)
            ax.contour(pred_boundaries, colors='cyan', linewidths=1.0, alpha=0.7)
            
            ax.set_title(f"Detailed Error Analysis\nGT Boundaries (yellow), Pred Boundaries (cyan)", fontsize=14)
            ax.axis('off')
            
            # Save the detailed error map
            detail_save_path = os.path.join(output_dir, f"sample_{sample_name}_detailed_error.png")
            plt.savefig(detail_save_path, dpi=200, bbox_inches='tight')
            plt.close()
            
            print(f"Saved visualizations for sample {sample_name}")
            
    print(f"Visualizations saved to {output_dir}")
