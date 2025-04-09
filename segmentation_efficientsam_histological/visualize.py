import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from skimage.measure import label
from skimage.segmentation import find_boundaries
from monai.metrics import DiceMetric
from monai.metrics import MeanIoU


def visualize_predictions(model, dataset, device, output_dir, num_samples=5):
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
        if not is_augmented:  
            original_indices.append(i)
    
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
            
            # import code; code.interact(local=locals())
            
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
