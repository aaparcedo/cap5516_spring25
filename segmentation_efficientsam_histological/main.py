import logging
import sys

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s', 
                   handlers=[logging.StreamHandler(sys.stdout)])

logger = logging.getLogger()
logger.info("Starting training and validation process") 

import random
import os
import matplotlib.pyplot as plt
import json
from datetime import datetime
logger.info("random os matplotlib json datetime") 
 
import numpy as np
logger.info("PIL") 

import torch
import torch.nn.functional as F
logger.info("torch") 

from torch.utils.data import DataLoader, SubsetRandomSampler
logger.info("Dataset, DataLoader, SubsetRandomSampler") 
from EfficientSAM.efficient_sam.efficient_sam import build_efficient_sam
logger.info("efficientsam") 

from sklearn.model_selection import KFold
logger.info("sklearn") 
from torch.optim.lr_scheduler import CosineAnnealingLR
logger.info("CosineAnnealingLR") 
from skimage.measure import label
logger.info("skimage")
logger.info("torchvision") 

from monai.metrics import DiceMetric
from monai.metrics import MeanIoU
from monai.metrics import PanopticQualityMetric
logger.info("monai") 
from tqdm import tqdm
logger.info("tqdm")


from dataset import AugmentedNuInsSegDataset
from util import EnhancedSegmentationLoss   
from model import LoRA_EfficientSAM
from visualize import visualize_predictions


from EfficientSAM.efficient_sam.efficient_sam import build_efficient_sam


# from https://github.com/yformer/EfficientSAM
def build_efficient_sam_vitt():
    return build_efficient_sam(
        encoder_patch_embed_dim=192,
        encoder_num_heads=3,
        checkpoint="EfficientSAM/weights/efficient_sam_vitt.pt",
    ).eval()

          
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


def validate(model, val_loader, criterion, device):
    """Evaluate the model on validation data"""
    model.eval()
    val_loss = 0.0
    val_batches = 0
    
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    jaccard_metric = MeanIoU()
    pq_metric = PanopticQualityMetric(num_classes=1)
    
    dice_scores = []
    jaccard_scores = []
    panoptic_quality_scores = []
    
    with torch.no_grad():
        for i, batch in tqdm(enumerate(val_loader), desc="(val) processing batches...", total=len(val_loader)):
            gt_panoptic = batch['panoptic_mask'].to(device)
                
            # Extract data from the batch and move to device
            images = batch['image'].to(device).float()
            gt_masks = batch['binary_mask'].to(device)
            gt_instance_mask = batch['instance_mask'].to(device)
            
            # Process prompt points and labels
            prompt_points = batch['prompt_points'].to(device).float()
            prompt_labels = batch['prompt_labels'].to(device).float()
            
            # Reshape for EfficientSAM
            if len(prompt_points.shape) == 2:
                prompt_points = prompt_points.unsqueeze(0)
            prompt_points = prompt_points.unsqueeze(1)
            
            if len(prompt_labels.shape) == 1:
                prompt_labels = prompt_labels.unsqueeze(0)
            prompt_labels = prompt_labels.unsqueeze(1)
            
            try:
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
                
                pq_metric.reset()
                pq_metric(y_pred=pred_panoptic, y=gt_panoptic)
                pq = pq_metric.aggregate().mean().item()
                panoptic_quality_scores.append(pq)
                
                loss = criterion(predicted_mask, gt_masks, gt_instance_mask)
                
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



def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, apply_augmentation=True):
    """Train for one epoch with optional augmentation"""
    model.train()
    epoch_loss = 0.0
    batch_count = 0

    for i, batch in tqdm(enumerate(train_loader), desc="(train) Processing batches", total=len(train_loader)):
            
        # Extract data from the batch and move to device
        images = batch['image'].to(device).float()
        gt_masks = batch['binary_mask'].to(device)
        gt_instance_mask = batch['instance_mask'].to(device)
        
        # Ensure images are in the right format (channels first, normalized)
        if images.dim() == 3:
            images = images.unsqueeze(0)
        
        # Ensure gt_masks are in the right format
        if gt_masks.dim() == 2:
            gt_masks = gt_masks.unsqueeze(0)
        
        # Process prompt points and labels
        prompt_points = batch['prompt_points'].to(device).float()
        prompt_labels = batch['prompt_labels'].to(device).float()
        
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
            
            if predicted_mask.shape[-2:] != gt_masks.shape[-2:]:
                predicted_mask = F.interpolate(
                    predicted_mask.unsqueeze(1),
                    size=gt_masks.shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(1)
            
            loss = criterion(predicted_mask, gt_masks, gt_instance_mask)
            
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
    
class Config():
    def __init__(self):
        self.seed = 42
        self.root_dir = "/home/cap5516.student2/cap5516_spring25/segmentation_efficientsam_histological/NuInsSeg"
        self.num_epochs = 100
        self.batch_size = 2
        self.num_folds = 5
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.patience = 10
        self.apply_augmentation = False
        self.augmentation_prob = 1.0
        self.visualize_results = True
        self.num_vis_samples = 10
        self.rank = 64
        self.alpha = 256
        self.learning_rate = 5e-5
        self.max_scheduler_iter = 10
        self.min_learning_rate = 5e-7
        self.lambda_focal = 1.0
        self.lambda_dice = 1.0
        self.lambda_boundary = 1.0
        self.lambda_contrastive = 1.0
        self.loss_alpha = 0.75
        self.loss_gamma = 2.0
        self.early_stopping_min_delta = 0.001

    
def main():
    cfg = Config()
    logger.info(cfg.__dict__)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_dir = f"results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    checkpoint_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if cfg.visualize_results:
        vis_dir = os.path.join(results_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
    
    dataset = AugmentedNuInsSegDataset(root_dir=cfg.root_dir, include_augmentations=cfg.apply_augmentation)
    logger.info(f"Dataset loaded with {len(dataset)} samples")
    logger.info(f'Checkpoint directory: {checkpoint_dir}')

    def worker_init_fn(worker_id):
        np.random.seed(cfg.seed + worker_id)
        random.seed(cfg.seed + worker_id)
    
    kfold = KFold(n_splits=cfg.num_folds, shuffle=True, random_state=cfg.seed)
    
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
    
    fold_train_losses = [[] for _ in range(cfg.num_epochs)]
    fold_val_losses = [[] for _ in range(cfg.num_epochs)]
    fold_dice_scores = [[] for _ in range(cfg.num_epochs)]
    fold_jaccard_scores = [[] for _ in range(cfg.num_epochs)]
    fold_panoptic_quality_scores = [[] for _ in range(cfg.num_epochs)]
    
    for fold, (train_indices, val_indices) in enumerate(kfold.split(dataset)):
        logger.info(f"{'='*20} Fold {fold+1}/{cfg.num_folds} {'='*20}")
        
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
            batch_size=cfg.batch_size,
            sampler=train_sampler,
            num_workers=2,
            worker_init_fn=worker_init_fn
        )
        
        val_loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            sampler=val_sampler,
            num_workers=1,
            worker_init_fn=worker_init_fn
        )
        
        original_sam_model = build_efficient_sam_vitt().to(cfg.device) 
        sam_model = build_efficient_sam_vitt().to(cfg.device)
        
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
        lora_sam = LoRA_EfficientSAM(config=cfg, sam_model=sam_model, r=cfg.rank, alpha=cfg.alpha).to(cfg.device)
        
        # get trainable and total parameter counts for the LoRA model
        trainable_params_lora = sum(p.numel() for p in lora_sam.parameters() if p.requires_grad)
        total_params_lora = sum(p.numel() for p in lora_sam.parameters())
        logger.info(f"LoRA-EfficientSAM trainable parameters: {trainable_params_lora:,}")
        logger.info(f"LoRA-EfficientSAM total parameters: {total_params_lora:,}")
        
        optimizer = torch.optim.Adam([p for p in lora_sam.parameters() if p.requires_grad], lr=cfg.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.max_scheduler_iter, eta_min=cfg.min_learning_rate)
        criterion = EnhancedSegmentationLoss(cfg)
        
        early_stopping = EarlyStopping(patience=cfg.patience, min_delta=cfg.early_stopping_min_delta, verbose=True)
        
        fold_checkpoint_path = os.path.join(checkpoint_dir, f"fold_{fold}_best_model.pth")
        
        for epoch in range(cfg.num_epochs):
            logger.info(f"Epoch {epoch+1}/{cfg.num_epochs}")
            
            train_loss = train_epoch(
                lora_sam, 
                train_loader, 
                criterion, 
                optimizer, 
                scheduler, 
                cfg.device,  
                apply_augmentation=cfg.apply_augmentation
            )
            logger.info(f"  Training Loss: {train_loss:.4f}")
            
            val_loss, dice_score, jaccard_score, panoptic_quality = validate(lora_sam, val_loader, criterion, cfg.device)
            logger.info(f"  Validation Loss: {val_loss:.4f}, Dice: {dice_score:.4f}, Jaccard: {jaccard_score:.4f}, PQ: {panoptic_quality:.4f}")
            
            results['folds'][str(fold)]['train_losses'].append(train_loss)
            results['folds'][str(fold)]['val_losses'].append(val_loss)
            results['folds'][str(fold)]['dice_scores'].append(dice_score)
            results['folds'][str(fold)]['jaccard_scores'].append(jaccard_score)
            results['folds'][str(fold)]['panoptic_quality_scores'].append(panoptic_quality)
            
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
                
                if cfg.visualize_results:
                    fold_vis_dir = os.path.join(vis_dir, f"fold_{fold}_epoch_{epoch+1}")
                    os.makedirs(fold_vis_dir, exist_ok=True)
                    
                    logger.info(f"  Generating visualizations for best model...")
                    visualize_predictions(
                        lora_sam, 
                        dataset, 
                        cfg.device, 
                        fold_vis_dir, 
                        num_samples=min(5, cfg.num_vis_samples), 
                    )
            
            if early_stopping.early_stop:
                logger.info("  Early stopping triggered")
                break
        
        results['best_val_losses'].append(results['folds'][str(fold)]['best_val_loss'])
        results['best_dice_scores'].append(results['folds'][str(fold)]['best_dice'])
        results['best_jaccard_scores'].append(results['folds'][str(fold)]['best_jaccard'])
        results['best_panoptic_quality_scores'].append(results['folds'][str(fold)]['best_panoptic_quality'])
        
        if cfg.visualize_results:
            best_model = LoRA_EfficientSAM(config=cfg, sam_model=original_sam_model, r=cfg.rank, alpha=cfg.alpha).to(cfg.device)
            best_model.load_lora_parameters(fold_checkpoint_path)
            best_model.eval()
            
            final_vis_dir = os.path.join(vis_dir, f"fold_{fold}_final")
            os.makedirs(final_vis_dir, exist_ok=True)
            
            logger.info(f"Visualizing final predictions for fold {fold}...")
            visualize_predictions(
                best_model, 
                dataset, 
                cfg.device, 
                final_vis_dir, 
                num_samples=cfg.num_vis_samples, 
            )
            
            del best_model
        del sam_model, lora_sam, optimizer, criterion, early_stopping
        torch.cuda.empty_cache()
    
    for epoch in range(cfg.num_epochs):
        if fold_train_losses[epoch]: 
            results['avg_train_losses'].append(np.mean(fold_train_losses[epoch]))
            results['avg_val_losses'].append(np.mean(fold_val_losses[epoch]))
            results['avg_dice_scores'].append(np.mean(fold_dice_scores[epoch]))
            results['avg_jaccard_scores'].append(np.mean(fold_jaccard_scores[epoch]))
            results['avg_panoptic_quality_scores'].append(np.mean(fold_panoptic_quality_scores[epoch]))
    
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
    if cfg.visualize_results:
        logger.info(f"Visualizations saved to {vis_dir}")
    
    
if __name__ == "__main__":
    main()