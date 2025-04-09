import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.filters as kf

class EnhancedSegmentationLoss(nn.Module):
    def __init__(self, config, smooth=1e-6):
        super(EnhancedSegmentationLoss, self).__init__()
        self.lambda_focal = config.lambda_focal
        self.lambda_dice = config.lambda_dice
        self.lambda_boundary = config.lambda_boundary
        self.lambda_contrastive = config.lambda_contrastive
        self.alpha = config.loss_alpha
        self.gamma = config.loss_gamma
        self.smooth = smooth
        
    def forward(self, predictions, targets, instance_masks=None):
        # Ensure predictions and targets have compatible dimensions
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
        
        # Squeeze dimensions for loss calculation
        predictions = predictions.squeeze()
        targets = targets.squeeze()
        
        predictions = torch.sigmoid(predictions)
        predictions = torch.clamp(predictions, min=self.smooth, max=1.0-self.smooth)
        targets = torch.clamp(targets, min=0.0, max=1.0) 
        
        focal_loss = self.focal_loss(predictions, targets)
        dice_loss = self.dice_loss(predictions, targets)
        boundary_loss = self._boundary_loss(predictions, targets)
        contrastive_loss = self.instance_contrastive_loss(predictions, instance_masks)
        combined_loss = 0
        combined_loss += self.lambda_focal * focal_loss 
        combined_loss += self.lambda_dice * dice_loss 
        combined_loss += self.lambda_boundary * boundary_loss
        combined_loss += self.lambda_contrastive * contrastive_loss
        return combined_loss
    
    def instance_contrastive_loss(self, predictions, instance_masks):
        """Encourage separation between adjacent instances."""
        batch_size = predictions.shape[0]
        loss = 0
        
        for b in range(batch_size):
            pred = predictions[b]
            instances = instance_masks[b]
            instance_ids = torch.unique(instances)[1:]  # Skip background
            
            if len(instance_ids) <= 1:
                continue
                
            # Calculate instance means in prediction space
            instance_means = {}
            for id in instance_ids:
                mask = (instances == id)
                if mask.sum() > 0:
                    instance_means[id.item()] = pred[mask].mean()
            
            # Calculate contrastive loss between instances
            contrast_loss = 0
            count = 0
            
            for id1 in instance_means:
                for id2 in instance_means:
                    if id1 < id2:  # Each pair only once
                        mean1 = instance_means[id1]
                        mean2 = instance_means[id2]
                        # Encourage separation between instance predictions
                        contrast_loss += torch.exp(-torch.abs(mean1 - mean2))
                        count += 1
            
            if count > 0:
                loss += contrast_loss / count
        
        return loss / batch_size
        
    
    def _boundary_loss(self, predictions, targets):
        """Enhanced boundary awareness loss with robust error handling."""
        # Ensure consistent dimensions and types
        if predictions.dim() == 2:
            predictions = predictions.unsqueeze(0)
        if targets.dim() == 2:
            targets = targets.unsqueeze(0)
        
        # Ensure floating point type and proper value range
        predictions = predictions.float()
        targets = targets.float()
        
        # Ensure proper shape for gradient calculation (B, C, H, W)
        if predictions.dim() == 3:
            predictions = predictions.unsqueeze(1)
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        
        # Clamp values to avoid numerical issues
        predictions = torch.clamp(predictions, min=self.smooth, max=1.0-self.smooth)
        targets = torch.clamp(targets, min=0, max=1.0)
        
        try:
            # Calculate gradients
            target_grad_x = kf.spatial_gradient(targets, order=1, mode='sobel', normalized=True)[:, :, 0]
            target_grad_y = kf.spatial_gradient(targets, order=1, mode='sobel', normalized=True)[:, :, 1]
            pred_grad_x = kf.spatial_gradient(predictions, order=1, mode='sobel', normalized=True)[:, :, 0]
            pred_grad_y = kf.spatial_gradient(predictions, order=1, mode='sobel', normalized=True)[:, :, 1]
            
            # Magnitude of gradients (edge maps)
            target_grad_mag = torch.sqrt(target_grad_x**2 + target_grad_y**2 + 1e-6)
            pred_grad_mag = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + 1e-6)
            
            # Weighted boundary loss - put more emphasis on boundary pixels
            boundary_weight = 1.0 + 5.0 * target_grad_mag  # Increase weight at boundaries
            
            # Weighted MSE loss for magnitude
            loss_mag = F.mse_loss(pred_grad_mag * boundary_weight, 
                                target_grad_mag * boundary_weight, 
                                reduction='mean')
            
            # Direction loss component
            direction_loss = torch.tensor(0.0, device=predictions.device)
            mask = (target_grad_mag > 0.1)  # Only consider significant gradients
            
            if mask.sum() > 0:
                # Normalize directions for cosine similarity
                target_norm = torch.sqrt(target_grad_x[mask]**2 + target_grad_y[mask]**2 + 1e-6)
                pred_norm = torch.sqrt(pred_grad_x[mask]**2 + pred_grad_y[mask]**2 + 1e-6)
                
                target_dx = target_grad_x[mask] / target_norm
                target_dy = target_grad_y[mask] / target_norm
                pred_dx = pred_grad_x[mask] / pred_norm
                pred_dy = pred_grad_y[mask] / pred_norm
                
                # Cosine similarity
                cos_sim = (target_dx * pred_dx + target_dy * pred_dy)
                direction_loss = torch.mean(1.0 - cos_sim)
            
            return loss_mag + direction_loss
            
        except RuntimeError as e:
            # Fallback to a simpler gradient calculation if Kornia fails
            print(f"Warning: Enhanced gradient calculation failed, using fallback method. Error: {e}")
            
    def focal_loss(self, predictions, targets):
        predictions = predictions.float()
        targets = targets.float()
        
        if predictions.dim() == 3:
            predictions = predictions.unsqueeze(0)
        if targets.dim() == 3:
            targets = targets.unsqueeze(0)
        
        predictions = torch.clamp(predictions, min=self.smooth, max=1.0-self.smooth)
        targets = torch.clamp(targets, min=0.0, max=1.0)
        
        bce_loss = F.binary_cross_entropy(
            predictions.squeeze(), 
            targets.squeeze(), 
            reduction='none'
        )
        
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
        Calculate the soft dice loss with proper value clamping.
        """
        predictions = torch.clamp(predictions, min=self.smooth, max=1.0-self.smooth)
        intersection = (predictions.view(-1) * targets.view(-1)).sum()
        sum_pred = predictions.view(-1).sum()
        sum_targets = targets.view(-1).sum()
        dice_coefficient = (2.0 * intersection + self.smooth) / (sum_pred + sum_targets + self.smooth)
        dice_loss = 1.0 - dice_coefficient
        return dice_loss
    
