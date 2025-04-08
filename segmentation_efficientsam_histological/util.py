import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.filters as kf

class EnhancedSegmentationLoss(nn.Module):
    def __init__(self, lambda_focal=1.0, lambda_dice=1.0, lambda_boundary=0.5, 
                 alpha=0.25, gamma=2.0, smooth=1e-6):
        super(EnhancedSegmentationLoss, self).__init__()
        self.lambda_focal = lambda_focal
        self.lambda_dice = lambda_dice
        self.lambda_boundary = lambda_boundary
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        
    def forward(self, predictions, targets):
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
        combined_loss = self.lambda_focal * focal_loss + self.lambda_dice * dice_loss + self.lambda_boundary * boundary_loss
        return combined_loss
        
    
    def _boundary_loss(self, predictions, targets):
        """Calculate boundary awareness loss to enhance separation between instances."""
        target_grad_x = kf.spatial_gradient(targets.unsqueeze(1), order=1, mode='sobel', normalized=True)[:, :, 0]
        target_grad_y = kf.spatial_gradient(targets.unsqueeze(1), order=1, mode='sobel', normalized=True)[:, :, 1]
        pred_grad_x = kf.spatial_gradient(predictions.unsqueeze(1), order=1, mode='sobel', normalized=True)[:, :, 0]
        pred_grad_y = kf.spatial_gradient(predictions.unsqueeze(1), order=1, mode='sobel', normalized=True)[:, :, 1]
        
        # boundary loss
        loss_x = F.mse_loss(pred_grad_x, target_grad_x)
        loss_y = F.mse_loss(pred_grad_y, target_grad_y)
        
        return loss_x + loss_y   
    
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
    
