import random
import os
import glob
import tifffile
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import cv2
from torchvision import transforms

class NuInsSegDataset(Dataset):
    def __init__(self, root_dir, tissue_types=None, num_points=370, apply_augmentation=True):
        """
        Args:
            root_dir: Root directory of the NuInsSeg dataset
            tissue_types: List of tissue types to include
            num_points: Fixed number of prompt points per image
            apply_augmentation: Whether to include augmented versions of samples
        """
        self.root_dir = root_dir
        self.num_points = num_points
        self.apply_augmentation = apply_augmentation
        
        tissue_types = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        
        # Collect all original samples
        self.original_samples = []
        
        # Collect all samples from specified tissue types
        for tissue in tissue_types:
            tissue_dir = os.path.join(root_dir, tissue)
            
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
                    self.original_samples.append((img_path, label_mask_path))
        
        self.samples = []
        
        # Add all original samples
        for sample in self.original_samples:
            self.samples.append((sample[0], sample[1], False, None))  # Original sample
            
            # Add augmented versions if enabled
            if apply_augmentation:
                # Choose a random augmentation type for each sample
                aug_types = ['rotation', 'flip', 'color']
                aug_type = random.choice(aug_types)
                self.samples.append((sample[0], sample[1], True, aug_type))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label_mask_path, is_augmented, aug_type = self.samples[idx]
        
        # Original sample loading code
        image = np.array(Image.open(img_path).convert("RGB"))
        instance_mask = tifffile.imread(label_mask_path)
        
        # Create binary mask from instance mask
        binary_mask = (instance_mask > 0).astype(np.float32) 
        
        panoptic_mask = np.zeros((2, instance_mask.shape[0], instance_mask.shape[1]), dtype=np.float32)
        panoptic_mask[0] = instance_mask  # Instance IDs
        panoptic_mask[1] = binary_mask

        normalized_image = self.normalize_he_stain(image)
        image_tensor = torch.from_numpy(normalized_image).permute(2, 0, 1).float() / 255.0

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                        std=[0.229, 0.224, 0.225])
        image_tensor = normalize(image_tensor)
        panoptic_mask_tensor = torch.from_numpy(panoptic_mask)
        
        # Generate prompt points from nuclei masks
        instance_ids = np.unique(instance_mask)[1:] 
        
        # Prepare arrays for prompt points and labels
        prompt_points = []
        prompt_labels = []
        
        # Get valid instance points using center of mass
        for instance_id in instance_ids:
            y, x = np.where(instance_mask == instance_id)
            if len(y) > 0:
                # Calculate center of mass
                center_y = int(np.mean(y))
                center_x = int(np.mean(x))
                prompt_points.append([center_x, center_y])
                prompt_labels.append(1)
        
        # Handle truncation/padding of points as in original code
        num_valid_points = len(prompt_points)
        
        if num_valid_points > self.num_points:
            indices = np.random.choice(num_valid_points, self.num_points, replace=False)
            prompt_points = [prompt_points[i] for i in indices]
            prompt_labels = [prompt_labels[i] for i in indices]
        elif num_valid_points < self.num_points:
            num_padding = self.num_points - num_valid_points
            
            if num_valid_points > 0:
                padding_points = [[0, 0]] * num_padding
                padding_labels = [-1] * num_padding
                
                prompt_points.extend(padding_points)
                prompt_labels.extend(padding_labels)
            else:
                prompt_points = [[0, 0]] * self.num_points
                prompt_labels = [-1] * self.num_points 
                
        prompt_points_tensor = torch.tensor(prompt_points, dtype=torch.float32)
        prompt_labels_tensor = torch.tensor(prompt_labels, dtype=torch.float32)
        
        sample = {
            'image': image_tensor,
            'panoptic_mask': panoptic_mask_tensor,  
            'binary_mask': torch.from_numpy(binary_mask),
            'instance_mask': torch.from_numpy(instance_mask).float(), 
            'prompt_points': prompt_points_tensor,
            'prompt_labels': prompt_labels_tensor,
            'image_path': img_path,
            'mask_path': label_mask_path
        }
        
        # Apply augmentation if this is an augmented sample
        if is_augmented:
            sample = self.apply_single_augmentation(sample, augmentation_type=aug_type)
            
        return sample
                
    def normalize_he_stain(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        normalized_lab = cv2.merge((cl, a, b))
        normalized_rgb = cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2RGB)
        return normalized_rgb
    
    def apply_single_augmentation(self, sample, augmentation_type=None):
        """
        Apply a single type of augmentation to a sample.
        
        Args:
            sample: Dictionary with 'image', 'binary_mask', 'prompt_points', etc.
            augmentation_type: One of 'rotation', 'flip', 'color', 'elastic', or None
                If None, a random augmentation type will be selected.
                
        Returns:
            Augmented sample
        """
                
        if augmentation_type is None:
            augmentation_types = ['rotation', 'flip', 'color']
            augmentation_type = random.choice(augmentation_types)
        
        # Extract data from sample
        image = sample['image'].numpy().transpose(1, 2, 0)  # CHW -> HWC
        
        # Denormalize for augmentation
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image * std + mean) * 255.0
        image = image.astype(np.uint8)
        
        binary_mask = sample['binary_mask'].numpy()
        instance_mask = sample['instance_mask'].numpy()
        prompt_points = sample['prompt_points'].numpy()
        prompt_labels = sample['prompt_labels'].numpy()
        
        # Only transform valid prompt points (those with label == 1)
        valid_points_mask = (prompt_labels == 1)
        
        H, W = image.shape[:2]
        
        if augmentation_type == 'rotation':
            angle = random.uniform(-30, 30)
            center = (W // 2, H // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Rotate image and masks
            image = cv2.warpAffine(image, rotation_matrix, (W, H), 
                                flags=cv2.INTER_LINEAR, borderValue=0)
            binary_mask = cv2.warpAffine(binary_mask, rotation_matrix, (W, H), 
                                        flags=cv2.INTER_NEAREST, borderValue=0)
            instance_mask = cv2.warpAffine(instance_mask, rotation_matrix, (W, H), 
                                        flags=cv2.INTER_NEAREST, borderValue=0)
            
            # Rotate valid points
            for i, point in enumerate(prompt_points[valid_points_mask]):
                x, y = point
                # Adjust coordinates to be relative to rotation center
                x -= center[0]
                y -= center[1]
                
                # Apply rotation
                angle_rad = np.deg2rad(angle)
                x_new = np.cos(angle_rad) * x - np.sin(angle_rad) * y
                y_new = np.sin(angle_rad) * x + np.cos(angle_rad) * y
                
                # Restore coordinates relative to image origin
                x_new += center[0]
                y_new += center[1]
                
                prompt_points[valid_points_mask][i] = [x_new, y_new]
                
        elif augmentation_type == 'flip':
            # Random horizontal or vertical flip
            flip_type = random.choice(['horizontal', 'vertical'])
            
            if flip_type == 'horizontal':
                # Horizontal flip
                image = cv2.flip(image, 1)  # 1 for horizontal flip
                binary_mask = cv2.flip(binary_mask, 1)
                instance_mask = cv2.flip(instance_mask, 1)
                
                # Flip x-coordinates of points
                for i, point in enumerate(prompt_points[valid_points_mask]):
                    x, y = point
                    prompt_points[valid_points_mask][i] = [W - 1 - x, y]
                    
            else:
                # Vertical flip
                image = cv2.flip(image, 0)  # 0 for vertical flip
                binary_mask = cv2.flip(binary_mask, 0)
                instance_mask = cv2.flip(instance_mask, 0)
                
                # Flip y-coordinates of points
                for i, point in enumerate(prompt_points[valid_points_mask]):
                    x, y = point
                    prompt_points[valid_points_mask][i] = [x, H - 1 - y]
                    
        elif augmentation_type == 'color':
            # Random color jittering
            jitter_type = random.choice(['brightness', 'contrast', 'hue', 'saturation'])
            
            if jitter_type == 'brightness':
                # Brightness adjustment
                brightness_factor = random.uniform(0.7, 1.3)
                image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)
                
            elif jitter_type == 'contrast':
                # Contrast adjustment
                contrast_factor = random.uniform(0.7, 1.3)
                mean_val = np.mean(image, axis=(0, 1), keepdims=True)
                image = cv2.convertScaleAbs((image - mean_val) * contrast_factor + mean_val)
                
            elif jitter_type in ['hue', 'saturation']:
                # Convert to HSV for hue/saturation adjustments
                hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
                
                if jitter_type == 'hue':
                    # Hue adjustment
                    hue_factor = random.uniform(-20, 20)  # Hue is in range [0, 179] in OpenCV
                    hsv[:, :, 0] = (hsv[:, :, 0] + hue_factor) % 180
                    
                else:  # saturation
                    # Saturation adjustment
                    saturation_factor = random.uniform(0.7, 1.3)
                    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)
                    
                # Convert back to RGB
                image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
                
        elif augmentation_type == 'stain_norm':
            # Stain normalization for H&E images
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            
            # Merge channels back
            normalized_lab = cv2.merge((cl, a, b))
            image = cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2RGB)
        
        prompt_points = np.clip(prompt_points, 0, [W-1, H-1])
        
        image = image.astype(np.float32) / 255.0
        image = (image - mean) / std
        image = torch.from_numpy(image).permute(2, 0, 1)  # Back to CHW
        
        sample['image'] = image
        sample['binary_mask'] = torch.from_numpy(binary_mask).float()
        sample['instance_mask'] = torch.from_numpy(instance_mask).float()
        sample['panoptic_mask'] = torch.stack([
            torch.from_numpy(instance_mask).float(),
            torch.from_numpy(binary_mask).float()
        ])
        sample['prompt_points'] = torch.from_numpy(prompt_points).float()
        
        return sample

                
class AugmentedNuInsSegDataset(NuInsSegDataset):
    def __init__(self, root_dir, tissue_types=None, num_points=370, include_augmentations=True):
        """
        Args:
            root_dir: Root directory of the NuInsSeg dataset
            tissue_types: List of tissue types to include
            num_points: Fixed number of prompt points per image
            include_augmentations: Whether to include augmented versions of samples
        """
        self.root_dir = root_dir
        self.num_points = num_points
        self.include_augmentations = include_augmentations
        
        # Get all available tissue types if none specified
        if tissue_types is None:
            tissue_types = [d for d in os.listdir(root_dir) 
                           if os.path.isdir(os.path.join(root_dir, d))]
        
        # Collect all original samples
        self.original_samples = []
        
        for tissue in tissue_types:
            tissue_dir = os.path.join(root_dir, tissue)
            
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
                    self.original_samples.append((img_path, label_mask_path))
        
        # Create the expanded sample list
        self.samples = []
        # Add all original samples
        for sample in self.original_samples:
            self.samples.append((sample[0], sample[1], False, None))  # Original sample
            
            # Add augmented versions if enabled
            if include_augmentations:
                # Add one augmented version of each sample with a random augmentation type
                aug_types = ['rotation', 'flip', 'color']
                aug_type = random.choice(aug_types)
                self.samples.append((sample[0], sample[1], True, aug_type))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label_mask_path, is_augmented, aug_type = self.samples[idx]
        
        # Load and process the image
        image = np.array(Image.open(img_path).convert("RGB"))
        instance_mask = tifffile.imread(label_mask_path)
        
        # Create binary mask from instance mask
        binary_mask = (instance_mask > 0).astype(np.float32) 
        
        panoptic_mask = np.zeros((2, instance_mask.shape[0], instance_mask.shape[1]), dtype=np.float32)
        panoptic_mask[0] = instance_mask  # Instance IDs
        panoptic_mask[1] = binary_mask

        normalized_image = self.normalize_he_stain(image)
        image_tensor = torch.from_numpy(normalized_image).permute(2, 0, 1).float() / 255.0

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                        std=[0.229, 0.224, 0.225])
        image_tensor = normalize(image_tensor)
        panoptic_mask_tensor = torch.from_numpy(panoptic_mask)
        
        # Generate prompt points from nuclei masks
        instance_ids = np.unique(instance_mask)[1:]  # Skip background (0)
        
        # Prepare arrays for prompt points and labels
        prompt_points = []
        prompt_labels = []
        
        # Get valid instance points using center of mass
        for instance_id in instance_ids:
            y, x = np.where(instance_mask == instance_id)
            if len(y) > 0:
                # Calculate center of mass
                center_y = int(np.mean(y))
                center_x = int(np.mean(x))
                prompt_points.append([center_x, center_y])
                prompt_labels.append(1)
        
        # Process points (truncate/pad) as in your original code
        num_valid_points = len(prompt_points)
        
        if num_valid_points > self.num_points:
            indices = np.random.choice(num_valid_points, self.num_points, replace=False)
            prompt_points = [prompt_points[i] for i in indices]
            prompt_labels = [prompt_labels[i] for i in indices]
        elif num_valid_points < self.num_points:
            num_padding = self.num_points - num_valid_points
            
            if num_valid_points > 0:
                padding_points = [[0, 0]] * num_padding
                padding_labels = [-1] * num_padding
                
                prompt_points.extend(padding_points)
                prompt_labels.extend(padding_labels)
            else:
                prompt_points = [[0, 0]] * self.num_points
                prompt_labels = [-1] * self.num_points 
                
        prompt_points_tensor = torch.tensor(prompt_points, dtype=torch.float32)
        prompt_labels_tensor = torch.tensor(prompt_labels, dtype=torch.float32)
        
        sample = {
            'image': image_tensor,
            'panoptic_mask': panoptic_mask_tensor,
            'binary_mask': torch.from_numpy(binary_mask),
            'instance_mask': torch.from_numpy(instance_mask).float(),
            'prompt_points': prompt_points_tensor,
            'prompt_labels': prompt_labels_tensor,
            'image_path': img_path,
            'mask_path': label_mask_path
        }
        
        # Apply augmentation if this is an augmented sample
        if is_augmented and aug_type is not None:
            sample = self.apply_single_augmentation(sample, augmentation_type=aug_type)
        
        return sample


def augment_batch(batch, augmentation_type=None, augmentation_prob=0.5, device='cuda'):
    """
    Apply a single augmentation to a batch of samples with correctly transformed prompt points.
    
    Args:
        batch: Dictionary with batched tensors
        augmentation_type: One of 'rotation', 'flip', 'color', or None for random selection
        augmentation_prob: Probability of applying augmentation to the batch
        device: Device where the batch tensors are located
        
    Returns:
        Augmented batch
    """
    
    if random.random() > augmentation_prob:
        return batch
    
    if augmentation_type is None:
        augmentation_types = ['rotate', 'flip', 'color']
        augmentation_type = random.choice(augmentation_types)
    
    # Extract tensors from batch
    images = batch['image']
    binary_masks = batch['binary_mask']
    instance_masks = batch['instance_mask']
    prompt_points = batch['prompt_points'].clone()  # Clone to avoid modifying original
    prompt_labels = batch['prompt_labels']
    
    batch_size = images.shape[0]
    
    _, C, H, W = images.shape
    
 
    if augmentation_type == 'color':
        jitter_type = random.choice(['brightness', 'contrast', 'saturation'])
        
        images_cpu = images.cpu().numpy()
        
        # Apply jittering to each image in the batch
        for b in range(batch_size):
            img = np.transpose(images_cpu[b], (1, 2, 0))  # CHW -> HWC
            
            # Denormalize for color operations
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = img * std + mean
            
            if jitter_type == 'brightness':
                # Brightness adjustment
                brightness_factor = random.uniform(0.7, 1.3)
                img = np.clip(img * brightness_factor, 0, 1)
            
            elif jitter_type == 'contrast':
                # Contrast adjustment
                contrast_factor = random.uniform(0.7, 1.3)
                mean_val = np.mean(img, axis=(0, 1), keepdims=True)
                img = np.clip((img - mean_val) * contrast_factor + mean_val, 0, 1)
            
            elif jitter_type == 'saturation':
                # Saturation adjustment - convert to HSV, adjust S, convert back
                img_hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
                saturation_factor = random.uniform(0.7, 1.3)
                img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] * saturation_factor, 0, 255)
                img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
            
            # Renormalize
            img = (img - mean) / std
            images_cpu[b] = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        
        # Convert back to tensor and move to device
        images = torch.from_numpy(images_cpu).to(device)
    
    for b in range(batch_size):
        valid_indices = (prompt_labels[b] == 1)
        if valid_indices.sum() > 0:
            prompt_points[b, valid_indices, 0] = torch.clamp(prompt_points[b, valid_indices, 0], 0, W-1)
            prompt_points[b, valid_indices, 1] = torch.clamp(prompt_points[b, valid_indices, 1], 0, H-1)
    
    batch['image'] = images
    batch['binary_mask'] = binary_masks
    batch['instance_mask'] = instance_masks
    batch['prompt_points'] = prompt_points
    
    batch['panoptic_mask'] = torch.stack([
        instance_masks,
        binary_masks
    ], dim=1)
    
    return batch
