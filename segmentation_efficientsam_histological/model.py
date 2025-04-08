
import math
import torch
import torch.nn as nn



# Modified version of code from https://github.com/JamesQFreeman/LoRA-ViT
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
                