import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import numpy as np
import torchvision


def gram_matrix(features):
    """
    Compute Gram matrix from feature maps
    features: [B, C, H, W]
    returns: [B, C, C]
    """
    B, C, H, W = features.size()
    features = features.view(B, C, H * W)  # Flatten spatial dims
    
    # Compute correlation: F * F^T
    gram = torch.bmm(features, features.transpose(1, 2))
    
    # Normalize by number of elements
    gram = gram / (C * H * W)
    
    return gram


class StyleLoss(nn.Module):
    def __init__(self, layers=['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3']):
        super().__init__()
        
        # Use pretrained VGG for feature extraction
        vgg = torchvision.models.vgg19(pretrained=True).features
        self.vgg = vgg.eval()
        
        # Freeze VGG weights
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        # Define which layers to use
        self.layer_names = layers
        self.layer_indices = {
            'conv1_2': 3,
            'conv2_2': 8,
            'conv3_3': 17,
            'conv4_3': 26,
            'conv5_3': 35
        }
    
    def extract_features(self, x):
        """Extract features from specified VGG layers"""
        features = {}
        for name, module in self.vgg._modules.items():
            x = module(x)
            idx = int(name)
            
            for layer_name, layer_idx in self.layer_indices.items():
                if idx == layer_idx and layer_name in self.layer_names:
                    features[layer_name] = x
        
        return features
    
    def forward(self, generated, target):
        """
        generated: [B, 3, H, W] - your generated RGB images
        target: [B, 3, H, W] - real RGB images (can be unpaired!)
        """
        # Normalize for VGG (ImageNet stats)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(generated.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(generated.device)
        
        gen_norm = (generated - mean) / std
        target_norm = (target - mean) / std
        
        # Extract features
        gen_features = self.extract_features(gen_norm)
        target_features = self.extract_features(target_norm)
        
        # Compute Gram matrices and compare
        style_loss = 0.0
        
        for layer_name in self.layer_names:
            gen_gram = gram_matrix(gen_features[layer_name])
            target_gram = gram_matrix(target_features[layer_name])
            
            style_loss += F.mse_loss(gen_gram, target_gram)
        
        return style_loss / len(self.layer_names)
    

class ClassConditionalStyleLoss(nn.Module):
    def __init__(self, num_classes, layers=['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3']):
        super().__init__()
        self.style_loss = StyleLoss(layers=layers)
        self.num_classes = num_classes
    
    def forward(self, generated, real, gen_masks, real_masks):
        """
        Compare style per semantic class
        generated: [B, 3, H, W]
        real: [B, 3, H, W]  
        gen_masks, real_masks: [B, H, W] - segmentation
        """
        total_loss = 0.0
        num_valid = 0

        for class_id in range(self.num_classes):
            # Get masks for this class
            gen_class_mask = (gen_masks == class_id)
            real_class_mask = (real_masks == class_id)
            
            if gen_class_mask.sum() < 100 or real_class_mask.sum() < 100:
                continue  # Skip if too few pixels
            
            # Mask out other classes (set to 0 or mean)
            gen_masked = generated.clone()
            real_masked = real.clone()

            gen_mask_expanded = gen_class_mask.unsqueeze(1).expand(-1, 3, -1, -1)   # [B, 3, H, W]
            real_mask_expanded = real_class_mask.unsqueeze(1).expand(-1, 3, -1, -1) # [B, 3, H, W]

            gen_masked[~gen_mask_expanded] = 0
            real_masked[~real_mask_expanded] = 0
            
            # Compute style loss for this class
            loss = self.style_loss(gen_masked, real_masked)
            total_loss += loss
            num_valid += 1
        
        return total_loss / max(num_valid, 1)