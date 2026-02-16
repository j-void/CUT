import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
import numpy as np
import torchvision





class ContrastiveAlignment(nn.Module):
    def __init__(self, layers = [0, 1]):
        super().__init__()
        self.layers = layers

    def tensor_correlation(self, a, b):
        return torch.einsum("nchw, ncij -> nhwij", a, b)
    
    def compute_loss(self, fd, sd, shift):
        return - (sd * (fd - shift)).mean()
        
    def forward(self, feat_A, feat_B, mask_onehot_A, mask_onehot_B):
        total_loss = 0.0
        for i in range(len(self.layers)):
            # Compute correlation between features
            feat_A[i] = F.normalize(feat_A[i], dim=1)
            feat_B[i] = F.normalize(feat_B[i], dim=1)
            
            fd_identity = self.tensor_correlation(feat_A[i], feat_A[i]) 
            fd_cross = self.tensor_correlation(feat_A[i], feat_B[i])     

            # Spatial centering
            fd_identity = fd_identity - fd_identity.mean(dim=[3,4], keepdim=True)
            fd_cross    = fd_cross - fd_cross.mean(dim=[3,4], keepdim=True)
            
            
            seg_mask_A = F.interpolate(mask_onehot_A.float(), size=feat_A[i].shape[2:], mode='nearest')  # [B, num_classes, H_feat, W_feat]
            seg_mask_B = F.interpolate(mask_onehot_B.float(), size=feat_B[i].shape[2:], mode='nearest')  # [B, num_classes, H_feat, W_feat]

            sd_identity = self.tensor_correlation(seg_mask_A, seg_mask_A)
            sd_cross = self.tensor_correlation(seg_mask_A, seg_mask_B)

            identity_loss = self.compute_loss(fd_identity, sd_identity, shift=0.18)
            cross_loss = self.compute_loss(fd_cross, sd_cross, shift=0.18)
            total_loss += (identity_loss + cross_loss) / 2.0
        

        return total_loss / len(self.layers)

            