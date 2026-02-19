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
        return - (sd.clamp(min=0) * (fd - shift)).mean()
    
    def sample(self, t: torch.Tensor, coords: torch.Tensor):
        return F.grid_sample(t, coords.permute(0, 2, 1, 3), padding_mode='border', align_corners=True)
        
    def forward(self, feat_A, feat_B, mask_onehot_A, mask_onehot_B):
        total_loss = 0.0

        ## for all layers
        self.layers = [i for i in range(len(feat_A))]
        for i in range(len(self.layers)):
            # Compute correlation between features

            seg_mask_A = F.interpolate(mask_onehot_A.float(), size=feat_A[i].shape[2:], mode='nearest')
            seg_mask_B = F.interpolate(mask_onehot_B.float(), size=feat_B[i].shape[2:], mode='nearest')

            feature_shape = min(feat_A[i].shape[2], 110)
            coord_shape = [feat_A[i].shape[0], feature_shape, feature_shape, 2]
            coords1 = torch.rand(coord_shape, device=feat_A[i].device) * 2 - 1
            coords2 = torch.rand(coord_shape, device=feat_A[i].device) * 2 - 1

            feature_sample_A = self.sample(feat_A[i], coords1) 
            seg_sample_A = self.sample(seg_mask_A, coords1)
            feature_sample_B = self.sample(feat_B[i], coords2)
            seg_sample_B = self.sample(seg_mask_B, coords2)

            feature_sample_A = F.normalize(feature_sample_A, dim=1)
            feature_sample_B = F.normalize(feature_sample_B, dim=1)

            fd_identity = self.tensor_correlation(feature_sample_A, feature_sample_A) 
            fd_cross = self.tensor_correlation(feature_sample_A, feature_sample_B)     

            # Spatial centering
            fd_identity = fd_identity - fd_identity.mean(dim=[3,4], keepdim=True)
            fd_cross    = fd_cross - fd_cross.mean(dim=[3,4], keepdim=True)
            

            sd_identity = self.tensor_correlation(seg_sample_A, seg_sample_A)
            sd_cross = self.tensor_correlation(seg_sample_A, seg_sample_B)

            identity_loss = self.compute_loss(fd_identity, sd_identity, shift=0.18)
            cross_loss = self.compute_loss(fd_cross, sd_cross, shift=0.18)
            total_loss += (identity_loss + cross_loss) / 2.0
        
        return total_loss / len(self.layers)

            