# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy.random as random

import torch
import torch.nn as nn
import torch.nn.functional as F
#from MinkowskiEngine import SparseTensor

class MinkowskiGRN(nn.Module):
    """ GRN layer for sparse tensors.
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim))
        self.beta = nn.Parameter(torch.zeros(1, dim))

    def forward(self, x):
        cm = x.coordinate_manager
        in_key = x.coordinate_map_key

        Gx = torch.norm(x.F, p=2, dim=0, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return SparseTensor(
                self.gamma * (x.F * Nx) + self.beta + x.F,
                coordinate_map_key=in_key,
                coordinate_manager=cm)

class MinkowskiDropPath(nn.Module):
    """ Drop Path for sparse tensors.
    """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(MinkowskiDropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
    
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        cm = x.coordinate_manager
        in_key = x.coordinate_map_key
        keep_prob = 1 - self.drop_prob
        mask = torch.cat([
            torch.ones(len(_)) if random.uniform(0, 1) > self.drop_prob
            else torch.zeros(len(_)) for _ in x.decomposed_coordinates
        ]).view(-1, 1).to(x.device)
        if keep_prob > 0.0 and self.scale_by_keep:
            mask.div_(keep_prob)
        return SparseTensor(
                x.F * mask,
                coordinate_map_key=in_key,
                coordinate_manager=cm)

class MinkowskiLayerNorm(nn.Module):
    """ Channel-wise layer normalization for sparse tensors.
    """

    def __init__(
        self,
        normalized_shape,
        eps=1e-6,
    ):
        super(MinkowskiLayerNorm, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape, eps=eps)
    def forward(self, input):
        output = self.ln(input.F)
        return SparseTensor(
            output,
            coordinate_map_key=input.coordinate_map_key,
            coordinate_manager=input.coordinate_manager)
            
class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
        

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class FeatureWiseAffineNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(FeatureWiseAffineNorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, 1, num_features))  # Scaling parameter
        self.beta = nn.Parameter(torch.zeros(1, 1, num_features))  # Shifting parameter

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)  # Compute mean along the feature axis
        std = x.std(dim=1, keepdim=True) + self.eps  # Compute std along the feature axis
        x_norm = (x - mean) / std  # Normalize
        return x_norm * self.gamma + self.beta  # Apply affine transformation



class SwitchableNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(SwitchableNorm, self).__init__()
        self.eps = eps
        self.momentum = momentum

        # Batch Normalization
        self.bn = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum, affine=True)

        # Layer Normalization (will be applied on permuted input)
        self.ln = nn.LayerNorm(num_features, eps=eps, elementwise_affine=False)

        # Learnable weights for BN and LN
        self.alpha = nn.Parameter(torch.Tensor(2))  # Two learnable parameters for interpolation
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1))  # Scaling
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1))  # Shifting

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize the learnable weights."""
        nn.init.constant_(self.alpha, 0.5)  # Start with equal contribution from BN and LN

    def forward(self, x):
        batch_size, num_features, seq_length = x.shape  # Expected input shape: [batch_size, num_features, seq_length]

        # Compute BN (Batch Normalization)
        x_bn = self.bn(x)

        # Compute LN (Layer Normalization)
        x_permuted = x.permute(0, 2, 1)  # Swap to [batch_size, seq_length, num_features] for LN
        x_ln = self.ln(x_permuted)
        x_ln = x_ln.permute(0, 2, 1)  # Swap back to [batch_size, num_features, seq_length]

        # Compute softmax weights
        alpha = F.softmax(self.alpha, dim=0)  # Ensure weights sum to 1
        w1, w2 = alpha[0], alpha[1]

        # Compute Switchable Normalization
        x_sn = w1 * x_bn + w2 * x_ln  # Weighted sum of BN and LN

        # Apply affine transformation
        return x_sn * self.gamma + self.beta

'''
class FourierFeatureMapping(nn.Module):
    def __init__(self, input_dim, mapping_size=2, scale=10.0):
        super(FourierFeatureMapping, self).__init__()
        self.B = torch.randn((input_dim, mapping_size)) * scale  # Random projection matrix

    def forward(self, x):
        x_proj = 2 * torch.pi * x @ self.B.to(x.device)  # Linear transformation
        #return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)  # Concatenate sine and cosine
        #return torch.sin(x_proj)  # Only return sine OR use torch.cos(x_proj)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)[:, :, :2]  # Ensure final dim=2
'''

class FourierFeatureMapping(nn.Module):
    def __init__(self, input_channels, mapping_size=120, scale=10.0):
        super().__init__()
        self.mapping_size = mapping_size
        self.scale = scale
        self.input_channels = input_channels  # Store for use in forward

    def forward(self, x):
        batch_size, length, channel = x.shape

        # Sample B from an isotropic Gaussian distribution on-the-fly
        B = torch.randn(self.input_channels, self.mapping_size, device=x.device) * self.scale

        # Project input: (batch, length, channel) @ (channel, mapping_size) → (batch, length, mapping_size)
        x_proj = (2 * torch.pi * x) @ B

        # Apply sin and cos, then concatenate: (batch, length, 2 * mapping_size)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        #return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)[:, :, :2]  # Ensure final dim=2



class GaussianFourierFeatureMapping(nn.Module):
    def __init__(self, input_channels, mapping_size=120, scale=10.0):
        super().__init__()
        self.mapping_size = mapping_size
        self.scale = scale
        self.input_channels = input_channels
        
        # Precompute the projection matrix B from a normal distribution
        self.register_buffer("B", torch.randn(input_channels, mapping_size) * scale)

    def forward(self, x):
        batch_size, length, channel = x.shape  # Expecting (batch, length, channels)
        
        # Project input: (batch, length, channel) @ (channel, mapping_size) -> (batch, length, mapping_size)
        x_proj = (2 * torch.pi * x) @ self.B
        
        # Apply sin and cos, then concatenate: (batch, length, 2 * mapping_size)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

'''
import torch
import torch.nn as nn
import pywt

class WaveletTransformFeatures(nn.Module):
    def __init__(self, input_channels, wavelet='haar', level=1):
        super().__init__()
        self.input_channels = input_channels
        self.wavelet = wavelet
        self.level = level

    def forward(self, x):
        batch_size, length, channel = x.shape
        wavelet_coeffs = []
        
        for i in range(channel):
            # Convert tensor to numpy for wavelet processing
            signal = x[:, :, i].cpu().numpy()
            coeffs = pywt.wavedec(signal, self.wavelet, level=self.level, axis=1)

            # Flatten and stack coefficients
            coeffs = [torch.tensor(c).to(x.device) for c in coeffs]
            wavelet_coeffs.append(torch.cat(coeffs, dim=-1))

        return torch.stack(wavelet_coeffs, dim=-1)  # Shape: (batch_size, length, new_features)
'''
