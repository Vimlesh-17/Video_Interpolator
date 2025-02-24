import logging
import subprocess
import os
import gc
import time
import psutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

class CropLayer(nn.Module):
    def __init__(self, crop):
        super().__init__()
        self.crop = crop

    def forward(self, x):
        if self.crop[1] == 0:
            return x[..., :, self.crop[0]:]
        return x[..., :, self.crop[0]:-self.crop[1]]
    
class VQFIGAN(nn.Module):
    def __init__(self, in_channels=3, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=(1,1)),
            nn.MaxPool2d(2, ceil_mode=True),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, latent_dim, 3, padding=1),
            nn.MaxPool2d(2, ceil_mode=True),
            nn.GroupNorm(8, latent_dim),
            nn.SiLU()
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(latent_dim, 128, 3, padding=1),  # Increased channels
            nn.GroupNorm(16, 128),  # Adjusted group norm
            nn.SiLU(),
            nn.Upsample(size=(480, 856), mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh(),  # Added Tanh activation
            CropLayer((0, 2))
        )

class LDMVFI(nn.Module):
    def __init__(self, num_timesteps=50):
        super().__init__()
        self.vqfigan = VQFIGAN()
        self.denoiser = nn.Sequential(
            nn.Conv2d(192, 256, 3, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.Conv2d(256, 64, 3, padding=1)
        )
        self.num_timesteps = num_timesteps

    def forward(self, img0, img1):
        z0 = self.vqfigan.encoder(img0)
        z1 = self.vqfigan.encoder(img1)
        z = torch.randn_like(z0)
        for _ in range(self.num_timesteps):
            z = self.denoiser(torch.cat([z, z0, z1], dim=1))
        return self.vqfigan.decoder(z)