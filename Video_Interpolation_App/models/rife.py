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

class IFBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.PReLU()
        )

    def forward(self, x):
        return self.convs(x)

class RIFE(nn.Module):
    def __init__(self):
        super().__init__()
        self.ifnet = nn.Sequential(
            IFBlock(6),
            IFBlock(64),
            IFBlock(64)
        )
        self.refinenet = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),  # Increased channels
            nn.PReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh()  # Added Tanh activation
        )

    def forward(self, img0, img1):
        x = torch.cat([img0, img1], dim=1)
        x = checkpoint(self.ifnet[0], x)
        for block in self.ifnet[1:]:
            x = block(x)
        return self.refinenet(x)