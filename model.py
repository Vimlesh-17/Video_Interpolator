# --------------- models.py ---------------
import torch
import torch.nn as nn

class CropLayer(nn.Module):
    def __init__(self, crop):
        super().__init__()
        self.crop = crop

    def forward(self, x):
        if self.crop[1] == 0:
            return x[..., :, self.crop[0]:]
        return x[..., :, self.crop[0]:-self.crop[1]]

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
            nn.Conv2d(64, 128, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, img0, img1):
        x = torch.cat([img0, img1], dim=1)
        x = checkpoint(self.ifnet[0], x)
        for block in self.ifnet[1:]:
            x = block(x)
        return self.refinenet(x)

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
            nn.Conv2d(latent_dim, 128, 3, padding=1),
            nn.GroupNorm(16, 128),
            nn.SiLU(),
            nn.Upsample(size=(480, 856), mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh(),
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