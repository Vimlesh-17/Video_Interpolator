# --------------- data_loader.py ---------------
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class FrameInterpolationDataset(Dataset):
    def __init__(self, root_dir):
        self.root = root_dir
        self.samples = []
        for subdir in os.listdir(root_dir):
            subdir_path = os.path.join(root_dir, subdir)
            if os.path.isdir(subdir_path):
                images = sorted([f for f in os.listdir(subdir_path) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
                for i in range(len(images)-2):
                    self.samples.append((
                        os.path.join(subdir_path, images[i]),
                        os.path.join(subdir_path, images[i+2]),
                        os.path.join(subdir_path, images[i+1])
                    ))
        self.transform = transforms.Compose([
            transforms.Resize(CONFIG["img_size"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        paths = self.samples[idx]
        img0 = Image.open(paths[0]).convert("RGB")
        img1 = Image.open(paths[1]).convert("RGB")
        target = Image.open(paths[2]).convert("RGB")
        return (
            self.transform(img0),
            self.transform(img1),
            self.transform(target)
        )

def create_dataloader(dataset, batch_size):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=CONFIG["memory"]["num_workers"],
        pin_memory=CONFIG["memory"]["pin_memory"],
        prefetch_factor=CONFIG["memory"]["prefetch_factor"],
        persistent_workers=True
    )