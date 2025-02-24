# models/utils.py
import os
import logging
import torch
import cv2
import numpy as np
from torchvision import transforms
from typing import Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CropLayer(torch.nn.Module):
    """Custom cropping layer for output adjustment"""
    def __init__(self, crop: Tuple[int, int]):
        super().__init__()
        self.crop = crop

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.crop[1] == 0:
            return x[..., :, self.crop[0]:]
        return x[..., :, self.crop[0]:-self.crop[1]]

def get_device() -> torch.device:
    """Get available compute device"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_weights(model: torch.nn.Module, weight_path: str) -> torch.nn.Module:
    """Load pretrained weights with device compatibility"""
    try:
        device = get_device()
        state_dict = torch.load(weight_path, map_location=device)
        
        # Handle DataParallel wrapping
        if all(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        model = model.to(device).eval()
        logger.info(f"Loaded weights from {weight_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading weights: {str(e)}")
        raise

def model_preprocess() -> transforms.Compose:
    """Create standard preprocessing transform pipeline"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

def prepare_frame(frame: np.ndarray) -> torch.Tensor:
    """Preprocess frame for model input"""
    # Convert BGR to RGB and resize
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_frame = cv2.resize(rgb_frame, (854, 480))  # Width, Height
    
    # Apply transforms and add batch dimension
    transform = model_preprocess()
    return transform(resized_frame).unsqueeze(0)

def tensor_to_frame(tensor: torch.Tensor) -> np.ndarray:
    """Convert model output tensor to numpy frame"""
    # Remove batch dimension and rearrange dimensions
    output = tensor.squeeze().permute(1, 2, 0)
    
    # Denormalize and convert to uint8
    output = (output * 0.5 + 0.5).clamp(0, 1)
    return (output.cpu().numpy() * 255).astype(np.uint8)

def validate_resolution(frame: np.ndarray):
    """Validate input frame resolution"""
    h, w = frame.shape[:2]
    if h != 480 or w != 854:
        raise ValueError(f"Invalid frame resolution {w}x{h}. Must be 854x480")

def memory_cleanup():
    """Clean up GPU memory if available"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Memory cleanup completed")

def setup_amp() -> Tuple[bool, torch.dtype]:
    """Configure mixed precision settings"""
    amp_enabled = torch.cuda.is_available()
    dtype = torch.float16 if amp_enabled else torch.float32
    return amp_enabled, dtype

def log_system_status():
    """Log hardware/software configuration"""
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f}GB")

def timed(func):
    """Decorator for execution timing"""
    def wrapper(*args, **kwargs):
        start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if start: start.record()
        result = func(*args, **kwargs)
        if end: end.record()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            logger.info(f"{func.__name__} executed in {start.elapsed_time(end):.2f}ms")
        return result
    return wrapper