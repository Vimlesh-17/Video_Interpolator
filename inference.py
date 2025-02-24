# --------------- inference.py ---------------
import cv2
import torch
import numpy as np
from torchvision import transforms
from .config import DEVICE
from .models import RIFE, LDMVFI

class VideoInterpolator:
    def __init__(self, model_type='rife'):
        self.model_type = model_type.lower()
        self.device = DEVICE
        self.model = self._load_model()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    def _load_model(self):
        model_map = {
            'rife': RIFE,
            'ldmvfi': LDMVFI
        }
        model = model_map[self.model_type]()
        model_path = f"..path_to_model_checkpoint/best_{self.model_type}.pth"
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model.to(self.device).eval()

    def process_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (854, 480))
        return self.transform(frame).unsqueeze(0).to(self.device)

    def interpolate(self, frame1, frame2):
        t1 = self.process_frame(frame1)
        t2 = self.process_frame(frame2)
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            output = self.model(t1, t2)
        return self._convert_output(output)

    def _convert_output(self, tensor):
        tensor = (tensor.squeeze().permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1)
        return (tensor.cpu().numpy() * 255).astype(np.uint8)