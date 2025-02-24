# --------------- training.py ---------------
import torch
import logging
import gc
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from .config import CONFIG, DEVICE
from .models import RIFE, LDMVFI
from .data_loader import create_dataloader

class TrainingSystem:
    def __init__(self):
        self.models = {
            "rife": RIFE().to(DEVICE),
            "ldmvfi": LDMVFI().to(DEVICE)
        }
        
    def clear_vram(self):
        torch.cuda.empty_cache()
        gc.collect()
        logging.info("VRAM cleared")

    def initialize_training(self, model_type):
        cfg = CONFIG["models"][model_type]
        model = self.models[model_type]
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scaler = GradScaler()
        return model, optimizer, scaler, cfg

    def load_checkpoint(self, model_type):
        checkpoint_path = os.path.join(CONFIG["checkpoint_dir"], f"{model_type}_checkpoint.pth")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            logging.info(f"Resuming {model_type} training from epoch {checkpoint['epoch']+1}")
            return checkpoint
        return None

    def train_epoch(self, model, train_loader, optimizer, scaler, criterion):
        model.train()
        train_loss = 0.0
        for img0, img1, target in tqdm(train_loader, desc="Training", leave=False):
            img0 = img0.to(DEVICE, non_blocking=True)
            img1 = img1.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type='cuda', dtype=torch.float16):
                pred = model(img0, img1)
                loss = criterion(pred, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        return train_loss / len(train_loader)

    def validate(self, model, val_loader, criterion):
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img0, img1, target in val_loader:
                img0 = img0.to(DEVICE)
                img1 = img1.to(DEVICE)
                target = target.to(DEVICE)
                pred = model(img0, img1)
                val_loss += criterion(pred, target).item()
        return val_loss / len(val_loader)

    def train_model(self, model_type):
        self.clear_vram()
        model, optimizer, scaler, cfg = self.initialize_training(model_type)
        checkpoint = self.load_checkpoint(model_type)
        
        # Restore checkpoint if exists
        start_epoch = 0
        history = []
        best_loss = float('inf')
        if checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            history = checkpoint['history']
            best_loss = checkpoint['best_loss']

        # Initialize dataset and metrics
        dataset = FrameInterpolationDataset(CONFIG["dataset_path"])
        train_size = int(0.9 * len(dataset))
        train_set, val_set = torch.utils.data.random_split(dataset, [train_size, len(dataset)-train_size])
        train_loader = create_dataloader(train_set, cfg["batch_size"])
        val_loader = create_dataloader(val_set, cfg["batch_size"])
        criterion = nn.L1Loss()

        # Training loop
        for epoch in range(start_epoch, cfg["epochs"]):
            start_time = time.time()
            
            # Train and validate
            train_loss = self.train_epoch(model, train_loader, optimizer, scaler, criterion)
            val_loss = self.validate(model, val_loader, criterion)

            # Update history and save checkpoints
            history.append({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch_time": time.time() - start_time
            })
            
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), 
                         os.path.join(CONFIG["checkpoint_dir"], f"best_{model_type}.pth"))

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'history': history,
                'best_loss': best_loss
            }, os.path.join(CONFIG["checkpoint_dir"], f"{model_type}_checkpoint.pth"))

        # Save final model
        torch.save(model.state_dict(), 
                 os.path.join(CONFIG["checkpoint_dir"], f"final_{model_type}.pth"))
        return history