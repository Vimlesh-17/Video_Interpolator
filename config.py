# --------------- config.py ---------------
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
CONFIG = {
    "dataset_path": "..path_to_training_data..",
    "checkpoint_dir": "..path_to_model_checkpoints..",
    "models": {
        "rife": {"batch_size": 24, "epochs": 40},
        "ldmvfi": {"batch_size": 6, "epochs": 10}
    },
    "memory": {
        "max_ram_utilization": 0.85,
        "pin_memory": True,
        "num_workers": 4,
        "prefetch_factor": 2
    },
    "img_size": (480, 854),
    "padding": {
        "mode": "replicate",
        "padding": (0, 0, 0, 2)
    }
}

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)