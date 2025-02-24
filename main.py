# --------------- main.py ---------------
from .training import TrainingSystem
from .inference import VideoInterpolator
from .config import CONFIG
import matplotlib.pyplot as plt

def main():
    # Initialize systems
    trainer = TrainingSystem()
    interpolator = VideoInterpolator()
    
    # Train models
    rife_history = trainer.train_model("rife")
    ldmvfi_history = trainer.train_model("ldmvfi")
    
    # Generate visualization
    plt.figure(figsize=(20, 15))
    plt.plot([h["val_loss"] for h in rife_history], label="RIFE")
    plt.plot([h["val_loss"] for h in ldmvfi_history], label="LDMVFI")
    plt.title("Validation Loss Comparison")
    plt.legend()
    plt.savefig(os.path.join(CONFIG["checkpoint_dir"], "loss_comparison.png"))
    
    # Process sample video
    interpolator.process_video(
        input_path="..Path_to_input_video",
        output_path="..path_to_save_the_interpolated_video"
    )

if __name__ == "__main__":
    main()