# Video Frame Generator

## 🚀 Next-Gen Video Interpolation System

### Features
- Dual-model architecture (RIFE + LDMVFI)
- Automated training pipeline
- Frame interpolation API
- Loss visualization
- Configurable parameters

### 📦 Installation
```bash
git clone https://github.com/Vimlesh-17/Video_Interpolator
cd Video_Interpolator
pip install -r requirements.txt
```

### 🛠 Usage
```python
from training import TrainingSystem
from inference import VideoInterpolator

# Initialize systems
trainer = TrainingSystem()
interpolator = VideoInterpolator()

# Train models
trainer.train_model('rife')
trainer.train_model('ldmvfi')

# Process video
interpolator.process_video(
    input_path='input.mp4',
    output_path='output_interpolated.mp4'
)
```

### 📂 Project Structure
```
Video_Frame_generator/
├── Video_Interpolation_App/    # GUI application
├── config.py                   # Configuration settings
├── data_loader.py              # Dataset handling
├── model.py                    # Model architectures
├── training.py                 # Training pipelines
├── inference.py                # Frame interpolation logic
└── main.py                     # Entry point
```

### ⚙ Configuration (config.py)
```python
{
    "batch_size": 8,
    "learning_rate": 1e-4,
    "num_epochs": 50,
    "checkpoint_dir": "checkpoints/",
    "dataset_path": "data/"
}
```

## 📈 Performance Monitoring
```bash
# Generate training curves
python visualize.py --model rife --metric val_loss
```

## 🤝 Contributing
1. Fork the repository
2. Create feature branch
3. Submit PR with detailed description

## 📄 License
MIT License
