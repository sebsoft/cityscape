"""
YOLO Training Script for Cityscapes Dataset

This script trains a YOLO11s model on the Cityscapes dataset for urban scene understanding.

Usage:
    python scripts/train.py
"""

from pathlib import Path
from ultralytics import YOLO
import torch


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# Model parameters
MODEL = 'yolo11s.pt'              # Model variant (yolo11n/s/m/l/x)
RESUME = None                     # Resume from checkpoint path (None for new training)

# Training parameters
EPOCHS = 100                      # Number of training epochs
BATCH_SIZE = 8                    # Batch size for training
IMAGE_SIZE = 1024                 # Input image size (pixels) - higher resolution for better small object detection
WORKERS = 8                       # Number of data loading workers

# Optimizer parameters
LEARNING_RATE_INITIAL = 0.01      # Initial learning rate
LEARNING_RATE_FINAL = 0.01        # Final learning rate (lr0 * lrf)
MOMENTUM = 0.937                  # SGD momentum/Adam beta1
WEIGHT_DECAY = 0.0005             # Optimizer weight decay

# Augmentation parameters
HSV_H = 0.015                     # HSV-Hue augmentation (fraction)
HSV_S = 0.7                       # HSV-Saturation augmentation (fraction)
HSV_V = 0.4                       # HSV-Value augmentation (fraction)
DEGREES = 0.0                     # Rotation augmentation (degrees)
TRANSLATE = 0.1                   # Translation augmentation (fraction)
SCALE = 0.5                       # Scale augmentation (fraction)
FLIP_UD = 0.0                     # Flip up-down augmentation probability
FLIP_LR = 0.5                     # Flip left-right augmentation probability
MOSAIC = 1.0                      # Mosaic augmentation probability

# Dataset and output
DATA_CONFIG = 'configs/cityscapes_filtered_1024.yaml'  # Path to dataset YAML file
PROJECT_DIR = 'runs/train'        # Project directory for outputs
EXPERIMENT_NAME = 'cityscapes_yolo11s_1024'  # Experiment name

# Other options
DEVICE = 'cpu'                    # Device: 'cpu' for CPU (stable), 'mps' for Apple Silicon (experimental), '0' for GPU
PATIENCE = 50                     # Epochs to wait for early stopping
SAVE_PERIOD = -1                  # Save checkpoint every x epochs (-1 = disabled)
CACHE = False                     # Cache images for faster training
PRETRAINED = True                 # Use pretrained model

# ============================================================================


def main():
    
    # Print configuration
    print("\n" + "="*80)
    print("YOLO11 TRAINING ON CITYSCAPES DATASET")
    print("="*80)
    print(f"Model: {MODEL}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Device: {DEVICE}")
    print(f"Dataset config: {DATA_CONFIG}")
    print(f"Output: {PROJECT_DIR}/{EXPERIMENT_NAME}")
    print("="*80 + "\n")
    
    # Check for GPU/MPS
    if torch.cuda.is_available():
        print(f"✓ CUDA is available - GPU training enabled")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
    elif torch.backends.mps.is_available():
        print(f"✓ Apple Silicon detected - MPS (Metal) acceleration enabled")
        print(f"  Using Apple M4 Neural Engine for training")
        print(f"  Tip: You can use larger batch sizes on Apple Silicon!\n")
    else:
        print("⚠ No GPU acceleration available - training will use CPU (slower)\n")
    
    # Load or create model
    if RESUME:
        print(f"Resuming training from: {RESUME}")
        model = YOLO(RESUME)
    else:
        print(f"Loading model: {MODEL}")
        model = YOLO(MODEL)
    
    # Train the model
    results = model.train(
        data=DATA_CONFIG,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMAGE_SIZE,
        workers=WORKERS,
        device=DEVICE,
        
        # Optimizer
        lr0=LEARNING_RATE_INITIAL,
        lrf=LEARNING_RATE_FINAL,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        
        # Augmentation
        hsv_h=HSV_H,
        hsv_s=HSV_S,
        hsv_v=HSV_V,
        degrees=DEGREES,
        translate=TRANSLATE,
        scale=SCALE,
        flipud=FLIP_UD,
        fliplr=FLIP_LR,
        mosaic=MOSAIC,
        
        # Output and saving
        project=PROJECT_DIR,
        name=EXPERIMENT_NAME,
        save_period=SAVE_PERIOD,
        patience=PATIENCE,
        
        # Other
        cache=CACHE,
        pretrained=PRETRAINED,
        verbose=True,
    )
    
    # Print training summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best model saved to: {model.trainer.best}")
    print(f"Last model saved to: {model.trainer.last}")
    print(f"Results saved to: {model.trainer.save_dir}")
    print("="*80 + "\n")
    
    # Validate the best model
    print("Validating best model on validation set...")
    metrics = model.val()
    
    print(f"\nValidation Results:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall: {metrics.box.mr:.4f}")


if __name__ == '__main__':
    main()
