"""
Resume YOLO Training Script for Cityscapes Dataset

This script resumes training from a checkpoint (last.pt or best.pt).

Usage:
    python scripts/resume_training.py
"""

from pathlib import Path
from ultralytics import YOLO
import torch


# ============================================================================
# RESUME TRAINING CONFIGURATION
# ============================================================================

# Checkpoint to resume from
CHECKPOINT = 'runs/train/cityscapes_yolo11n/weights/last.pt'  # Path to checkpoint file

# Training parameters (optional - will use saved values from checkpoint if not specified)
ADDITIONAL_EPOCHS = 50            # Additional epochs to train (None to continue to original EPOCHS)
BATCH_SIZE = 8                    # Batch size (None to use checkpoint value)
IMAGE_SIZE = None                 # Image size (None to use checkpoint value)
WORKERS = 8                       # Number of data loading workers

# Other options
DEVICE = 'cpu'                    # Device: 'mps' for Apple Silicon, '0' for GPU, 'cpu' for CPU
PATIENCE = 50                     # Epochs to wait for early stopping

# ============================================================================


def main():
    
    # Print configuration
    print("\n" + "="*80)
    print("RESUME YOLO11 TRAINING ON CITYSCAPES DATASET")
    print("="*80)
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Additional epochs: {ADDITIONAL_EPOCHS if ADDITIONAL_EPOCHS else 'Continue to completion'}")
    print(f"Batch size: {BATCH_SIZE if BATCH_SIZE else 'Use checkpoint value'}")
    print(f"Device: {DEVICE}")
    print("="*80 + "\n")
    
    # Check if checkpoint exists
    checkpoint_path = Path(CHECKPOINT)
    if not checkpoint_path.exists():
        print(f"❌ Error: Checkpoint not found at {CHECKPOINT}")
        print("\nAvailable checkpoints:")
        
        # Look for checkpoints in runs directory
        runs_dir = Path('runs/train')
        if runs_dir.exists():
            checkpoints = list(runs_dir.glob('*/weights/last.pt'))
            checkpoints += list(runs_dir.glob('*/weights/best.pt'))
            
            if checkpoints:
                for cp in checkpoints:
                    print(f"  - {cp}")
            else:
                print("  No checkpoints found. Train a model first with: python scripts/train.py")
        else:
            print("  No training runs found. Train a model first with: python scripts/train.py")
        
        return
    
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
    
    # Load model from checkpoint
    print(f"Loading checkpoint: {CHECKPOINT}")
    model = YOLO(CHECKPOINT)
    
    # Build training arguments
    train_args = {
        'resume': True,
        'device': DEVICE,
        'workers': WORKERS,
        'patience': PATIENCE,
        'verbose': True,
    }
    
    # Add optional parameters if specified
    if ADDITIONAL_EPOCHS:
        # Get current epoch from checkpoint and add additional epochs
        train_args['epochs'] = model.ckpt['epoch'] + ADDITIONAL_EPOCHS + 1
        print(f"Training from epoch {model.ckpt['epoch']} to epoch {train_args['epochs']}")
    
    if BATCH_SIZE:
        train_args['batch'] = BATCH_SIZE
    
    if IMAGE_SIZE:
        train_args['imgsz'] = IMAGE_SIZE
    
    print("\nResuming training...\n")
    
    # Resume training
    results = model.train(**train_args)
    
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
