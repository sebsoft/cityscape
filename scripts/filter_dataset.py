"""
Filter Cityscapes Dataset Labels

This script removes unwanted classes from the dataset and remaps class IDs.
Creates a new filtered dataset with only vehicles and people.

Usage:
    python scripts/filter_dataset.py --input INPUT_DIR --output OUTPUT_DIR
"""

import argparse
import shutil
from pathlib import Path
from tqdm import tqdm


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Filter Cityscapes dataset to vehicles and people only')
    parser.add_argument('--input', type=str, required=True, help='Input dataset directory')
    parser.add_argument('--output', type=str, required=True, help='Output filtered dataset directory')
    return parser.parse_args()


# Original dataset paths (will be set from command line)
DATA_DIR = None
TRAIN_LABELS = None
VAL_LABELS = None
TRAIN_IMAGES = None
VAL_IMAGES = None

# New filtered dataset paths (will be set from command line)
FILTERED_DIR = None
FILTERED_TRAIN_LABELS = None
FILTERED_VAL_LABELS = None
FILTERED_TRAIN_IMAGES = None
FILTERED_VAL_IMAGES = None

# Classes to keep (vehicles and people) - DYNAMIC REMOVED
# Original class IDs from the dataset (based on data.yaml)
CLASSES_TO_KEEP = {
    0: 0,   # bicycle -> 0
    2: 1,   # bus -> 1
    3: 2,   # car -> 2
    4: 3,   # caravan -> 3
    8: 4,   # motorcycle -> 4
    9: 5,   # person -> 5
    11: 6,  # rider -> 6
    14: 7,  # trailer -> 7
    15: 8,  # truck -> 8
}

# Class names for reference
CLASS_NAMES = [
    'bicycle',    # 0
    'bus',        # 1
    'car',        # 2
    'caravan',    # 3
    'motorcycle', # 4
    'person',     # 5
    'rider',      # 6
    'trailer',    # 7
    'truck',      # 8
]


def filter_label_file(input_path, output_path):
    """
    Filter a single label file to keep only desired classes.
    Returns True if file has valid labels after filtering, False otherwise.
    """
    with open(input_path, 'r') as f:
        lines = f.readlines()
    
    filtered_lines = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        
        class_id = int(parts[0])
        
        # Check if this class should be kept
        if class_id in CLASSES_TO_KEEP:
            # Remap class ID
            new_class_id = CLASSES_TO_KEEP[class_id]
            # Write line with new class ID
            filtered_lines.append(f"{new_class_id} {' '.join(parts[1:])}\n")
    
    # Only write file if it has labels
    if filtered_lines:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.writelines(filtered_lines)
        return True
    
    return False


def filter_dataset(input_dir, output_dir):
    """Filter the entire dataset."""
    
    # Set global paths
    global DATA_DIR, TRAIN_LABELS, VAL_LABELS, TRAIN_IMAGES, VAL_IMAGES
    global FILTERED_DIR, FILTERED_TRAIN_LABELS, FILTERED_VAL_LABELS
    global FILTERED_TRAIN_IMAGES, FILTERED_VAL_IMAGES
    
    DATA_DIR = Path(input_dir)
    TRAIN_LABELS = DATA_DIR / 'train' / 'labels'
    VAL_LABELS = DATA_DIR / 'valid' / 'labels'
    TRAIN_IMAGES = DATA_DIR / 'train' / 'images'
    VAL_IMAGES = DATA_DIR / 'valid' / 'images'
    
    FILTERED_DIR = Path(output_dir)
    FILTERED_TRAIN_LABELS = FILTERED_DIR / 'train' / 'labels'
    FILTERED_VAL_LABELS = FILTERED_DIR / 'valid' / 'labels'
    FILTERED_TRAIN_IMAGES = FILTERED_DIR / 'train' / 'images'
    FILTERED_VAL_IMAGES = FILTERED_DIR / 'valid' / 'images'
    
    print("\n" + "="*80)
    print("FILTERING CITYSCAPES DATASET")
    print("="*80)
    print(f"\nKeeping {len(CLASSES_TO_KEEP)} classes:")
    for old_id, new_id in CLASSES_TO_KEEP.items():
        print(f"  Class {old_id} -> {new_id} ({CLASS_NAMES[new_id]})")
    print("\n" + "="*80 + "\n")
    
    # Create output directories
    FILTERED_TRAIN_LABELS.mkdir(parents=True, exist_ok=True)
    FILTERED_VAL_LABELS.mkdir(parents=True, exist_ok=True)
    FILTERED_TRAIN_IMAGES.mkdir(parents=True, exist_ok=True)
    FILTERED_VAL_IMAGES.mkdir(parents=True, exist_ok=True)
    
    # Filter training set
    print("Filtering training set...")
    train_label_files = sorted(TRAIN_LABELS.glob('*.txt'))
    train_kept = 0
    
    for label_file in tqdm(train_label_files, desc="Train labels"):
        output_label = FILTERED_TRAIN_LABELS / label_file.name
        
        if filter_label_file(label_file, output_label):
            # Copy corresponding image
            image_name = label_file.stem + '.jpg'
            input_image = TRAIN_IMAGES / image_name
            output_image = FILTERED_TRAIN_IMAGES / image_name
            
            if input_image.exists():
                shutil.copy2(input_image, output_image)
                train_kept += 1
    
    print(f"✓ Kept {train_kept}/{len(train_label_files)} training images")
    
    # Filter validation set
    print("\nFiltering validation set...")
    val_label_files = sorted(VAL_LABELS.glob('*.txt'))
    val_kept = 0
    
    for label_file in tqdm(val_label_files, desc="Val labels"):
        output_label = FILTERED_VAL_LABELS / label_file.name
        
        if filter_label_file(label_file, output_label):
            # Copy corresponding image
            image_name = label_file.stem + '.jpg'
            input_image = VAL_IMAGES / image_name
            output_image = FILTERED_VAL_IMAGES / image_name
            
            if input_image.exists():
                shutil.copy2(input_image, output_image)
                val_kept += 1
    
    print(f"✓ Kept {val_kept}/{len(val_label_files)} validation images")
    
    # Print summary
    print("\n" + "="*80)
    print("FILTERING COMPLETE")
    print("="*80)
    print(f"Output directory: {FILTERED_DIR}")
    print(f"Training images: {train_kept}")
    print(f"Validation images: {val_kept}")
    print(f"Total images: {train_kept + val_kept}")
    print("\nNext step:")
    print(f"  Update DATA_CONFIG in train.py to: 'configs/cityscapes_filtered.yaml'")
    print("="*80 + "\n")


if __name__ == '__main__':
    args = parse_args()
    filter_dataset(args.input, args.output)
