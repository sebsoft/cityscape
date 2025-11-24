"""
Filter Cityscapes Dataset Labels

This script removes unwanted classes from the dataset and remaps class IDs.
Creates a new filtered dataset with only vehicles and people.

Usage:
    python scripts/filter_dataset.py
"""

import shutil
from pathlib import Path
from tqdm import tqdm

# Original dataset paths
DATA_DIR = Path('/Users/sebsoft/projects/cityscape/data')
TRAIN_LABELS = DATA_DIR / 'train' / 'labels'
VAL_LABELS = DATA_DIR / 'valid' / 'labels'
TRAIN_IMAGES = DATA_DIR / 'train' / 'images'
VAL_IMAGES = DATA_DIR / 'valid' / 'images'

# New filtered dataset paths
FILTERED_DIR = DATA_DIR.parent / 'data_filtered'
FILTERED_TRAIN_LABELS = FILTERED_DIR / 'train' / 'labels'
FILTERED_VAL_LABELS = FILTERED_DIR / 'valid' / 'labels'
FILTERED_TRAIN_IMAGES = FILTERED_DIR / 'train' / 'images'
FILTERED_VAL_IMAGES = FILTERED_DIR / 'valid' / 'images'

# Classes to keep (vehicles and people)
# Original class IDs from the dataset
CLASSES_TO_KEEP = {
    0: 0,   # bicycle -> 0
    2: 1,   # bus -> 1
    3: 2,   # car -> 2
    4: 3,   # caravan -> 3
    5: 4,   # dynamic -> 4
    8: 5,   # motorcycle -> 5
    9: 6,   # person -> 6
    11: 7,  # rider -> 7
    14: 8,  # trailer -> 8
    15: 9,  # truck -> 9
}

# Class names for reference
CLASS_NAMES = [
    'bicycle',    # 0
    'bus',        # 1
    'car',        # 2
    'caravan',    # 3
    'dynamic',    # 4
    'motorcycle', # 5
    'person',     # 6
    'rider',      # 7
    'trailer',    # 8
    'truck',      # 9
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


def filter_dataset():
    """Filter the entire dataset."""
    
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
    filter_dataset()
