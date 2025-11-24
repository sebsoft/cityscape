"""
Cityscapes Dataset Preparation Script for YOLO Training

This script downloads and converts the Cityscapes dataset to YOLO format.
Cityscapes contains urban street scenes with various classes like cars, people, roads, etc.

Usage:
    python scripts/prepare_dataset.py --data-dir ./data --download
"""

import os
import json
import shutil
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm


# Cityscapes class mapping (30 classes -> reduced to common detection classes)
CITYSCAPES_CLASSES = {
    'person': 24,
    'rider': 25,
    'car': 26,
    'truck': 27,
    'bus': 28,
    'train': 31,
    'motorcycle': 32,
    'bicycle': 33,
}

# YOLO class names (simplified for object detection)
YOLO_CLASSES = [
    'person',
    'rider', 
    'car',
    'truck',
    'bus',
    'train',
    'motorcycle',
    'bicycle'
]


def convert_polygon_to_bbox(polygon):
    """Convert polygon annotation to bounding box [x_min, y_min, x_max, y_max]"""
    x_coords = [point[0] for point in polygon]
    y_coords = [point[1] for point in polygon]
    
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)
    
    return [x_min, y_min, x_max, y_max]


def bbox_to_yolo_format(bbox, img_width, img_height):
    """
    Convert bounding box to YOLO format
    bbox: [x_min, y_min, x_max, y_max]
    returns: [x_center, y_center, width, height] normalized to [0, 1]
    """
    x_min, y_min, x_max, y_max = bbox
    
    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    
    return [x_center, y_center, width, height]


def convert_cityscapes_to_yolo(cityscapes_root, output_dir, split='train'):
    """
    Convert Cityscapes annotations to YOLO format
    
    Args:
        cityscapes_root: Path to Cityscapes dataset root
        output_dir: Path to output directory for YOLO format
        split: Dataset split ('train', 'val', 'test')
    """
    print(f"\nConverting Cityscapes {split} split to YOLO format...")
    
    # Paths
    img_dir = Path(cityscapes_root) / 'leftImg8bit' / split
    ann_dir = Path(cityscapes_root) / 'gtFine' / split
    
    output_img_dir = Path(output_dir) / 'images' / split
    output_label_dir = Path(output_dir) / 'labels' / split
    
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)
    
    if not img_dir.exists():
        print(f"Warning: {img_dir} does not exist. Skipping {split} split.")
        return
    
    # Process each city
    cities = sorted([d for d in img_dir.iterdir() if d.is_dir()])
    
    for city_dir in tqdm(cities, desc=f"Processing {split} cities"):
        city_name = city_dir.name
        ann_city_dir = ann_dir / city_name
        
        if not ann_city_dir.exists():
            continue
        
        # Process each image
        for img_path in sorted(city_dir.glob('*.png')):
            # Find corresponding annotation file (polygon format)
            base_name = img_path.stem.replace('_leftImg8bit', '')
            ann_path = ann_city_dir / f"{base_name}_gtFine_polygons.json"
            
            if not ann_path.exists():
                continue
            
            # Load annotation
            with open(ann_path, 'r') as f:
                ann_data = json.load(f)
            
            img_width = ann_data['imgWidth']
            img_height = ann_data['imgHeight']
            
            # Convert annotations to YOLO format
            yolo_annotations = []
            
            for obj in ann_data['objects']:
                label = obj['label']
                
                # Check if this is a class we want to detect
                if label not in CITYSCAPES_CLASSES:
                    continue
                
                class_id = YOLO_CLASSES.index(label)
                
                # Get polygon and convert to bbox
                polygon = obj['polygon']
                bbox = convert_polygon_to_bbox(polygon)
                
                # Convert to YOLO format
                yolo_bbox = bbox_to_yolo_format(bbox, img_width, img_height)
                
                # Validate bbox (must be within [0, 1] and have positive width/height)
                if all(0 <= coord <= 1 for coord in yolo_bbox) and yolo_bbox[2] > 0 and yolo_bbox[3] > 0:
                    yolo_annotations.append([class_id] + yolo_bbox)
            
            # Skip images with no valid annotations
            if not yolo_annotations:
                continue
            
            # Copy image to output directory
            output_img_path = output_img_dir / img_path.name
            shutil.copy2(img_path, output_img_path)
            
            # Write YOLO label file
            output_label_path = output_label_dir / f"{img_path.stem}.txt"
            with open(output_label_path, 'w') as f:
                for ann in yolo_annotations:
                    f.write(f"{ann[0]} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}\n")
    
    print(f"Completed {split} split conversion!")


def main():
    parser = argparse.ArgumentParser(description='Prepare Cityscapes dataset for YOLO training')
    parser.add_argument('--data-dir', type=str, default='./data', help='Root directory for datasets')
    parser.add_argument('--cityscapes-dir', type=str, default=None, 
                        help='Path to existing Cityscapes dataset (if already downloaded)')
    parser.add_argument('--download', action='store_true', 
                        help='Download Cityscapes dataset (requires credentials)')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine Cityscapes source directory
    if args.cityscapes_dir:
        cityscapes_root = Path(args.cityscapes_dir)
    else:
        cityscapes_root = data_dir / 'cityscapes_raw'
    
    if args.download:
        print("\n" + "="*80)
        print("CITYSCAPES DATASET DOWNLOAD")
        print("="*80)
        print("\nNote: Cityscapes dataset requires registration and manual download.")
        print("Please visit: https://www.cityscapes-dataset.com/")
        print("\nYou need to download:")
        print("  1. leftImg8bit_trainvaltest.zip (11GB)")
        print("  2. gtFine_trainvaltest.zip (241MB)")
        print(f"\nExtract both archives to: {cityscapes_root}")
        print("="*80 + "\n")
        
        # Create directory structure
        cityscapes_root.mkdir(parents=True, exist_ok=True)
        return
    
    # Check if Cityscapes dataset exists
    if not cityscapes_root.exists():
        print(f"\nError: Cityscapes dataset not found at {cityscapes_root}")
        print("Please use --download flag for instructions or specify --cityscapes-dir")
        return
    
    # Output directory for YOLO format
    yolo_output_dir = data_dir / 'cityscapes_yolo'
    yolo_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert each split
    for split in ['train', 'val']:
        convert_cityscapes_to_yolo(cityscapes_root, yolo_output_dir, split)
    
    print(f"\n✓ Dataset preparation complete!")
    print(f"YOLO format dataset saved to: {yolo_output_dir}")


if __name__ == '__main__':
    main()
