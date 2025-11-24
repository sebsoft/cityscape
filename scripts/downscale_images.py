"""
Downscale Dataset Images to 1024px

This script downscales all images in the dataset to 1024px (longest side)
to reduce storage size and speed up data loading during training.

Original Cityscapes images: 2048x1024
Downscaled images: 1024x512

Usage:
    python scripts/downscale_images.py --input data_filtered --output data_filtered_1024
"""

import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import shutil


def downscale_image(input_path: Path, output_path: Path, target_size: int = 1024):
    """
    Downscale image to target size (longest side).
    
    Args:
        input_path: Path to input image
        output_path: Path to save downscaled image
        target_size: Target size for longest side (default: 1024)
    """
    # Load image
    img = Image.open(input_path)
    
    # Calculate new size maintaining aspect ratio
    width, height = img.size
    if width > height:
        new_width = target_size
        new_height = int(height * (target_size / width))
    else:
        new_height = target_size
        new_width = int(width * (target_size / height))
    
    # Resize image with high-quality resampling
    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Save image
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img_resized.save(output_path, quality=85, optimize=True)


def copy_labels(input_dir: Path, output_dir: Path):
    """
    Copy label files without modification (bounding boxes are normalized 0-1).
    
    Args:
        input_dir: Input labels directory
        output_dir: Output labels directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    label_files = list(input_dir.glob('*.txt'))
    for label_file in tqdm(label_files, desc=f"Copying labels from {input_dir.name}"):
        output_file = output_dir / label_file.name
        shutil.copy2(label_file, output_file)


def downscale_dataset(input_root: Path, output_root: Path, target_size: int = 1024):
    """
    Downscale entire dataset.
    
    Args:
        input_root: Root directory of input dataset
        output_root: Root directory for output dataset
        target_size: Target size for longest side
    """
    print(f"Downscaling dataset from {input_root} to {output_root}")
    print(f"Target size: {target_size}px (longest side)")
    
    # Process train and validation splits
    for split in ['train', 'valid']:
        print(f"\nProcessing {split} split...")
        
        # Downscale images
        input_images_dir = input_root / split / 'images'
        output_images_dir = output_root / split / 'images'
        
        if not input_images_dir.exists():
            print(f"  Warning: {input_images_dir} not found, skipping")
            continue
        
        image_files = list(input_images_dir.glob('*'))
        image_files = [f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        
        print(f"  Found {len(image_files)} images")
        
        for image_file in tqdm(image_files, desc=f"  Downscaling {split} images"):
            output_file = output_images_dir / image_file.name
            try:
                downscale_image(image_file, output_file, target_size)
            except Exception as e:
                print(f"  Error processing {image_file.name}: {e}")
        
        # Copy labels (normalized coordinates don't need adjustment)
        input_labels_dir = input_root / split / 'labels'
        output_labels_dir = output_root / split / 'labels'
        
        if input_labels_dir.exists():
            copy_labels(input_labels_dir, output_labels_dir)
        else:
            print(f"  Warning: {input_labels_dir} not found, skipping labels")
    
    # Copy data.yaml if exists
    data_yaml = input_root / 'data.yaml'
    if data_yaml.exists():
        output_yaml = output_root / 'data.yaml'
        shutil.copy2(data_yaml, output_yaml)
        print(f"\nCopied {data_yaml} to {output_yaml}")
    
    print("\n✅ Dataset downscaling complete!")
    
    # Show size comparison
    input_size = sum(f.stat().st_size for f in input_root.rglob('*') if f.is_file())
    output_size = sum(f.stat().st_size for f in output_root.rglob('*') if f.is_file())
    
    print(f"\nSize comparison:")
    print(f"  Input:  {input_size / 1024 / 1024:.1f} MB")
    print(f"  Output: {output_size / 1024 / 1024:.1f} MB")
    print(f"  Saved:  {(input_size - output_size) / 1024 / 1024:.1f} MB ({100 * (1 - output_size/input_size):.1f}% reduction)")


def main():
    parser = argparse.ArgumentParser(description='Downscale dataset images')
    parser.add_argument('--input', type=str, default='data_filtered',
                        help='Input dataset directory (default: data_filtered)')
    parser.add_argument('--output', type=str, default='data_filtered_1024',
                        help='Output dataset directory (default: data_filtered_1024)')
    parser.add_argument('--size', type=int, default=1024,
                        help='Target size for longest side (default: 1024)')
    
    args = parser.parse_args()
    
    input_root = Path(args.input)
    output_root = Path(args.output)
    
    if not input_root.exists():
        print(f"❌ Error: Input directory {input_root} does not exist")
        return
    
    if output_root.exists():
        response = input(f"⚠️  Output directory {output_root} already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled")
            return
        print(f"Removing existing {output_root}...")
        shutil.rmtree(output_root)
    
    downscale_dataset(input_root, output_root, args.size)


if __name__ == '__main__':
    main()
