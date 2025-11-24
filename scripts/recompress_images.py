"""
Recompress JPEG images to reduce file size

This script recompresses existing JPEG images with a lower quality setting
to reduce storage size while maintaining sufficient quality for training.

Usage:
    python scripts/recompress_images.py --input data_dir --quality 85
"""

import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import tempfile
import shutil


def recompress_image(image_path: Path, quality: int = 85):
    """
    Recompress a JPEG image with specified quality.
    
    Args:
        image_path: Path to image file
        quality: JPEG quality (0-100, default: 85)
    """
    # Load image
    img = Image.open(image_path)
    
    # Save to temporary file with new quality
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        img.save(tmp.name, format='JPEG', quality=quality, optimize=True)
        tmp_path = tmp.name
    
    # Replace original with compressed version
    shutil.move(tmp_path, image_path)


def recompress_directory(directory: Path, quality: int = 85):
    """
    Recompress all JPEG images in a directory.
    
    Args:
        directory: Directory containing images
        quality: JPEG quality (0-100)
    """
    # Find all JPEG images
    image_files = list(directory.glob('**/*.jpg')) + list(directory.glob('**/*.jpeg'))
    
    if not image_files:
        print(f"No JPEG images found in {directory}")
        return
    
    # Get initial size
    initial_size = sum(f.stat().st_size for f in image_files)
    
    print(f"Found {len(image_files)} images")
    print(f"Initial size: {initial_size / 1024 / 1024:.1f} MB")
    print(f"Target quality: {quality}")
    
    # Recompress all images
    for image_file in tqdm(image_files, desc="Recompressing"):
        try:
            recompress_image(image_file, quality)
        except Exception as e:
            print(f"Error processing {image_file.name}: {e}")
    
    # Get final size
    final_size = sum(f.stat().st_size for f in image_files)
    
    print(f"\n✅ Recompression complete!")
    print(f"Initial size: {initial_size / 1024 / 1024:.1f} MB")
    print(f"Final size:   {final_size / 1024 / 1024:.1f} MB")
    print(f"Saved:        {(initial_size - final_size) / 1024 / 1024:.1f} MB ({100 * (1 - final_size/initial_size):.1f}% reduction)")


def main():
    parser = argparse.ArgumentParser(description='Recompress JPEG images')
    parser.add_argument('--input', type=str, required=True,
                        help='Input directory containing images')
    parser.add_argument('--quality', type=int, default=85,
                        help='JPEG quality (0-100, default: 85)')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    
    if not input_dir.exists():
        print(f"❌ Error: Directory {input_dir} does not exist")
        return
    
    if not (0 <= args.quality <= 100):
        print(f"❌ Error: Quality must be between 0 and 100")
        return
    
    response = input(f"⚠️  This will overwrite all JPEG images in {input_dir} with quality={args.quality}. Continue? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled")
        return
    
    recompress_directory(input_dir, args.quality)


if __name__ == '__main__':
    main()
