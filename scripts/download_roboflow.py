"""
Download Cityscapes Dataset from Roboflow

This script downloads a pre-processed Cityscapes dataset from Roboflow Universe.
The dataset is already in YOLO format, so no conversion is needed.

Usage:
    python scripts/download_roboflow.py --workspace <workspace> --project <project> --version <version>
    
Example:
    python scripts/download_roboflow.py --workspace cityscape --project cityscapes-detection --version 1
"""

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Download Cityscapes from Roboflow')
    parser.add_argument('--workspace', type=str, default='cityscapes',
                        help='Roboflow workspace name')
    parser.add_argument('--project', type=str, default='cityscapes-segmentation',
                        help='Roboflow project name')
    parser.add_argument('--version', type=int, default=1,
                        help='Dataset version number')
    parser.add_argument('--api-key', type=str, default=None,
                        help='Roboflow API key (or set ROBOFLOW_API_KEY env variable)')
    parser.add_argument('--output-dir', type=str, default='./data',
                        help='Output directory for dataset')
    parser.add_argument('--format', type=str, default='yolov8',
                        choices=['yolov8', 'yolov5', 'coco', 'voc'],
                        help='Dataset format')
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv('ROBOFLOW_API_KEY')
    
    if not api_key:
        print("\n" + "="*80)
        print("ROBOFLOW SETUP INSTRUCTIONS")
        print("="*80)
        print("\n1. Go to: https://roboflow.com/")
        print("2. Sign up for a free account")
        print("3. Get your API key from: https://app.roboflow.com/settings/api")
        print("\n4. Then run this script with your API key:")
        print(f"   python scripts/download_roboflow.py --api-key YOUR_API_KEY")
        print("\nOr set it as an environment variable:")
        print("   export ROBOFLOW_API_KEY=YOUR_API_KEY")
        print("   python scripts/download_roboflow.py")
        print("\n" + "="*80)
        print("\nALTERNATIVELY - Browse public datasets:")
        print("  1. Visit: https://universe.roboflow.com/")
        print("  2. Search for 'cityscapes' or 'urban detection'")
        print("  3. Click on a dataset you like")
        print("  4. Click 'Download' button")
        print("  5. Select 'YOLOv8' format")
        print("  6. Copy the code snippet provided")
        print("="*80 + "\n")
        return
    
    try:
        from roboflow import Roboflow
    except ImportError:
        print("\n❌ Error: 'roboflow' package not installed")
        print("\nInstalling roboflow package...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'roboflow'])
        from roboflow import Roboflow
        print("✓ Roboflow package installed!\n")
    
    print("\n" + "="*80)
    print("DOWNLOADING FROM ROBOFLOW")
    print("="*80)
    print(f"Workspace: {args.workspace}")
    print(f"Project: {args.project}")
    print(f"Version: {args.version}")
    print(f"Format: {args.format}")
    print(f"Output: {args.output_dir}")
    print("="*80 + "\n")
    
    # Initialize Roboflow
    rf = Roboflow(api_key=api_key)
    
    # Get project
    project = rf.workspace(args.workspace).project(args.project)
    
    # Get dataset version
    dataset = project.version(args.version)
    
    # Download dataset
    output_path = Path(args.output_dir) / 'cityscapes_roboflow'
    print(f"Downloading to: {output_path}")
    
    dataset.download(
        model_format=args.format,
        location=str(output_path)
    )
    
    print("\n" + "="*80)
    print("DOWNLOAD COMPLETE!")
    print("="*80)
    print(f"Dataset saved to: {output_path}")
    print("\nNext steps:")
    print("1. Update configs/cityscapes.yaml to point to this dataset")
    print("2. Start training with: python scripts/train.py")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
