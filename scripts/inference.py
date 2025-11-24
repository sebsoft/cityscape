"""
YOLO Inference Script for Cityscapes-trained Model

This script runs inference using a trained YOLO model on images or videos.

Usage:
    # Inference on single image
    python scripts/inference.py --model runs/train/cityscapes_yolo/weights/best.pt --source test.jpg
    
    # Inference on directory of images
    python scripts/inference.py --model runs/train/cityscapes_yolo/weights/best.pt --source images/
    
    # Inference on video
    python scripts/inference.py --model runs/train/cityscapes_yolo/weights/best.pt --source video.mp4
    
    # Real-time webcam inference
    python scripts/inference.py --model runs/train/cityscapes_yolo/weights/best.pt --source 0
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import cv2


def main():
    parser = argparse.ArgumentParser(description='Run YOLO inference on Cityscapes-trained model')
    
    # Model and source
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained YOLO model (.pt file)')
    parser.add_argument('--source', type=str, required=True,
                        help='Source: image file, directory, video file, or webcam (0)')
    
    # Inference parameters
    parser.add_argument('--img', type=int, default=640,
                        help='Inference image size (pixels)')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold for detections')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=300,
                        help='Maximum number of detections per image')
    
    # Output options
    parser.add_argument('--save', action='store_true', default=True,
                        help='Save inference results')
    parser.add_argument('--nosave', action='store_true',
                        help='Do not save inference results')
    parser.add_argument('--save-txt', action='store_true',
                        help='Save results to *.txt files')
    parser.add_argument('--save-conf', action='store_true',
                        help='Save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true',
                        help='Save cropped prediction boxes')
    parser.add_argument('--project', type=str, default='runs/predict',
                        help='Project directory for outputs')
    parser.add_argument('--name', type=str, default='cityscapes_inference',
                        help='Experiment name')
    
    # Display options
    parser.add_argument('--show', action='store_true',
                        help='Display results in window')
    parser.add_argument('--show-labels', action='store_true', default=True,
                        help='Show object labels in results')
    parser.add_argument('--show-conf', action='store_true', default=True,
                        help='Show confidence scores in results')
    parser.add_argument('--line-width', type=int, default=None,
                        help='Bounding box line width (pixels)')
    
    # Other options
    parser.add_argument('--device', type=str, default='',
                        help='CUDA device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--vid-stride', type=int, default=1,
                        help='Video frame-rate stride (process every Nth frame)')
    parser.add_argument('--classes', type=int, nargs='+', default=None,
                        help='Filter by class: --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='Class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='Use augmented inference')
    
    args = parser.parse_args()
    
    # Determine save flag
    save = args.save and not args.nosave
    
    # Print configuration
    print("\n" + "="*80)
    print("YOLO INFERENCE")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Source: {args.source}")
    print(f"Image size: {args.img}")
    print(f"Confidence threshold: {args.conf}")
    print(f"IoU threshold: {args.iou}")
    print(f"Device: {args.device if args.device else 'auto'}")
    print(f"Save results: {save}")
    if save:
        print(f"Output: {args.project}/{args.name}")
    print("="*80 + "\n")
    
    # Load model
    print(f"Loading model from: {args.model}")
    model = YOLO(args.model)
    
    # Print model info
    print(f"Model loaded successfully!")
    print(f"  Classes: {len(model.names)}")
    print(f"  Class names: {list(model.names.values())}\n")
    
    # Run inference
    results = model.predict(
        source=args.source,
        imgsz=args.img,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device if args.device else None,
        
        # Save options
        save=save,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        save_crop=args.save_crop,
        project=args.project,
        name=args.name,
        
        # Display options
        show=args.show,
        show_labels=args.show_labels,
        show_conf=args.show_conf,
        line_width=args.line_width,
        
        # Other options
        vid_stride=args.vid_stride,
        classes=args.classes,
        agnostic_nms=args.agnostic_nms,
        augment=args.augment,
        verbose=True,
    )
    
    # Print summary
    print("\n" + "="*80)
    print("INFERENCE COMPLETE")
    print("="*80)
    
    if save:
        save_dir = Path(args.project) / args.name
        print(f"Results saved to: {save_dir}")
        
        if args.save_txt:
            print(f"  Text labels saved to: {save_dir / 'labels'}")
        if args.save_crop:
            print(f"  Cropped detections saved to: {save_dir / 'crops'}")
    
    # Print detection statistics for batch inference
    if len(results) > 1:
        total_detections = sum(len(r.boxes) for r in results)
        avg_detections = total_detections / len(results)
        print(f"\nProcessed {len(results)} images")
        print(f"  Total detections: {total_detections}")
        print(f"  Average detections per image: {avg_detections:.2f}")
    
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
