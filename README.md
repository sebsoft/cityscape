# YOLO Cityscapes Training Project

A complete project for training YOLOv8 models on the Cityscapes dataset for urban scene understanding and object detection.

## Overview

This project provides scripts and configurations to:
- Prepare the Cityscapes dataset in YOLO format
- Train YOLOv8 models with customizable parameters
- Run inference on images, videos, or webcam feeds
- Evaluate model performance on urban street scenes

## Dataset

**Cityscapes** is a large-scale dataset focused on semantic understanding of urban street scenes. It contains:
- 5,000 high-quality annotated images (2,975 train, 500 val, 1,525 test)
- 20,000 additional weakly annotated images
- Dense pixel-level annotations
- 30 classes covering vehicles, pedestrians, road infrastructure

**Classes used in this project (8 classes):**
- person
- rider
- car
- truck
- bus
- train
- motorcycle
- bicycle

## Project Structure

```
cityscape/
├── configs/
│   └── cityscapes.yaml          # Dataset configuration for YOLO
├── data/
│   ├── cityscapes_raw/          # Raw Cityscapes dataset (after download)
│   └── cityscapes_yolo/         # Converted dataset in YOLO format
│       ├── images/
│       │   ├── train/
│       │   └── val/
│       └── labels/
│           ├── train/
│           └── val/
├── models/                      # Directory for pre-trained models
├── runs/
│   ├── train/                   # Training outputs
│   └── predict/                 # Inference outputs
├── scripts/
│   ├── prepare_dataset.py       # Dataset preparation script
│   ├── train.py                 # Training script
│   └── inference.py             # Inference script
└── requirements.txt             # Python dependencies
```

## Installation

### 1. Clone the repository and navigate to project directory

```bash
cd /Users/sebsoft/projects/cityscape
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Download Cityscapes Dataset

1. Register at [Cityscapes Dataset](https://www.cityscapes-dataset.com/)
2. Download the following files:
   - `leftImg8bit_trainvaltest.zip` (11GB) - Images
   - `gtFine_trainvaltest.zip` (241MB) - Fine annotations

3. Extract both archives to `data/cityscapes_raw/`:

```bash
mkdir -p data/cityscapes_raw
# Extract downloaded files
unzip leftImg8bit_trainvaltest.zip -d data/cityscapes_raw/
unzip gtFine_trainvaltest.zip -d data/cityscapes_raw/
```

Expected structure:
```
data/cityscapes_raw/
├── leftImg8bit/
│   ├── train/
│   ├── val/
│   └── test/
└── gtFine/
    ├── train/
    ├── val/
    └── test/
```

### Step 2: Prepare Dataset for YOLO

Convert Cityscapes annotations to YOLO format:

```bash
python scripts/prepare_dataset.py --cityscapes-dir data/cityscapes_raw
```

This will:
- Convert polygon annotations to bounding boxes
- Normalize coordinates to YOLO format
- Filter to include only the 8 object detection classes
- Create train/val splits in `data/cityscapes_yolo/`

### Step 3: Train YOLO Model

Start training with default settings:

```bash
python scripts/train.py --model yolov8n.pt --epochs 100 --batch 16
```

**Training options:**

```bash
# Train with different model sizes
python scripts/train.py --model yolov8s.pt --epochs 100 --batch 16  # Small model
python scripts/train.py --model yolov8m.pt --epochs 150 --batch 8   # Medium model
python scripts/train.py --model yolov8l.pt --epochs 200 --batch 4   # Large model

# Train with larger images
python scripts/train.py --model yolov8n.pt --epochs 100 --batch 8 --img 1024

# Resume training from checkpoint
python scripts/train.py --resume runs/train/cityscapes_yolo/weights/last.pt

# Custom learning rate and augmentation
python scripts/train.py --model yolov8n.pt --epochs 100 --lr0 0.001 --mosaic 0.8
```

**Key parameters:**
- `--model`: Model variant (yolov8n/s/m/l/x - nano to xlarge)
- `--epochs`: Number of training epochs
- `--batch`: Batch size (reduce if out of memory)
- `--img`: Input image size (640, 1024, etc.)
- `--device`: GPU device (0, 0,1, cpu)
- `--workers`: Number of data loading workers

Training outputs will be saved to `runs/train/cityscapes_yolo/`:
- `weights/best.pt` - Best model checkpoint
- `weights/last.pt` - Latest model checkpoint
- Training curves, confusion matrix, sample predictions

### Step 4: Run Inference

Inference on single image:

```bash
python scripts/inference.py \
    --model runs/train/cityscapes_yolo/weights/best.pt \
    --source path/to/image.jpg
```

Inference on directory:

```bash
python scripts/inference.py \
    --model runs/train/cityscapes_yolo/weights/best.pt \
    --source path/to/images/
```

Inference on video:

```bash
python scripts/inference.py \
    --model runs/train/cityscapes_yolo/weights/best.pt \
    --source path/to/video.mp4
```

Real-time webcam inference:

```bash
python scripts/inference.py \
    --model runs/train/cityscapes_yolo/weights/best.pt \
    --source 0
```

**Inference options:**

```bash
# Adjust confidence threshold
python scripts/inference.py --model best.pt --source image.jpg --conf 0.5

# Filter specific classes (e.g., only cars and trucks)
python scripts/inference.py --model best.pt --source image.jpg --classes 2 3

# Save crops of detected objects
python scripts/inference.py --model best.pt --source image.jpg --save-crop

# Display results in window
python scripts/inference.py --model best.pt --source image.jpg --show
```

Results will be saved to `runs/predict/cityscapes_inference/`

## Model Performance

Expected performance on Cityscapes validation set (after training):

| Model    | mAP50 | mAP50-95 | Speed (ms) | Parameters |
|----------|-------|----------|------------|------------|
| YOLOv8n  | ~0.65 | ~0.45    | 5-10       | 3.2M       |
| YOLOv8s  | ~0.70 | ~0.50    | 10-15      | 11.2M      |
| YOLOv8m  | ~0.73 | ~0.53    | 20-30      | 25.9M      |
| YOLOv8l  | ~0.75 | ~0.55    | 35-50      | 43.7M      |

*Note: Performance varies based on training epochs, batch size, and image resolution.*

## Tips for Better Results

1. **Longer training**: Urban scenes are complex - train for 150-300 epochs
2. **Larger images**: Use `--img 1024` for better small object detection
3. **Data augmentation**: Experiment with mosaic, mixup, and HSV augmentation
4. **Class weighting**: Some classes (train, bus) are rare - consider focal loss
5. **Multi-scale training**: Helps detect objects at various distances
6. **Post-processing**: Adjust NMS IoU threshold for dense urban scenes

## Troubleshooting

**Out of memory errors:**
- Reduce batch size: `--batch 4` or `--batch 2`
- Reduce image size: `--img 480`
- Use smaller model: `yolov8n.pt`

**Slow training:**
- Increase workers: `--workers 16`
- Enable caching: `--cache`
- Use GPU if available

**Poor performance:**
- Train longer (more epochs)
- Use larger model variant
- Increase image size
- Check dataset conversion was successful

## Advanced Usage

### Custom Training Configuration

Create a custom training config by modifying parameters in `scripts/train.py` or pass them via command line:

```bash
python scripts/train.py \
    --model yolov8m.pt \
    --epochs 200 \
    --batch 8 \
    --img 1024 \
    --lr0 0.01 \
    --lrf 0.01 \
    --momentum 0.937 \
    --weight-decay 0.0005 \
    --mosaic 1.0 \
    --fliplr 0.5 \
    --scale 0.5 \
    --patience 50
```

### Export Model for Deployment

Export trained model to different formats:

```python
from ultralytics import YOLO

model = YOLO('runs/train/cityscapes_yolo/weights/best.pt')

# Export to ONNX
model.export(format='onnx')

# Export to TensorRT
model.export(format='engine')

# Export to CoreML (macOS)
model.export(format='coreml')
```

## Citation

If you use Cityscapes dataset in your research, please cite:

```bibtex
@inproceedings{Cordts2016Cityscapes,
  title={The Cityscapes Dataset for Semantic Urban Scene Understanding},
  author={Cordts, Marius and Omran, Mohamed and Ramos, Sebastian and Rehfeld, Timo and Enzweiler, Markus and Benenson, Rodrigo and Franke, Uwe and Roth, Stefan and Schiele, Bernt},
  booktitle={Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2016}
}
```

## License

This project is for educational and research purposes. Please respect the Cityscapes dataset license terms.

## Resources

- [Cityscapes Dataset](https://www.cityscapes-dataset.com/)
- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [YOLO Paper](https://arxiv.org/abs/2305.09972)

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review Ultralytics documentation
3. Check Cityscapes dataset documentation

---

**Happy Training! 🚗🚌🚲👤**
