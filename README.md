# BaselineNet ~ U-Net Baseline Detection

This project implements a U-Net-based model for detecting text baselines in document images. The model is trained to detect baselines and their start/end points, and can generate polygons that encompass the text lines.

## Features

- U-Net architecture with attention mechanism
- Multi-task learning for baseline and point detection
- Data augmentation using albumentations
- Training with early stopping and model checkpointing
- TensorBoard integration for monitoring training
- PAGE-XML output format compatible with Kraken
- JSON custom output format for line and polygon annotation
- Automatic polygon generation around baselines

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

1. Prepare your dataset in the following structure:
```
dataset/
├── images/
│   ├── image1.png
│   ├── image2.png
│   └── ...
└── labels/
    ├── image1.npy
    ├── image2.npy
    └── ...
```

2. Convert PAGE-XML files to training data:
```bash
python data/dataset_preparation.py \
    --xml_dir path/to/xml/files \
    --image_dir path/to/images \
    --output_dir path/to/dataset \
    --image_size 1024 1024
```

## Training

1. Configure training parameters in `configs/config.py`

2. Start training:
```bash
python train.py
```

The training script will:
- Split data into train/validation sets
- Train the model with early stopping
- Save checkpoints and best model
- Log metrics to TensorBoard

## Inference

Run inference on new images:
```bash
python predict.py \
    --model_path path/to/model/weights.pth \
    --image_path path/to/image.png \
    --output_path path/to/output.xml
```

## Model Architecture

The model uses a U-Net architecture with:
- Encoder-decoder structure
- Attention gates for better feature fusion
- Multi-task output (start points, end points, baseline)
- Skip connections for preserving spatial information

## Training Metrics

The model is trained using:
- Binary Cross Entropy loss for point detection
- Dice loss for baseline detection
- Combined loss for overall training
- Early stopping based on validation loss
- Learning rate scheduling

## Output Format

The model outputs PAGE-XML files containing:
- Detected baselines
- Generated polygons around baselines
- Metadata and image information

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
