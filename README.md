# Data Matrix YOLO Detection

A comprehensive project for training YOLO (You Only Look Once) models to detect Data Matrix codes in documents and images.

## Overview

This repository provides a complete pipeline for:

1. **Data Preparation**: Tools to convert, split, and validate datasets
2. **Model Training**: Advanced training tools for YOLO models
3. **Inference**: Fast and accurate detection of Data Matrix codes

The implementation supports both the [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) and [Hugging Face YOLO/DETR](https://huggingface.co/docs/transformers/model_doc/detr) implementations.

## Features

- **Flexible Model Options**: Support for YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l, and YOLOv8x
- **Customized Data Augmentation**: Optimized for Data Matrix detection
- **Advanced Training**: Learning rate scheduling, mixed precision, callbacks
- **Multiple Export Formats**: ONNX, TorchScript support
- **Comprehensive Logging**: TensorBoard and Weights & Biases integration
- **Inference Tools**: Batch processing and visualization

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/data-matrix-yolo-mcp.git
   cd data-matrix-yolo-mcp
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation

### Converting COCO Format to YOLO

```bash
python scripts/prepare_data.py convert --coco_file path/to/annotations.json --output_dir path/to/output
```

### Splitting a Dataset

```bash
python scripts/prepare_data.py split --input_dir path/to/dataset --output_dir path/to/output --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1
```

### Dataset Statistics

```bash
python scripts/prepare_data.py stats --data_dir path/to/dataset
```

## Training

### Basic Training

```bash
python scripts/train.py --config src/config/default.yaml --name my_training_run
```

### Custom Configuration

You can override any config parameter via command line:

```bash
python scripts/train.py --config src/config/default.yaml --model yolov8m --batch_size 8 --epochs 50 --lr 0.001 --img_size 640 --data_dir path/to/dataset
```

### Using Ultralytics vs. Transformers

By default, the training uses Ultralytics if available. To explicitly use transformers:

```bash
python scripts/train.py --config src/config/default.yaml --name transformers_run
```

And modify the config file to set `use_ultralytics: false`.

## Inference

### Single Image Inference

```bash
python scripts/inference.py --model path/to/model.pt --input path/to/image.jpg --output path/to/output.jpg
```

### Batch Processing

```bash
python scripts/inference.py --model path/to/model.pt --input path/to/images_dir --output path/to/output_dir --batch
```

### Adjusting Detection Parameters

```bash
python scripts/inference.py --model path/to/model.pt --input path/to/image.jpg --conf 0.3 --iou 0.5
```

## Project Structure

```
data-matrix-yolo-mcp/
├── data/                # Dataset storage
├── outputs/             # Training outputs and models
├── scripts/             # Command-line scripts
│   ├── prepare_data.py  # Dataset preparation
│   ├── train.py         # Training script
│   └── inference.py     # Inference script
├── src/                 # Core package
│   ├── config/          # Configuration
│   ├── data/            # Data loading
│   ├── models/          # Model definitions
│   ├── training/        # Training logic
│   └── utils/           # Utilities
├── tests/               # Tests
├── requirements.txt     # Dependencies
└── README.md            # Documentation
```

## Configuration

The training configuration is defined in YAML files. Key parameters include:

```yaml
# Data settings
data_dir: "path/to/dataset"
data_format: "yolo"  # Options: yolo, coco

# Model settings
model_name: "yolov8n"  # Options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
model_weights: null     # Path to pre-trained weights or null

# Training settings
batch_size: 16
img_size: 640
epochs: 100
learning_rate: 0.001
```

See `src/config/default.yaml` for a complete example.

## Tips for Data Matrix Detection

1. **Use Appropriate Resolution**: Data matrices often require higher resolution to be detected accurately. Consider using `img_size: 640` or higher.

2. **Balanced Augmentation**: The included augmentations are balanced for Data Matrix detection. Too aggressive transformations can make codes unreadable.

3. **Export for Deployment**: Use the ONNX export for deployment on edge devices or in production environments.

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
