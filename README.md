# Dental Caries Detection

A deep learning model for detecting dental caries in X-ray images using PyTorch.

## Overview

This project implements a dental caries detection system using a Mask R-CNN model with ResNet34 backbone. The model is designed to identify and segment dental caries in X-ray images.

## Running on Google Colab

1. Open the `colab_train.ipynb` notebook in Google Colab
2. Upload your dataset to Google Drive:
   - Create a folder named `dental_caries_dataset` in your Drive
   - Upload the following files:
     - `X_train.npy`: Training images
     - `y_train.npy`: Training labels
     - `X_val.npy`: Validation images
     - `y_val.npy`: Validation labels
3. Run the notebook cells in order:
   - First cell installs dependencies and verifies GPU
   - Second cell sets up data paths
   - Third cell starts the training process

## Model Architecture

- Backbone: ResNet34 (memory efficient)
- Feature Pyramid Network (FPN)
- Region Proposal Network (RPN)
- ROI Align for feature extraction
- Mask head for segmentation

## Training Configuration

The model uses the following optimizations:
- Mixed precision training
- Memory-efficient training loop
- Gradient accumulation
- Conservative GPU memory usage

Key parameters:
- Batch size: 1 (with gradient accumulation)
- Learning rate: 1e-5
- Early stopping patience: 10 epochs
- Optimizer: AdamW with weight decay

## Requirements

See `requirements.txt` for full list. Key dependencies:
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- albumentations >= 1.3.0
- OpenCV >= 4.7.0

## Project Structure

```
dental-caries-detection/
├── src/
│   ├── configs/         # Model and training configurations
│   ├── data/           # Dataset and data loading
│   ├── models/         # Model architecture
│   └── training/       # Training loops and utilities
├── colab_train.ipynb   # Colab training notebook
└── requirements.txt    # Project dependencies
```

## GPU Requirements

- Recommended: Any CUDA-capable GPU with 12GB+ memory
- Minimum: 8GB GPU memory (with current memory optimizations)
- Google Colab's T4/P100 GPUs are sufficient

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```
@misc{dental-caries-detection,
  author = {Your Name},
  title = {Dental Caries Detection using Deep Learning},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/projectprasanth42/dental-caries-detection}
}
```

## Features

- Dental caries detection and segmentation
- Severity classification
- Real-time analysis
- Web interface for clinical use

## Setup

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run training:
```bash
python src/training/train.py
```

## Author

- projectprasanth42

## Project Structure

```
├── preprocessed_dataset/     # Preprocessed training data
│   ├── X_train.npy          # Training images
│   ├── X_val.npy            # Validation images
│   ├── y_train.npy          # Training labels
│   └── y_val.npy            # Validation labels
├── src/
│   ├── data/                # Data loading and preprocessing
│   ├── models/              # Model architectures
│   ├── training/            # Training scripts
│   ├── utils/               # Utility functions
│   └── webapp/              # Web interface
├── configs/                 # Configuration files
├── notebooks/              # Jupyter notebooks for analysis
└── requirements.txt        # Project dependencies
```

## Usage

1. Data Preprocessing:
```bash
python src/data/preprocess.py
```

2. Training:
```bash
python src/training/train.py
```

3. Evaluation:
```bash
python src/training/evaluate.py
```

4. Web Interface:
```bash
python src/webapp/app.py
```

## Model Architecture

The system uses a Mask R-CNN model with ResNet-50 backbone for:
- Object Detection: Localizing caries regions
- Instance Segmentation: Precise delineation of caries
- Classification: Determining caries severity

## Performance Metrics

- Detection: Mean Average Precision (mAP)
- Segmentation: Dice Coefficient & IoU
- Classification: F1-score

## License

[Your chosen license] 