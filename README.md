# Dental Caries Detection Project

A deep learning-based system for detecting and analyzing dental caries using Mask R-CNN.

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