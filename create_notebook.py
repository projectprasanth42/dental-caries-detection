import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {"id": "title"},
            "source": [
                "# Dental Caries Detection - Training Notebook\n\n"
                "This notebook implements training for dental caries detection using Google Colab GPU.\n\n"
                "## Setup Steps:\n"
                "1. Verify GPU availability\n"
                "2. Install dependencies\n"
                "3. Clone repository and set up environment\n"
                "4. Prepare dataset\n"
                "5. Start training"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": "verify_gpu"},
            "source": ["# First, verify GPU is enabled\n!nvidia-smi"],
            "outputs": []
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": "install_dependencies"},
            "source": [
                "# Install PyTorch with CUDA support\n"
                "!pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118\n\n"
                "# Install other dependencies\n"
                "!pip install albumentations==1.3.1 opencv-python==4.8.0.74 numpy==1.24.3 tqdm==4.65.0\n\n"
                "# Import basic libraries\n"
                "import os\n"
                "import sys\n"
                "import torch\n"
                "import gc\n\n"
                "# Set CUDA environment variables\n"
                "os.environ['CUDA_LAUNCH_BLOCKING'] = '1'\n"
                "os.environ['TORCH_USE_CUDA_DSA'] = '1'\n"
                "os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'\n\n"
                "# Clear any existing memory\n"
                "gc.collect()\n"
                "if torch.cuda.is_available():\n"
                "    torch.cuda.empty_cache()\n"
                "    torch.cuda.reset_peak_memory_stats()"
            ],
            "outputs": []
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": "setup_repository"},
            "source": [
                "# Clone repository\n"
                "!git clone https://github.com/projectprasanth42/dental-caries-detection.git\n"
                "%cd dental-caries-detection\n\n"
                "# Add project to path\n"
                "project_path = os.path.abspath('.')\n"
                "if project_path not in sys.path:\n"
                "    sys.path.append(project_path)\n\n"
                "# Test CUDA setup\n"
                "def test_cuda():\n"
                "    try:\n"
                "        print(\"\\nGPU Information:\")\n"
                "        print(f\"PyTorch Version: {torch.__version__}\")\n"
                "        print(f\"CUDA Available: {torch.cuda.is_available()}\")\n"
                "        if torch.cuda.is_available():\n"
                "            print(f\"GPU Device: {torch.cuda.get_device_name(0)}\")\n"
                "            print(f\"CUDA Version: {torch.version.cuda}\")\n"
                "            \n"
                "            # Test small tensor operations\n"
                "            x = torch.ones(2, 2, device='cuda')\n"
                "            y = x + x\n"
                "            print(\"\\nCUDA Test Successful!\")\n"
                "            print(f\"Test tensor device: {y.device}\")\n"
                "            print(f\"Current memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB\")\n"
                "            \n"
                "            del x, y\n"
                "            torch.cuda.empty_cache()\n"
                "            return True\n"
                "    except Exception as e:\n"
                "        print(f\"\\nError testing CUDA: {str(e)}\")\n"
                "        return False\n\n"
                "cuda_ok = test_cuda()\n"
                "if not cuda_ok:\n"
                "    raise RuntimeError(\"CUDA setup failed. Please ensure GPU is enabled in Colab.\")"
            ],
            "outputs": []
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": "mount_drive"},
            "source": [
                "from google.colab import drive\n\n"
                "# Mount Google Drive\n"
                "drive.mount('/content/drive')\n\n"
                "# Set data paths\n"
                "DRIVE_PATH = '/content/drive/MyDrive/dental_caries_dataset'\n\n"
                "# Create config\n"
                "from src.configs.model_config import ModelConfig\n\n"
                "config = ModelConfig()\n\n"
                "# Update paths\n"
                "config.train_data_path = os.path.join(DRIVE_PATH, 'X_train.npy')\n"
                "config.train_labels_path = os.path.join(DRIVE_PATH, 'y_train.npy')\n"
                "config.val_data_path = os.path.join(DRIVE_PATH, 'X_val.npy')\n"
                "config.val_labels_path = os.path.join(DRIVE_PATH, 'y_val.npy')\n\n"
                "# Verify dataset\n"
                "import numpy as np\n\n"
                "def verify_dataset():\n"
                "    print(\"\\nChecking dataset:\")\n"
                "    for name, path in [\n"
                "        ('Training Data', config.train_data_path),\n"
                "        ('Training Labels', config.train_labels_path),\n"
                "        ('Validation Data', config.val_data_path),\n"
                "        ('Validation Labels', config.val_labels_path)\n"
                "    ]:\n"
                "        if os.path.exists(path):\n"
                "            data = np.load(path)\n"
                "            print(f\"{name}: ✓ Found - Shape: {data.shape}\")\n"
                "            del data\n"
                "        else:\n"
                "            print(f\"{name}: ✗ Not found at {path}\")\n"
                "            raise FileNotFoundError(f\"Dataset file not found: {path}\")\n\n"
                "verify_dataset()"
            ],
            "outputs": []
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": "start_training"},
            "source": [
                "from src.training.memory_efficient_train import memory_efficient_training\n"
                "import logging\n\n"
                "# Configure logging\n"
                "logging.basicConfig(\n"
                "    level=logging.INFO,\n"
                "    format='%(asctime)s - %(levelname)s - %(message)s'\n"
                ")\n\n"
                "# Additional memory cleanup before training\n"
                "gc.collect()\n"
                "if torch.cuda.is_available():\n"
                "    torch.cuda.empty_cache()\n"
                "    torch.cuda.reset_peak_memory_stats()\n\n"
                "# Start training with error handling\n"
                "try:\n"
                "    memory_efficient_training(config)\n"
                "except Exception as e:\n"
                "    logging.error(f\"Training failed: {str(e)}\")\n"
                "    logging.info(\"Cleaning up GPU memory...\")\n"
                "    gc.collect()\n"
                "    if torch.cuda.is_available():\n"
                "        torch.cuda.empty_cache()\n"
                "        torch.cuda.reset_peak_memory_stats()\n"
                "    raise"
            ],
            "outputs": []
        }
    ],
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "gpuType": "T4",
            "provenance": []
        },
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

with open('colab_train.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1) 