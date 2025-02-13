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
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "evaluation_title"},
            "source": [
                "## Model Evaluation and Visualization\n\n"
                "After training, we'll evaluate the model's performance and visualize some predictions."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": "load_best_model"},
            "source": [
                "import torch\n"
                "from src.models.mask_rcnn import DentalCariesMaskRCNN\n"
                "import matplotlib.pyplot as plt\n"
                "import cv2\n\n"
                "def load_best_model(config):\n"
                "    model = DentalCariesMaskRCNN(\n"
                "        num_classes=config.num_classes,\n"
                "        hidden_dim=config.hidden_dim\n"
                "    ).to('cuda')\n"
                "    \n"
                "    # Load the best checkpoint\n"
                "    checkpoints = [f for f in os.listdir('.') if f.endswith('.pth')]\n"
                "    if not checkpoints:\n"
                "        raise FileNotFoundError(\"No checkpoint files found!\")\n"
                "    \n"
                "    # Find the best checkpoint based on loss\n"
                "    best_loss = float('inf')\n"
                "    best_checkpoint = None\n"
                "    \n"
                "    for checkpoint in checkpoints:\n"
                "        state = torch.load(checkpoint)\n"
                "        if state['loss'] < best_loss:\n"
                "            best_loss = state['loss']\n"
                "            best_checkpoint = checkpoint\n"
                "    \n"
                "    print(f\"Loading best model from {best_checkpoint} with loss {best_loss:.4f}\")\n"
                "    state_dict = torch.load(best_checkpoint)\n"
                "    model.load_state_dict(state_dict['model_state_dict'])\n"
                "    return model\n\n"
                "# Load the best model\n"
                "model = load_best_model(config)\n"
                "model.eval();"
            ],
            "outputs": []
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": "evaluate_model"},
            "source": [
                "from src.data.dataset import DentalCariesDataset\n"
                "from torch.utils.data import DataLoader\n"
                "import numpy as np\n\n"
                "def evaluate_model(model, config):\n"
                "    val_dataset = DentalCariesDataset(\n"
                "        config.val_data_path,\n"
                "        config.val_labels_path,\n"
                "        is_training=False\n"
                "    )\n"
                "    \n"
                "    val_loader = DataLoader(\n"
                "        val_dataset,\n"
                "        batch_size=1,\n"
                "        shuffle=False,\n"
                "        num_workers=0,\n"
                "        collate_fn=lambda x: tuple(zip(*x))\n"
                "    )\n"
                "    \n"
                "    model.eval()\n"
                "    total_loss = 0\n"
                "    metrics = {\n"
                "        'detection_loss': 0,\n"
                "        'classification_loss': 0,\n"
                "        'segmentation_loss': 0\n"
                "    }\n"
                "    \n"
                "    print(\"\\nEvaluating model on validation set...\")\n"
                "    with torch.no_grad():\n"
                "        for images, targets in val_loader:\n"
                "            images = [img.to('cuda') for img in images]\n"
                "            targets = [{k: v.to('cuda') for k, v in t.items()} for t in targets]\n"
                "            \n"
                "            loss_dict = model(images, targets)\n"
                "            total_loss += sum(loss for loss in loss_dict.values())\n"
                "            \n"
                "            for k, v in loss_dict.items():\n"
                "                if k in metrics:\n"
                "                    metrics[k] += v.item()\n"
                "    \n"
                "    # Calculate average metrics\n"
                "    num_batches = len(val_loader)\n"
                "    avg_loss = total_loss / num_batches\n"
                "    metrics = {k: v/num_batches for k, v in metrics.items()}\n"
                "    \n"
                "    print(f\"\\nValidation Results:\")\n"
                "    print(f\"Average Loss: {avg_loss:.4f}\")\n"
                "    for k, v in metrics.items():\n"
                "        print(f\"{k}: {v:.4f}\")\n"
                "    \n"
                "    return avg_loss, metrics\n\n"
                "# Evaluate the model\n"
                "val_loss, metrics = evaluate_model(model, config)"
            ],
            "outputs": []
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": "visualize_predictions"},
            "source": [
                "def visualize_predictions(model, dataset, num_samples=5):\n"
                "    model.eval()\n"
                "    plt.figure(figsize=(20, 4*num_samples))\n"
                "    \n"
                "    for i in range(num_samples):\n"
                "        # Get a random sample\n"
                "        idx = np.random.randint(len(dataset))\n"
                "        image, target = dataset[idx]\n"
                "        \n"
                "        # Get prediction\n"
                "        with torch.no_grad():\n"
                "            prediction = model([image.to('cuda')])[0]\n"
                "        \n"
                "        # Move tensors to CPU for visualization\n"
                "        image = image.cpu().numpy().transpose(1, 2, 0)\n"
                "        masks = prediction['masks'].cpu().numpy()\n"
                "        scores = prediction['scores'].cpu().numpy()\n"
                "        labels = prediction['labels'].cpu().numpy()\n"
                "        \n"
                "        # Plot original image\n"
                "        plt.subplot(num_samples, 3, i*3 + 1)\n"
                "        plt.imshow(image)\n"
                "        plt.title('Original Image')\n"
                "        plt.axis('off')\n"
                "        \n"
                "        # Plot ground truth\n"
                "        plt.subplot(num_samples, 3, i*3 + 2)\n"
                "        plt.imshow(image)\n"
                "        for mask in target['masks'].cpu().numpy():\n"
                "            plt.imshow(mask[0], alpha=0.5, cmap='jet')\n"
                "        plt.title('Ground Truth')\n"
                "        plt.axis('off')\n"
                "        \n"
                "        # Plot prediction\n"
                "        plt.subplot(num_samples, 3, i*3 + 3)\n"
                "        plt.imshow(image)\n"
                "        for mask, score, label in zip(masks, scores, labels):\n"
                "            if score > 0.5:  # Confidence threshold\n"
                "                plt.imshow(mask[0], alpha=0.5, cmap='jet')\n"
                "                plt.text(10, 10, f'Class {label}: {score:.2f}', \n"
                "                        color='white', bbox=dict(facecolor='red', alpha=0.5))\n"
                "        plt.title('Prediction')\n"
                "        plt.axis('off')\n"
                "    \n"
                "    plt.tight_layout()\n"
                "    plt.show()\n\n"
                "# Create validation dataset\n"
                "val_dataset = DentalCariesDataset(\n"
                "    config.val_data_path,\n"
                "    config.val_labels_path,\n"
                "    is_training=False\n"
                ")\n\n"
                "# Visualize some predictions\n"
                "visualize_predictions(model, val_dataset, num_samples=3)"
            ],
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "inference_title"},
            "source": [
                "## Model Inference\n\n"
                "Use the trained model to make predictions on new images."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": "inference_function"},
            "source": [
                "def predict_image(model, image_path, confidence_threshold=0.5):\n"
                "    # Load and preprocess image\n"
                "    image = cv2.imread(image_path)\n"
                "    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)\n"
                "    \n"
                "    # Convert to tensor\n"
                "    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0\n"
                "    \n"
                "    # Make prediction\n"
                "    model.eval()\n"
                "    with torch.no_grad():\n"
                "        prediction = model([image_tensor.to('cuda')])[0]\n"
                "    \n"
                "    # Visualize results\n"
                "    plt.figure(figsize=(10, 5))\n"
                "    \n"
                "    # Original image\n"
                "    plt.subplot(1, 2, 1)\n"
                "    plt.imshow(image)\n"
                "    plt.title('Original Image')\n"
                "    plt.axis('off')\n"
                "    \n"
                "    # Prediction\n"
                "    plt.subplot(1, 2, 2)\n"
                "    plt.imshow(image)\n"
                "    \n"
                "    masks = prediction['masks'].cpu().numpy()\n"
                "    scores = prediction['scores'].cpu().numpy()\n"
                "    labels = prediction['labels'].cpu().numpy()\n"
                "    \n"
                "    for mask, score, label in zip(masks, scores, labels):\n"
                "        if score > confidence_threshold:\n"
                "            plt.imshow(mask[0], alpha=0.5, cmap='jet')\n"
                "            plt.text(10, 10 + label*20, f'Class {label}: {score:.2f}', \n"
                "                    color='white', bbox=dict(facecolor='red', alpha=0.5))\n"
                "    \n"
                "    plt.title('Prediction')\n"
                "    plt.axis('off')\n"
                "    plt.show()\n"
                "    \n"
                "    return prediction\n\n"
                "# Example usage:\n"
                "# prediction = predict_image(model, 'path_to_new_image.jpg')"
            ],
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {"id": "save_model"},
            "source": [
                "## Save Model for Deployment\n\n"
                "Save the trained model for future use or deployment."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": "save_model_code"},
            "source": [
                "# Save model to Drive\n"
                "SAVE_PATH = os.path.join(DRIVE_PATH, 'trained_model.pth')\n\n"
                "torch.save({\n"
                "    'model_state_dict': model.state_dict(),\n"
                "    'config': config.__dict__,\n"
                "    'metrics': metrics\n"
                "}, SAVE_PATH)\n\n"
                "print(f\"Model saved to {SAVE_PATH}\")"
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