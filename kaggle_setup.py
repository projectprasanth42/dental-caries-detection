import os
import subprocess
import sys

def setup_environment():
    """Set up the Kaggle environment with required packages and configurations"""
    print("Setting up the environment...")
    
    # Install specific PyTorch version compatible with P100
    subprocess.check_call([sys.executable, "-m", "pip", "install", 
        "torch==2.1.0+cu118", 
        "torchvision==0.16.0+cu118",
        "albumentations",
        "opencv-python",
        "numpy>=1.19.5",
        "tqdm"
    ], index_url="https://download.pytorch.org/whl/cu118")
    
    # Set CUDA environment variables
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TORCH_USE_CUDA_DSA'] = '1'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    # Clone the repository
    if not os.path.exists('dental-caries-detection'):
        subprocess.check_call(['git', 'clone', 'https://github.com/projectprasanth42/dental-caries-detection.git'])
    
    # Add project directory to Python path
    project_path = os.path.abspath('dental-caries-detection')
    if project_path not in sys.path:
        sys.path.append(project_path)
    
    # Create directories
    os.makedirs('checkpoints', exist_ok=True)
    
    print("Environment setup completed!")

if __name__ == "__main__":
    setup_environment() 