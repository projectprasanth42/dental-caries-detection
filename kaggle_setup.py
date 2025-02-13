import os
import subprocess
import sys

def setup_environment():
    """Set up the Kaggle environment with required packages and configurations"""
    print("Setting up the environment...")
    
    # Install required packages
    subprocess.check_call([sys.executable, "-m", "pip", "install", 
        "torch", 
        "torchvision", 
        "albumentations",
        "opencv-python",
        "numpy>=1.19.5",
        "tqdm"
    ])
    
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