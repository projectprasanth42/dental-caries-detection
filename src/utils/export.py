import torch
import onnx
import onnxruntime
import numpy as np
from pathlib import Path
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mask_rcnn import DentalCariesMaskRCNN
from configs.model_config import ModelConfig

def export_to_onnx(model, output_path, input_shape=(3, 512, 512)):
    """
    Export PyTorch model to ONNX format
    
    Args:
        model: PyTorch model
        output_path: Path to save the ONNX model
        input_shape: Input shape for the model (C, H, W)
    """
    # Create dummy input
    dummy_input = torch.randn(1, *input_shape)
    
    # Export model
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    # Verify the exported model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"Model saved to {output_path}")
    
    return onnx_model

def test_onnx_model(onnx_path, test_input):
    """
    Test the exported ONNX model
    
    Args:
        onnx_path: Path to the ONNX model
        test_input: Test input tensor
    """
    # Create ONNX Runtime session
    session = onnxruntime.InferenceSession(onnx_path)
    
    # Prepare input
    input_name = session.get_inputs()[0].name
    input_data = test_input.numpy()
    
    # Run inference
    outputs = session.run(None, {input_name: input_data})
    
    return outputs

def main():
    # Load configuration
    config = ModelConfig()
    
    # Initialize model
    model = DentalCariesMaskRCNN(
        num_classes=config.num_classes,
        hidden_dim=config.hidden_dim
    )
    
    # Load trained weights
    checkpoint_path = Path('checkpoints/best_model.pth')
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded trained model weights")
    else:
        print("Warning: No trained weights found")
    
    # Set model to evaluation mode
    model.eval()
    
    # Create output directory
    output_dir = Path('exported_models')
    output_dir.mkdir(exist_ok=True)
    
    # Export model
    output_path = output_dir / 'dental_caries_detection.onnx'
    onnx_model = export_to_onnx(model, output_path)
    
    # Test the exported model
    test_input = torch.randn(1, 3, 512, 512)
    outputs = test_onnx_model(output_path, test_input)
    
    print("Model exported and tested successfully")
    print(f"Output shapes: {[out.shape for out in outputs]}")

if __name__ == "__main__":
    main() 