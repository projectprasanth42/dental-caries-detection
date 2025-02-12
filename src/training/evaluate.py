import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mask_rcnn import DentalCariesMaskRCNN
from data.dataset import DentalCariesDataset
from utils.evaluation import DentalEvaluator
from configs.model_config import ModelConfig

def evaluate_model(model, test_loader, evaluator, device, save_dir):
    """
    Evaluate model performance on test set
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    # Create directories for saving results
    save_dir = Path(save_dir)
    (save_dir / 'visualizations').mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(tqdm(test_loader, desc="Evaluating")):
            # Move data to device
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Get predictions
            predictions = model.evaluate_step(images)
            
            # Store predictions and targets
            all_predictions.extend(predictions)
            all_targets.extend(targets)
            
            # Save visualization for first 10 samples
            if i < 10:
                evaluator.visualize_predictions(
                    image=images[0],
                    prediction=predictions[0],
                    target=targets[0]
                )
                plt.savefig(save_dir / 'visualizations' / f'sample_{i}.png')
                plt.close()
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics(all_predictions, all_targets)
    
    # Analyze failure cases
    failure_analysis = evaluator.analyze_failure_cases(all_predictions, all_targets)
    
    # Combine results
    results = {
        'metrics': metrics,
        'failure_analysis': {
            'false_positives': len(failure_analysis['false_positives']),
            'false_negatives': len(failure_analysis['false_negatives']),
            'low_iou_cases': len(failure_analysis['low_iou'])
        }
    }
    
    # Save results
    with open(save_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    return results

def main():
    # Load configuration
    config = ModelConfig()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = DentalCariesMaskRCNN(
        num_classes=config.num_classes,
        hidden_dim=config.hidden_dim
    ).to(device)
    
    # Load trained weights
    checkpoint_path = Path('checkpoints/best_model.pth')
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded trained model weights")
    else:
        raise ValueError("No trained model weights found")
    
    # Create test dataset and loader
    test_dataset = DentalCariesDataset(
        config.val_data_path,  # Using validation set as test set for this example
        config.val_labels_path,
        is_training=False
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    # Initialize evaluator
    evaluator = DentalEvaluator(num_classes=config.num_classes)
    
    # Create results directory
    results_dir = Path('evaluation_results') / datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate model
    results = evaluate_model(model, test_loader, evaluator, device, results_dir)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"mAP: {results['metrics']['mAP']:.4f}")
    print(f"mAP@50: {results['metrics']['mAP_50']:.4f}")
    print(f"mAP@75: {results['metrics']['mAP_75']:.4f}")
    print(f"Mean IoU: {results['metrics']['mean_IoU']:.4f}")
    
    print("\nFailure Analysis:")
    print(f"False Positives: {results['failure_analysis']['false_positives']}")
    print(f"False Negatives: {results['failure_analysis']['false_negatives']}")
    print(f"Low IoU Cases: {results['failure_analysis']['low_iou_cases']}")

if __name__ == "__main__":
    main() 