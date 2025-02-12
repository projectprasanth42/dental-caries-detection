import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
import torchmetrics
from typing import List, Dict, Tuple

class DentalEvaluator:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.map_metric = torchmetrics.detection.MeanAveragePrecision()
        
    def calculate_metrics(self, predictions: List[Dict[str, torch.Tensor]], 
                         targets: List[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """
        Calculate detection and segmentation metrics
        """
        # Format predictions and targets for MAP calculation
        pred_formatted = []
        target_formatted = []
        
        for pred, target in zip(predictions, targets):
            pred_formatted.append({
                'boxes': pred['boxes'],
                'scores': pred['scores'],
                'labels': pred['labels'],
                'masks': pred['masks']
            })
            
            target_formatted.append({
                'boxes': target['boxes'],
                'labels': target['labels'],
                'masks': target['masks']
            })
        
        # Calculate mAP
        self.map_metric.update(pred_formatted, target_formatted)
        map_results = self.map_metric.compute()
        
        # Calculate IoU for segmentation
        ious = []
        for pred, target in zip(predictions, targets):
            pred_masks = pred['masks'].squeeze(1)
            target_masks = target['masks']
            
            for p_mask, t_mask in zip(pred_masks, target_masks):
                intersection = torch.logical_and(p_mask, t_mask).sum()
                union = torch.logical_or(p_mask, t_mask).sum()
                iou = (intersection / union).item() if union > 0 else 0.0
                ious.append(iou)
        
        metrics = {
            'mAP': map_results['map'].item(),
            'mAP_50': map_results['map_50'].item(),
            'mAP_75': map_results['map_75'].item(),
            'mean_IoU': np.mean(ious) if ious else 0.0
        }
        
        return metrics

    def visualize_predictions(self, image: torch.Tensor,
                            prediction: Dict[str, torch.Tensor],
                            target: Dict[str, torch.Tensor] = None,
                            score_threshold: float = 0.5) -> None:
        """
        Visualize detection and segmentation results
        """
        # Convert image to uint8 for visualization
        image = (image * 255).byte()
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2 if target is not None else 1, figsize=(15, 7))
        if target is None:
            axes = [axes]
        
        # Plot prediction
        pred_image = image.clone()
        
        # Filter predictions by score threshold
        mask = prediction['scores'] > score_threshold
        boxes = prediction['boxes'][mask]
        labels = prediction['labels'][mask]
        masks = prediction['masks'][mask]
        
        # Draw boxes and masks
        pred_image = draw_bounding_boxes(pred_image, boxes, labels.tolist())
        pred_image = draw_segmentation_masks(pred_image, masks.squeeze(1))
        
        axes[0].imshow(pred_image.permute(1, 2, 0))
        axes[0].set_title('Predictions')
        axes[0].axis('off')
        
        # Plot ground truth if available
        if target is not None:
            target_image = image.clone()
            target_image = draw_bounding_boxes(target_image, target['boxes'], target['labels'].tolist())
            target_image = draw_segmentation_masks(target_image, target['masks'])
            
            axes[1].imshow(target_image.permute(1, 2, 0))
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()

    def analyze_failure_cases(self, predictions: List[Dict[str, torch.Tensor]],
                            targets: List[Dict[str, torch.Tensor]],
                            score_threshold: float = 0.5) -> Dict[str, List[int]]:
        """
        Analyze failure cases to identify patterns
        """
        failure_cases = {
            'false_positives': [],
            'false_negatives': [],
            'low_iou': []
        }
        
        for idx, (pred, target) in enumerate(zip(predictions, targets)):
            # Filter predictions by score threshold
            mask = pred['scores'] > score_threshold
            pred_boxes = pred['boxes'][mask]
            pred_labels = pred['labels'][mask]
            
            target_boxes = target['boxes']
            target_labels = target['labels']
            
            # Calculate IoU between predicted and target boxes
            ious = torchmetrics.detection.intersection_over_union(pred_boxes, target_boxes)
            
            # Check for false positives and false negatives
            if len(pred_boxes) > len(target_boxes):
                failure_cases['false_positives'].append(idx)
            elif len(pred_boxes) < len(target_boxes):
                failure_cases['false_negatives'].append(idx)
            
            # Check for low IoU cases
            if ious.shape[0] > 0 and ious.max() < 0.5:
                failure_cases['low_iou'].append(idx)
        
        return failure_cases 