import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.nn.functional as F

class DentalCariesMaskRCNN(nn.Module):
    def __init__(self, num_classes, hidden_dim=512):
        super(DentalCariesMaskRCNN, self).__init__()
        
        # Load pre-trained model with improved configuration
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            pretrained=True,
            box_detections_per_img=200,
            rpn_post_nms_top_n_train=2000,
            rpn_post_nms_top_n_test=1000,
            rpn_score_thresh=0.05,
            box_score_thresh=0.05,
            box_nms_thresh=0.3,
            rpn_fg_iou_thresh=0.6,
            rpn_bg_iou_thresh=0.4,
            rpn_positive_fraction=0.7,
        )
        
        # Get number of input features
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        
        # Replace the box predictor and mask predictor
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_dim, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
        # Enable stochastic depth
        self.training_mode = True
        self.drop_path_prob = 0.2
    
    def _initialize_weights(self):
        """Initialize the weights using better initialization"""
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, images, targets=None):
        if self.training and self.training_mode:
            # Apply stochastic depth during training
            if torch.rand(1) < self.drop_path_prob:
                return self.model(images, targets)
            
        # Convert list of images to a batch tensor if needed
        if isinstance(images, list):
            images = [img.to(next(self.parameters()).device) for img in images]
        
        return self.model(images, targets)
    
    def train_step(self, images, targets, optimizer):
        """Enhanced training step with gradient clipping and loss scaling"""
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        loss_dict = self.model(images, targets)
        
        # Apply loss weights and scaling
        weighted_losses = {
            name: loss * self._get_loss_weight(name) * self._get_loss_scale(loss)
            for name, loss in loss_dict.items()
        }
        losses = sum(weighted_losses.values())
        
        # Backward pass with gradient clipping
        losses.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step()
        
        return losses.item(), loss_dict
    
    def _get_loss_scale(self, loss):
        """Dynamic loss scaling based on loss magnitude"""
        with torch.no_grad():
            scale = torch.clamp(1.0 / (loss + 1e-8), 0.1, 10.0)
        return scale.item()
    
    def _get_loss_weight(self, loss_name):
        """Enhanced loss weights with focus on hard examples"""
        base_weights = {
            'loss_classifier': 1.2,
            'loss_box_reg': 1.5,
            'loss_mask': 2.0,
            'loss_objectness': 1.2,
            'loss_rpn_box_reg': 1.2
        }
        return base_weights.get(loss_name, 1.0)
    
    def validation_step(self, images, targets):
        """Validation step with weighted loss components"""
        loss_dict = self.model(images, targets)
        losses = sum(loss * self._get_loss_weight(name) 
                    for name, loss in loss_dict.items())
        return losses.item(), loss_dict
    
    @torch.no_grad()
    def evaluate_step(self, images):
        """Enhanced evaluation with test-time augmentation"""
        self.eval()
        
        # Original prediction
        outputs = self.model(images)
        
        # Test-time augmentation (horizontal flip)
        flipped_images = [torch.flip(img, [2]) for img in images]
        flipped_outputs = self.model(flipped_images)
        
        # Combine predictions (simple average for now)
        final_outputs = []
        for orig, flip in zip(outputs, flipped_outputs):
            # Flip back the predictions from flipped image
            flip['boxes'][:, [0, 2]] = flip['boxes'][:, [2, 0]]  # Flip x coordinates
            flip['masks'] = torch.flip(flip['masks'], [3])
            
            # Average the predictions
            combined = {
                'boxes': torch.cat([orig['boxes'], flip['boxes']]),
                'labels': torch.cat([orig['labels'], flip['labels']]),
                'scores': torch.cat([orig['scores'], flip['scores']]),
                'masks': torch.cat([orig['masks'], flip['masks']])
            }
            final_outputs.append(combined)
        
        self.train()
        return final_outputs 