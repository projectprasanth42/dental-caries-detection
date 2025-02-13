from dataclasses import dataclass

@dataclass
class ModelConfig:
    # Model Architecture - Lighter version
    backbone = "resnet18"  # Changed to even lighter backbone
    num_classes = 3
    hidden_dim = 64  # Reduced from 128
    nheads = 4  # Reduced from 8
    num_encoder_layers = 2  # Reduced from 4
    num_decoder_layers = 2  # Reduced from 4
    
    # Advanced Model Settings - Reduced complexity
    fpn_channels = 64  # Reduced from 128
    roi_pool_size = 5
    attention_dropout = 0.1
    stochastic_depth_prob = 0.1
    
    # Training Parameters - Conservative settings
    batch_size = 1
    gradient_accumulation_steps = 4
    learning_rate = 0.00001
    weight_decay = 0.0001
    num_epochs = 50  # Reduced from 100
    early_stopping_patience = 5  # Reduced from 10
    num_workers = 0  # Keep at 0 for Colab
    
    # Optimizer Settings
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    gradient_clip_val = 0.5
    
    # Learning Rate Schedule
    warmup_epochs = 2
    min_lr = 1e-7
    lr_schedule_patience = 2
    lr_reduce_factor = 0.5
    
    # Loss weights - Balanced for stability
    detection_loss_weight = 1.0
    classification_loss_weight = 1.0
    segmentation_loss_weight = 1.5
    rpn_loss_weight = 1.0
    mask_loss_weight = 1.5
    
    # RPN Settings - More conservative
    rpn_fg_iou_thresh = 0.6
    rpn_bg_iou_thresh = 0.3
    rpn_positive_fraction = 0.5
    rpn_score_thresh = 0.05
    
    # ROI Settings
    box_score_thresh = 0.05
    box_nms_thresh = 0.5
    box_detections_per_img = 25  # Reduced from 50
    
    # Data Augmentation - Reduced intensity
    augmentation_prob = 0.5
    rotation_range = (-15, 15)  # Reduced from (-30, 30)
    scale_range = (0.9, 1.1)  # Reduced from (0.8, 1.2)
    brightness_range = (0.9, 1.1)
    contrast_range = (0.9, 1.1)
    
    # Advanced Training Features - Reduced complexity
    mixup_alpha = 0.2
    cutmix_alpha = 0.5
    label_smoothing = 0.05
    focal_loss_gamma = 2.0
    
    # Memory Management
    max_grad_norm = 0.5
    pin_memory = False
    persistent_workers = False
    
    # Paths - Will be set in Colab notebook
    train_data_path = None
    train_labels_path = None
    val_data_path = None
    val_labels_path = None
    
    # Device
    device = "cuda"
    