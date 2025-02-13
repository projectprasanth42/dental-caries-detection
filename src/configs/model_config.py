from dataclasses import dataclass

@dataclass
class ModelConfig:
    # Model Architecture
    backbone = "resnet34"  # Changed from resnet101 for memory efficiency
    num_classes = 3
    hidden_dim = 128  # Further reduced from 256
    nheads = 8  # Reduced from 16
    num_encoder_layers = 4  # Reduced from 8
    num_decoder_layers = 4  # Reduced from 8
    
    # Advanced Model Settings
    fpn_channels = 128  # Reduced from 256
    roi_pool_size = 5  # Reduced from 7
    attention_dropout = 0.1  # Reduced from 0.3
    stochastic_depth_prob = 0.1  # Reduced from 0.2
    
    # Training Parameters
    batch_size = 1  # Reduced from 2
    gradient_accumulation_steps = 8  # Increased from 4
    learning_rate = 0.00001  # Reduced from 0.00005
    weight_decay = 0.0001  # Reduced from 0.0005
    num_epochs = 100
    early_stopping_patience = 10  # Reduced from 15
    num_workers = 0  # Changed from 2 to avoid memory issues
    
    # Optimizer Settings
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    gradient_clip_val = 0.5  # Reduced from 1.0
    
    # Learning Rate Schedule
    warmup_epochs = 3  # Reduced from 5
    min_lr = 1e-7  # Reduced from 1e-6
    lr_schedule_patience = 3  # Reduced from 5
    lr_reduce_factor = 0.1  # Changed from 0.5
    
    # Loss weights
    detection_loss_weight = 1.5
    classification_loss_weight = 1.2
    segmentation_loss_weight = 2.0
    rpn_loss_weight = 1.2
    mask_loss_weight = 2.0
    
    # RPN Settings
    rpn_fg_iou_thresh = 0.7
    rpn_bg_iou_thresh = 0.3
    rpn_positive_fraction = 0.7
    rpn_score_thresh = 0.05
    
    # ROI Settings
    box_score_thresh = 0.05
    box_nms_thresh = 0.5
    box_detections_per_img = 50  # Reduced from 100
    
    # Data Augmentation
    augmentation_prob = 0.8
    rotation_range = (-30, 30)
    scale_range = (0.8, 1.2)
    brightness_range = (0.9, 1.1)
    contrast_range = (0.9, 1.1)
    
    # Advanced Training Features
    mixup_alpha = 0.2
    cutmix_alpha = 1.0
    label_smoothing = 0.1
    focal_loss_gamma = 2.0
    
    # Paths
    train_data_path = "/kaggle/input/dental-caries-dataset/X_train.npy"  # Fixed path
    train_labels_path = "/kaggle/input/dental-caries-dataset/y_train.npy"  # Fixed path
    val_data_path = "/kaggle/input/dental-caries-dataset/X_val.npy"  # Fixed path
    val_labels_path = "/kaggle/input/dental-caries-dataset/y_val.npy"  # Fixed path
    
    # Device
    device = "cuda"  # Using GPU for training
    