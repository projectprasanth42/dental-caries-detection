from dataclasses import dataclass

@dataclass
class ModelConfig:
    # Model Architecture
    backbone = "resnet101"
    num_classes = 3
    hidden_dim = 512
    nheads = 16
    num_encoder_layers = 8
    num_decoder_layers = 8
    
    # Advanced Model Settings
    fpn_channels = 256
    roi_pool_size = 14
    attention_dropout = 0.3
    stochastic_depth_prob = 0.2
    
    # Training Parameters
    batch_size = 8  # Increased for GPU
    learning_rate = 0.0001  # Adjusted for GPU
    weight_decay = 0.0005
    num_epochs = 100
    early_stopping_patience = 15
    num_workers = 4  # Increased for GPU
    
    # Optimizer Settings
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    gradient_clip_val = 1.0
    
    # Learning Rate Schedule
    warmup_epochs = 3
    min_lr = 1e-6
    lr_schedule_patience = 5
    lr_reduce_factor = 0.5
    
    # Loss weights
    detection_loss_weight = 1.5
    classification_loss_weight = 1.2
    segmentation_loss_weight = 2.0
    rpn_loss_weight = 1.2
    mask_loss_weight = 2.0
    
    # RPN Settings
    rpn_fg_iou_thresh = 0.6
    rpn_bg_iou_thresh = 0.4
    rpn_positive_fraction = 0.7
    rpn_score_thresh = 0.05
    
    # ROI Settings
    box_score_thresh = 0.05
    box_nms_thresh = 0.3
    box_detections_per_img = 200
    
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
    train_data_path = "preprocessed_dataset/X_train.npy"
    train_labels_path = "preprocessed_dataset/y_train.npy"
    val_data_path = "preprocessed_dataset/X_val.npy"
    val_labels_path = "preprocessed_dataset/y_val.npy"
    
    # Device
    device = "cuda"  # Using GPU for training 