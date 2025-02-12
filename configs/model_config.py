from dataclasses import dataclass

@dataclass
class ModelConfig:
    # Model Architecture
    backbone = "resnet50"
    num_classes = 3  # background, caries, deep caries
    hidden_dim = 256
    nheads = 8
    num_encoder_layers = 6
    num_decoder_layers = 6

    # Training Parameters
    batch_size = 8
    learning_rate = 0.001
    weight_decay = 0.0001
    num_epochs = 100
    early_stopping_patience = 10
    
    # Loss weights
    detection_loss_weight = 1.0
    classification_loss_weight = 1.0
    segmentation_loss_weight = 1.0
    
    # Data Augmentation
    augmentation_prob = 0.5
    rotation_range = (-30, 30)
    scale_range = (0.8, 1.2)
    
    # Paths
    train_data_path = "preprocessed_dataset/X_train.npy"
    train_labels_path = "preprocessed_dataset/y_train.npy"
    val_data_path = "preprocessed_dataset/X_val.npy"
    val_labels_path = "preprocessed_dataset/y_val.npy"
    
    # Device
    device = "cuda"  # or "cpu" based on availability 