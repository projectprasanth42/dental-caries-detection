import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
from tqdm import tqdm
from datetime import datetime
import logging
import gc

from src.data.dataset import DentalCariesDataset
from src.models.mask_rcnn import DentalCariesMaskRCNN
from src.configs.model_config import ModelConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)

def collate_fn(batch):
    """Custom collate function for the DataLoader that properly handles dictionaries"""
    images = []
    targets = []
    for image, target in batch:
        images.append(image)
        targets.append(target)
    return images, targets

def get_dataloader_kwargs(config, is_train=True):
    """Get DataLoader kwargs based on config and whether it's training"""
    kwargs = {
        'batch_size': config.batch_size,
        'shuffle': is_train,
        'num_workers': config.num_workers,
        'collate_fn': collate_fn,
        'drop_last': is_train,  # Drop last incomplete batch during training
        'pin_memory': True  # Enable pin memory for faster GPU transfer
    }
    return kwargs

def train_model(config: ModelConfig):
    # Set device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
    
    # Enable garbage collection
    gc.enable()
    
    try:
        logging.info("Starting GPU training process")
        
        # Create datasets with augmentation
        logging.info("Creating datasets...")
        train_dataset = DentalCariesDataset(
            config.train_data_path,
            config.train_labels_path,
            is_training=True
        )
        
        val_dataset = DentalCariesDataset(
            config.val_data_path,
            config.val_labels_path,
            is_training=False
        )

        # Create data loaders
        logging.info("Creating data loaders...")
        train_loader = DataLoader(
            train_dataset,
            **get_dataloader_kwargs(config, is_train=True)
        )
        
        val_loader = DataLoader(
            val_dataset,
            **get_dataloader_kwargs(config, is_train=False)
        )

        # Initialize model
        logging.info("Initializing model...")
        model = DentalCariesMaskRCNN(
            num_classes=config.num_classes,
            hidden_dim=config.hidden_dim
        ).to(device)

        # Initialize optimizer with weight decay
        param_groups = [
            {'params': [], 'weight_decay': 0.0},  # no weight decay
            {'params': [], 'weight_decay': config.weight_decay}  # with weight decay
        ]
        
        for name, param in model.named_parameters():
            if any(nd in name for nd in ['bias', 'LayerNorm', 'BatchNorm']):
                param_groups[0]['params'].append(param)
            else:
                param_groups[1]['params'].append(param)

        optimizer = AdamW(
            param_groups,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps
        )

        # Cosine learning rate scheduler
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.num_epochs // 3,
            T_mult=2,
            eta_min=config.min_lr
        )

        # Training loop setup
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Create directory for saving models
        save_dir = os.path.join('checkpoints', datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(save_dir, exist_ok=True)
        logging.info(f"Model checkpoints will be saved to {save_dir}")

        # Training loop
        for epoch in range(config.num_epochs):
            logging.info(f"\nEpoch {epoch+1}/{config.num_epochs}")
            
            # Training phase
            model.train()
            train_losses = []
            progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
            
            for batch_idx, (images, targets) in enumerate(progress_bar):
                try:
                    # Move data to GPU
                    images = [image.to(device) for image in images]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    
                    # Forward pass and compute loss
                    loss, loss_dict = model.train_step(images, targets, optimizer)
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_val)
                    
                    # Update weights
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # Update progress bar
                    loss_item = loss.detach().cpu().item()
                    train_losses.append(loss_item)
                    progress_bar.set_postfix({
                        'loss': f"{loss_item:.4f}",
                        'avg_loss': f"{np.mean(train_losses):.4f}",
                        'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
                    })
                    
                    # Clear memory
                    del images, targets, loss_dict, loss
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    logging.error(f"Error in batch {batch_idx}: {str(e)}")
                    continue
            
            avg_train_loss = np.mean(train_losses)
            logging.info(f"Average training loss: {avg_train_loss:.4f}")
            
            # Validation phase
            model.eval()
            val_losses = []
            val_loss_dict = {}
            progress_bar = tqdm(val_loader, desc="Validation")
            
            with torch.no_grad():
                for images, targets in progress_bar:
                    try:
                        # Move data to GPU
                        images = [image.to(device) for image in images]
                        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                        
                        # Forward pass
                        loss, batch_loss_dict = model.validation_step(images, targets)
                        
                        # Process loss
                        loss_item = loss.detach().cpu().item()
                        if not np.isnan(loss_item):
                            val_losses.append(loss_item)
                            
                            # Accumulate individual losses
                            for k, v in batch_loss_dict.items():
                                if k not in val_loss_dict:
                                    val_loss_dict[k] = []
                                val_loss_dict[k].append(v.detach().cpu().item())
                        
                        # Update progress bar
                        progress_bar.set_postfix({'val_loss': f"{np.mean(val_losses):.4f}"})
                        
                        # Clear memory
                        del images, targets, batch_loss_dict, loss
                        torch.cuda.empty_cache()
                        
                    except Exception as e:
                        logging.error(f"Error in validation batch: {str(e)}")
                        continue
            
            # Calculate average validation loss
            avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
            
            # Log validation metrics
            logging.info(f"Average validation loss: {avg_val_loss:.4f}")
            if val_loss_dict:
                logging.info("Validation loss components:")
                for k, v in val_loss_dict.items():
                    avg = np.mean(v)
                    logging.info(f"  {k}: {avg:.4f}")
            
            # Learning rate scheduling
            if val_losses:
                scheduler.step()
            
            # Save best model
            if val_losses and (avg_val_loss < best_val_loss):
                best_val_loss = avg_val_loss
                patience_counter = 0
                checkpoint_path = os.path.join(save_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': avg_val_loss,
                }, checkpoint_path)
                logging.info(f"Saved best model to {checkpoint_path}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= config.early_stopping_patience:
                logging.info(f"Early stopping triggered after {epoch+1} epochs")
                break

    except Exception as e:
        logging.error(f"Training error: {str(e)}")
        raise e

if __name__ == "__main__":
    config = ModelConfig()
    train_model(config) 