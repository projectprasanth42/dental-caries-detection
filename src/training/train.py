import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
from datetime import datetime
import sys
import logging
import gc

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

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
    """
    Custom collate function for the DataLoader that properly handles dictionary targets
    """
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
        'pin_memory': True,  # Enable pin memory for GPU
    }
    
    # Only add persistent_workers if num_workers > 0
    if config.num_workers > 0:
        kwargs['persistent_workers'] = True
        
    return kwargs

def train_model(config: ModelConfig):
    try:
        # Enable garbage collection
        gc.enable()
        
        # Set device
        device = torch.device(config.device)
        logging.info(f"Using device: {device}")
        
        # Enable cuDNN benchmarking and deterministic mode
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = True
        
        # Initialize mixed precision training
        scaler = GradScaler()

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

        # Create data loaders with GPU optimizations
        logging.info("Creating data loaders...")
        train_loader = DataLoader(
            train_dataset,
            **get_dataloader_kwargs(config, is_train=True)
        )
        
        val_loader = DataLoader(
            val_dataset,
            **get_dataloader_kwargs(config, is_train=False)
        )

        # Initialize model with GPU optimizations
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

        # Cosine learning rate scheduler with warm restarts
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.num_epochs // 3,  # Restart every 1/3 of total epochs
            T_mult=2,  # Double the restart interval after each restart
            eta_min=config.min_lr
        )

        # Training loop with improvements
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Create directory for saving models
        save_dir = os.path.join('checkpoints', datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(save_dir, exist_ok=True)
        logging.info(f"Model checkpoints will be saved to {save_dir}")

        for epoch in range(config.num_epochs):
            logging.info(f"\nEpoch {epoch+1}/{config.num_epochs}")
            
            # Training phase
            model.train()
            train_losses = []
            
            pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{config.num_epochs}")
            
            for batch_idx, (images, targets) in enumerate(pbar):
                try:
                    # Clear memory periodically
                    if batch_idx % 5 == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
                    # Move data to device
                    images = [image.to(device) for image in images]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    
                    # Mixed precision training
                    with autocast(device_type='cuda', dtype=torch.float16):
                        loss, loss_dict = model.train_step(images, targets, optimizer)
                    
                    # Scale loss and backpropagate
                    scaler.scale(loss).backward()
                    
                    # Gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_val)
                    
                    # Update weights with gradient scaling
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    
                    train_losses.append(loss.item())
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'avg_loss': f'{np.mean(train_losses):.4f}',
                        'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
                    })
                    
                    # Log batch information
                    if batch_idx % 5 == 0:
                        logging.info(
                            f"Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)} - "
                            f"Loss: {loss.item():.4f}, Avg Loss: {np.mean(train_losses):.4f}, "
                            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
                        )
                    
                    # Clear memory
                    del images, targets, loss_dict, loss
                    
                except Exception as e:
                    logging.error(f"Error in batch {batch_idx}: {str(e)}")
                    continue
            
            avg_train_loss = np.mean(train_losses)
            logging.info(f"Average training loss: {avg_train_loss:.4f}")
            
            # Validation phase
            model.eval()
            val_losses = []
            val_loss_dict = {}
            
            pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{config.num_epochs}")
            
            with torch.no_grad():
                for images, targets in pbar:
                    try:
                        # Clear memory
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        # Move data to device
                        images = [image.to(device) for image in images]
                        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                        
                        # Mixed precision inference
                        with autocast(device_type='cuda', dtype=torch.float16):
                            loss, batch_loss_dict = model.validation_step(images, targets)
                        
                        if not torch.isnan(loss):
                            val_losses.append(loss.item())
                            
                            # Accumulate individual losses
                            for k, v in batch_loss_dict.items():
                                if k not in val_loss_dict:
                                    val_loss_dict[k] = []
                                val_loss_dict[k].append(v.item())
                        
                        current_avg = np.mean(val_losses) if val_losses else float('inf')
                        pbar.set_postfix({
                            'val_loss': f'{loss.item():.4f}',
                            'avg_val_loss': f'{current_avg:.4f}'
                        })
                        
                        # Clear memory
                        del images, targets, batch_loss_dict, loss
                        
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
                    'scaler_state_dict': scaler.state_dict(),
                    'val_loss': avg_val_loss,
                }, checkpoint_path)
                logging.info(f"Saved best model to {checkpoint_path}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= config.early_stopping_patience:
                logging.info(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            # Clear memory at the end of epoch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        config = ModelConfig()
        # Set number of workers based on CPU count
        if config.num_workers == 0:
            config.num_workers = min(4, os.cpu_count() or 1)
        train_model(config)
    except Exception as e:
        logging.error(f"Failed to start training: {str(e)}")
        raise 