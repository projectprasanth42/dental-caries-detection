import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
    Custom collate function for the DataLoader
    """
    return tuple(zip(*batch))

def train_model(config: ModelConfig):
    try:
        # Enable garbage collection
        gc.enable()
        
        # Set device
        device = torch.device(config.device)
        logging.info(f"Using device: {device}")
        
        # Set torch to use deterministic algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Create datasets
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
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=collate_fn,
            pin_memory=False  # Set to False for CPU training
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=collate_fn,
            pin_memory=False  # Set to False for CPU training
        )

        # Initialize model
        logging.info("Initializing model...")
        model = DentalCariesMaskRCNN(
            num_classes=config.num_classes,
            hidden_dim=config.hidden_dim
        ).to(device)

        # Initialize optimizer
        optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Initialize learning rate scheduler
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=5,
            verbose=True
        )

        # Training loop
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
            
            # Create progress bar
            pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{config.num_epochs}")
            
            for batch_idx, (images, targets) in enumerate(pbar):
                try:
                    # Clear memory
                    if batch_idx % 5 == 0:
                        gc.collect()
                    
                    # Move data to device
                    images = [image.to(device) for image in images]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    
                    # Training step
                    loss, loss_dict = model.train_step(images, targets, optimizer)
                    train_losses.append(loss)
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'loss': f'{loss:.4f}',
                        'avg_loss': f'{np.mean(train_losses):.4f}'
                    })
                    
                    # Log batch information
                    if batch_idx % 5 == 0:  # Reduced logging frequency
                        logging.info(
                            f"Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)} - "
                            f"Loss: {loss:.4f}, Avg Loss: {np.mean(train_losses):.4f}"
                        )
                    
                    # Clear some memory
                    del images, targets, loss_dict
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        
                except Exception as e:
                    logging.error(f"Error in batch {batch_idx}: {str(e)}")
                    continue
            
            avg_train_loss = np.mean(train_losses)
            logging.info(f"Average training loss: {avg_train_loss:.4f}")
            
            # Validation phase
            model.eval()
            val_losses = []
            val_loss_dict = {}
            
            # Create validation progress bar
            pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{config.num_epochs}")
            
            with torch.no_grad():
                for images, targets in pbar:
                    try:
                        # Clear some memory
                        gc.collect()
                        
                        # Move data to device
                        images = [image.to(device) for image in images]
                        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                        
                        # Use validation_step instead of train_step
                        loss, batch_loss_dict = model.validation_step(images, targets)
                        
                        # Only append valid loss values
                        if not torch.isnan(torch.tensor(loss)):
                            val_losses.append(loss)
                            
                            # Accumulate individual losses
                            for k, v in batch_loss_dict.items():
                                if k not in val_loss_dict:
                                    val_loss_dict[k] = []
                                val_loss_dict[k].append(v.item())
                        
                        # Update progress bar with current loss
                        current_avg = np.mean(val_losses) if val_losses else float('inf')
                        pbar.set_postfix({
                            'val_loss': f'{loss:.4f}',
                            'avg_val_loss': f'{current_avg:.4f}'
                        })
                        
                        # Clear memory
                        del images, targets, batch_loss_dict
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        
                    except Exception as e:
                        logging.error(f"Error in validation batch: {str(e)}")
                        continue
            
            # Calculate average validation loss safely
            avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
            
            # Log detailed validation metrics
            logging.info(f"Average validation loss: {avg_val_loss:.4f}")
            if val_loss_dict:
                logging.info("Validation loss components:")
                for k, v in val_loss_dict.items():
                    avg = np.mean(v)
                    logging.info(f"  {k}: {avg:.4f}")
            
            # Learning rate scheduling - only if we have valid losses
            if val_losses:
                scheduler.step(avg_val_loss)
            
            # Save best model - only if we have valid losses
            if val_losses and (avg_val_loss < best_val_loss):
                best_val_loss = avg_val_loss
                patience_counter = 0
                checkpoint_path = os.path.join(save_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
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
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        config = ModelConfig()
        train_model(config)
    except Exception as e:
        logging.error(f"Failed to start training: {str(e)}")
        raise 