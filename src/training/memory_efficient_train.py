import torch
import gc
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
import os
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set CUDA environment variables
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

from src.models.mask_rcnn import DentalCariesMaskRCNN
from src.data.dataset import DentalCariesDataset
from src.configs.model_config import ModelConfig

def safe_cuda_check():
    """Safely check CUDA availability and memory"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please enable GPU in Colab.")
    
    try:
        # Test CUDA with a small tensor
        x = torch.ones(1, device='cuda')
        del x
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        logging.error(f"CUDA initialization error: {str(e)}")
        return False

def safe_to_device(tensor, device):
    """Safely move tensor to device with error handling"""
    try:
        if isinstance(tensor, dict):
            return {k: safe_to_device(v, device) for k, v in tensor.items()}
        if isinstance(tensor, (list, tuple)):
            return [safe_to_device(t, device) for t in tensor]
        if isinstance(tensor, torch.Tensor):
            return tensor.to(device, non_blocking=True)
        return tensor
    except Exception as e:
        logging.error(f"Error moving tensor to device: {str(e)}")
        raise

def clear_memory():
    """Aggressively clear GPU memory"""
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception as e:
        logging.error(f"Error clearing memory: {str(e)}")

def collate_fn(batch):
    """Custom collate function with error handling"""
    try:
        return tuple(zip(*batch))
    except Exception as e:
        logging.error(f"Error in collate_fn: {str(e)}")
        raise

def train_with_gradient_accumulation(model, images, targets, optimizer, scaler, config):
    """Training step with gradient accumulation and error handling"""
    accumulated_loss = 0
    optimizer.zero_grad(set_to_none=True)
    
    try:
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast():
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss = losses / config.gradient_accumulation_steps
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        
        accumulated_loss = loss.item() * config.gradient_accumulation_steps
        
        # Memory cleanup
        del loss_dict, losses, loss
        clear_memory()
        
        return accumulated_loss
        
    except Exception as e:
        logging.error(f"Error in training step: {str(e)}")
        clear_memory()
        return 0

def memory_efficient_training(config):
    """Main training loop with memory optimizations"""
    try:
        # Verify CUDA
        if not safe_cuda_check():
            raise RuntimeError("CUDA initialization failed")
        
        device = torch.device('cuda')
        logging.info(f"Using device: {device}")
        
        # Initialize model on CPU first
        logging.info("Initializing model...")
        model = DentalCariesMaskRCNN(
            num_classes=config.num_classes,
            hidden_dim=config.hidden_dim
        )
        
        # Move model to GPU safely
        model = model.to(device)
        
        # Initialize optimizer and scaler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            eps=config.eps
        )
        
        scaler = torch.cuda.amp.GradScaler()
        
        # Create dataset and dataloader
        train_dataset = DentalCariesDataset(
            config.train_data_path,
            config.train_labels_path,
            is_training=True
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            collate_fn=collate_fn
        )
        
        logging.info("Starting training...")
        for epoch in range(config.num_epochs):
            model.train()
            epoch_losses = []
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
            
            for batch_idx, (images, targets) in enumerate(pbar):
                try:
                    # Move data to GPU safely
                    images = [safe_to_device(img, device) for img in images]
                    targets = [safe_to_device(t, device) for t in targets]
                    
                    # Training step
                    loss = train_with_gradient_accumulation(
                        model, images, targets, optimizer, scaler, config
                    )
                    
                    # Update optimizer
                    if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                    
                    if loss > 0:
                        epoch_losses.append(loss)
                    
                    # Update progress bar
                    avg_loss = np.mean(epoch_losses) if epoch_losses else float('inf')
                    pbar.set_postfix({
                        'loss': f'{loss:.4f}',
                        'avg_loss': f'{avg_loss:.4f}'
                    })
                    
                    # Memory cleanup
                    del images, targets
                    clear_memory()
                    
                    # Add delay if memory pressure is high
                    if torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() > 0.8:
                        time.sleep(0.1)
                    
                except Exception as e:
                    logging.error(f"Error in batch {batch_idx}: {str(e)}")
                    clear_memory()
                    continue
            
            # End of epoch
            avg_epoch_loss = np.mean(epoch_losses) if epoch_losses else float('inf')
            logging.info(f"Epoch {epoch+1} average loss: {avg_epoch_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                try:
                    checkpoint_path = f'checkpoint_epoch_{epoch+1}.pth'
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_epoch_loss,
                    }, checkpoint_path)
                    logging.info(f"Saved checkpoint to {checkpoint_path}")
                except Exception as e:
                    logging.error(f"Error saving checkpoint: {str(e)}")
            
            # Memory cleanup at end of epoch
            clear_memory()
            
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise 