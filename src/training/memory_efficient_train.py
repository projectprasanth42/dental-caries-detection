import torch
import gc
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
import os

# Set CUDA environment variables
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

from src.models.mask_rcnn import DentalCariesMaskRCNN
from src.data.dataset import DentalCariesDataset
from src.configs.model_config import ModelConfig

def collate_fn(batch):
    """Custom collate function for the DataLoader"""
    return tuple(zip(*batch))

def safe_to_device(tensor, device):
    """Safely move tensor to device"""
    try:
        return tensor.to(device, non_blocking=True)
    except Exception as e:
        print(f"Error moving tensor to device: {str(e)}")
        return tensor

def train_with_gradient_accumulation(model, images, targets, optimizer, scaler, config):
    """Training step with gradient accumulation and error handling"""
    accumulated_loss = 0
    optimizer.zero_grad(set_to_none=True)
    
    try:
        # Forward pass with mixed precision
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss = losses / config.gradient_accumulation_steps
        
        # Backward pass
        scaler.scale(loss).backward()
        accumulated_loss = loss.item() * config.gradient_accumulation_steps
        
        # Memory cleanup
        del loss_dict, losses, loss
        torch.cuda.empty_cache()
        
        return accumulated_loss
        
    except Exception as e:
        print(f"Error in training step: {str(e)}")
        return 0

def memory_efficient_training(config):
    try:
        # Initialize model on CPU first
        print("Initializing model on CPU...")
        model = DentalCariesMaskRCNN(
            num_classes=config.num_classes,
            hidden_dim=config.hidden_dim
        )
        
        # Move model to GPU carefully
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Moving model to {device}...")
            model = model.to(device)
        else:
            device = torch.device('cpu')
            print("CUDA not available, using CPU")
        
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
            pin_memory=False,
            collate_fn=collate_fn
        )
        
        print("Starting training...")
        for epoch in range(config.num_epochs):
            model.train()
            epoch_losses = []
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
            
            for batch_idx, (images, targets) in enumerate(pbar):
                try:
                    # Move data to device
                    images = [safe_to_device(img, device) for img in images]
                    targets = [{k: safe_to_device(v, device) for k, v in t.items()} for t in targets]
                    
                    # Training step
                    loss = train_with_gradient_accumulation(
                        model, images, targets, optimizer, scaler, config
                    )
                    
                    # Update optimizer
                    if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_val)
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
                    if batch_idx % 2 == 0:
                        torch.cuda.empty_cache()
                        gc.collect()
                    
                except Exception as e:
                    print(f"Error in batch {batch_idx}: {str(e)}")
                    continue
            
            # End of epoch
            avg_epoch_loss = np.mean(epoch_losses) if epoch_losses else float('inf')
            print(f"Epoch {epoch+1} average loss: {avg_epoch_loss:.4f}")
            
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
                    print(f"Saved checkpoint to {checkpoint_path}")
                except Exception as e:
                    print(f"Error saving checkpoint: {str(e)}")
            
            # Memory cleanup
            torch.cuda.empty_cache()
            gc.collect()
    
    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise 