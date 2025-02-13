import torch
import gc
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader

from src.models.mask_rcnn import DentalCariesMaskRCNN
from src.data.dataset import DentalCariesDataset
from src.configs.model_config import ModelConfig

def collate_fn(batch):
    """Custom collate function for the DataLoader"""
    images = []
    targets = []
    for image, target in batch:
        images.append(image)
        targets.append(target)
    return images, targets

def train_with_gradient_accumulation(model, images, targets, optimizer, scaler, config):
    """Training step with gradient accumulation"""
    # Split batch into smaller chunks
    chunk_size = len(images) // config.gradient_accumulation_steps
    accumulated_loss = 0
    
    optimizer.zero_grad()
    
    for i in range(config.gradient_accumulation_steps):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        
        chunk_images = images[start_idx:end_idx]
        chunk_targets = targets[start_idx:end_idx]
        
        with torch.cuda.amp.autocast():
            loss, loss_dict = model.train_step(chunk_images, chunk_targets, optimizer)
            loss = loss / config.gradient_accumulation_steps
        
        scaler.scale(loss).backward()
        accumulated_loss += loss.item() * config.gradient_accumulation_steps
        
        # Clear memory
        del chunk_images, chunk_targets, loss, loss_dict
        torch.cuda.empty_cache()
    
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    
    return accumulated_loss

def memory_efficient_training(config):
    try:
        # Enable garbage collection
        gc.enable()
        
        # Initialize model, optimizer, etc.
        device = torch.device(config.device)
        model = DentalCariesMaskRCNN(
            num_classes=config.num_classes,
            hidden_dim=config.hidden_dim
        ).to(device)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        scaler = torch.cuda.amp.GradScaler()
        
        # Create datasets and dataloaders
        train_dataset = DentalCariesDataset(
            config.train_data_path,
            config.train_labels_path,
            is_training=True
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size * config.gradient_accumulation_steps,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        # Training loop
        for epoch in range(config.num_epochs):
            model.train()
            epoch_losses = []
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
            
            for batch_idx, (images, targets) in enumerate(pbar):
                try:
                    # Clear memory
                    if batch_idx % 2 == 0:
                        gc.collect()
                        torch.cuda.empty_cache()
                    
                    # Move data to GPU
                    images = [image.to(device) for image in images]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    
                    # Training step with gradient accumulation
                    loss = train_with_gradient_accumulation(
                        model, images, targets, optimizer, scaler, config
                    )
                    
                    epoch_losses.append(loss)
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'loss': f'{loss:.4f}',
                        'avg_loss': f'{np.mean(epoch_losses):.4f}'
                    })
                    
                    # Clear memory
                    del images, targets
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"Error in batch {batch_idx}: {str(e)}")
                    continue
            
            print(f"Epoch {epoch+1} average loss: {np.mean(epoch_losses):.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, f'checkpoint_epoch_{epoch+1}.pth')
    
    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise 