import os
import gc
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm.auto import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DentalDataset(Dataset):
    def __init__(self, images_path, labels_path):
        self.images = np.load(images_path)
        self.labels = np.load(labels_path)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx]).float()
        target = {
            'boxes': torch.from_numpy(self.labels[idx]['boxes']).float(),
            'labels': torch.from_numpy(self.labels[idx]['labels']).long(),
            'masks': torch.from_numpy(self.labels[idx]['masks']).float()
        }
        return image, target

def collate_fn(batch):
    return tuple(zip(*batch))

def create_model(num_classes=3):
    # Use a lighter backbone for better memory efficiency
    model = torchvision.models.detection.maskrcnn_resnet34_fpn(
        pretrained=True,
        box_detections_per_img=100,
        rpn_post_nms_top_n_train=1000,
        rpn_post_nms_top_n_test=500
    )
    
    # Modify the box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    
    # Modify the mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask, 128, num_classes
    )
    
    return model

def train_one_epoch(model, optimizer, data_loader, device, scaler):
    model.train()
    total_loss = 0
    
    pbar = tqdm(data_loader, desc='Training')
    for images, targets in pbar:
        # Clear memory
        torch.cuda.empty_cache()
        
        try:
            # Move data to GPU
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Update progress bar
            total_loss += losses.item()
            pbar.set_postfix({'loss': f'{losses.item():.4f}'})
            
            # Clear memory
            del images, targets, losses, loss_dict
            torch.cuda.empty_cache()
            
        except Exception as e:
            logging.error(f"Error in batch: {str(e)}")
            continue
    
    return total_loss / len(data_loader)

def main():
    # Set memory-efficient CUDA settings
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Clear initial memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Set paths
    data_dir = '/content/drive/MyDrive/dental_caries_dataset'
    train_images = os.path.join(data_dir, 'X_train.npy')
    train_labels = os.path.join(data_dir, 'y_train.npy')
    
    # Create dataset and dataloader
    dataset = DentalDataset(train_images, train_labels)
    data_loader = DataLoader(
        dataset,
        batch_size=1,  # Small batch size for memory efficiency
        shuffle=True,
        num_workers=0,  # No multiprocessing for stability
        collate_fn=collate_fn
    )
    
    # Initialize model and move to GPU
    device = torch.device('cuda')
    model = create_model().to(device)
    
    # Initialize optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001)
    scaler = torch.cuda.amp.GradScaler()
    
    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        try:
            avg_loss = train_one_epoch(model, optimizer, data_loader, device, scaler)
            print(f"Average loss: {avg_loss:.4f}")
            
            # Save checkpoint to CPU memory first
            if (epoch + 1) % 5 == 0:
                # Move model to CPU for saving
                model.cpu()
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss
                }
                
                save_path = os.path.join(data_dir, f'checkpoint_epoch_{epoch+1}.pth')
                torch.save(checkpoint, save_path)
                print(f"Saved checkpoint to {save_path}")
                
                # Move model back to GPU
                model.cuda()
                del checkpoint
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error in epoch {epoch+1}: {str(e)}")
            continue

if __name__ == "__main__":
    main() 