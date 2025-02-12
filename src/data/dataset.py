import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy import ndimage
import cv2

class DentalCariesDataset(Dataset):
    def __init__(self, image_path, label_path, transform=None, is_training=True):
        """
        Args:
            image_path (str): Path to the .npy file containing images
            label_path (str): Path to the .npy file containing labels
            transform (callable, optional): Optional transform to be applied on a sample
            is_training (bool): Whether this is for training or validation/testing
        """
        print(f"Loading data from {image_path} and {label_path}")
        self.images = np.load(image_path)
        self.labels = np.load(label_path)
        self.is_training = is_training
        
        print(f"Loaded images shape: {self.images.shape}")
        print(f"Loaded labels shape: {self.labels.shape}")
        
        # Create transforms
        self.transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        if is_training:
            self.transform = A.Compose([
                A.RandomRotate90(p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image and label
        image = self.images[idx].astype(np.float32)
        label = self.labels[idx].astype(np.float32)
        
        # Ensure image is in correct format (H, W, C)
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        
        # Normalize image to [0, 1] range if not already
        if image.max() > 1.0:
            image = image / 255.0
        
        # Convert label to uint8 for mask
        mask = (label > 0).astype(np.uint8)
        
        # Apply transforms
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']
        
        # Get boxes
        boxes = self._get_boxes(label)
        
        # Create target dictionary
        target = {
            'boxes': boxes,
            'labels': torch.ones(boxes.shape[0], dtype=torch.int64),
            'masks': torch.as_tensor(mask[None], dtype=torch.uint8)
        }
        
        return image, target

    def _get_boxes(self, mask):
        """
        Extract bounding boxes from mask with additional validation
        to ensure boxes have positive height and width
        """
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()
        
        if mask.sum() == 0:
            return torch.zeros((0, 4), dtype=torch.float32)
        
        # Find connected components in the mask
        labeled_mask, num_components = ndimage.label(mask)
        boxes = []
        
        for component in range(1, num_components + 1):
            # Get coordinates for this component
            component_mask = labeled_mask == component
            rows = np.any(component_mask, axis=1)
            cols = np.any(component_mask, axis=0)
            
            if rows.sum() == 0 or cols.sum() == 0:
                continue
                
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            
            # Ensure box has positive width and height
            if rmax > rmin and cmax > cmin:
                # Add small padding to ensure positive width/height
                rmin = max(0, rmin - 1)
                cmin = max(0, cmin - 1)
                rmax = min(mask.shape[0] - 1, rmax + 1)
                cmax = min(mask.shape[1] - 1, cmax + 1)
                
                boxes.append([float(cmin), float(rmin), float(cmax), float(rmax)])
        
        if not boxes:
            return torch.zeros((0, 4), dtype=torch.float32)
            
        return torch.tensor(boxes, dtype=torch.float32) 