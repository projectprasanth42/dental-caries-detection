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
        
        if transform is None:
            self.transform = self.get_default_transforms(is_training)
        else:
            self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].astype(np.float32)
        label = self.labels[idx].astype(np.float32)
        
        # Ensure image is in correct format (H, W, C)
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        
        # Normalize image to [0, 1] range if not already
        if image.max() > 1.0:
            image = image / 255.0
        
        # Get boxes before transformation
        boxes = self._get_boxes(label)
        
        # Prepare data for transformation
        if len(boxes) == 0:
            # If no valid boxes found, create a dummy target
            transformed = self.transform(image=image, mask=label)
            image = transformed['image']
            label = transformed['mask']
            target = {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.int64),
                'masks': torch.zeros((0, label.shape[0], label.shape[1]), dtype=torch.uint8)
            }
        else:
            # Convert boxes to list for albumentations
            boxes_list = boxes.tolist()
            labels_list = [1] * len(boxes)  # Assuming all boxes are caries
            
            # Apply transformations with bounding boxes
            transformed = self.transform(
                image=image,
                mask=label,
                bboxes=boxes_list,
                class_labels=labels_list
            )
            
            image = transformed['image']
            label = transformed['mask']
            transformed_boxes = torch.tensor(transformed['bboxes'], dtype=torch.float32)
            transformed_labels = torch.tensor(transformed['class_labels'], dtype=torch.int64)
            
            target = {
                'boxes': transformed_boxes,
                'labels': transformed_labels,
                'masks': label.unsqueeze(0)
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
            
        boxes = torch.tensor(boxes, dtype=torch.float32)
        return boxes

    @staticmethod
    def get_default_transforms(is_training):
        """Enhanced augmentation pipeline with dental-specific transforms"""
        if is_training:
            return A.Compose([
                # Geometric Transforms with dental considerations
                A.RandomRotate90(p=0.5),
                A.Flip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.2,
                    rotate_limit=45,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.7
                ),
                
                # Dental-specific spatial transforms
                A.OneOf([
                    A.ElasticTransform(
                        alpha=120,
                        sigma=120 * 0.05,
                        border_mode=cv2.BORDER_CONSTANT,
                        p=1.0
                    ),
                    A.GridDistortion(
                        num_steps=5,
                        distort_limit=0.3,
                        border_mode=cv2.BORDER_CONSTANT,
                        p=1.0
                    ),
                    A.OpticalDistortion(
                        distort_limit=0.5,
                        shift_limit=0.5,
                        border_mode=cv2.BORDER_CONSTANT,
                        p=1.0
                    ),
                ], p=0.5),
                
                # Enhanced noise simulation
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                    A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
                ], p=0.4),
                
                # Dental X-ray specific enhancements
                A.OneOf([
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2,
                        contrast_limit=0.2,
                        p=1.0
                    ),
                    A.RandomGamma(
                        gamma_limit=(80, 120),
                        p=1.0
                    ),
                    A.CLAHE(
                        clip_limit=4.0,
                        tile_grid_size=(8, 8),
                        p=1.0
                    ),
                ], p=0.5),
                
                # Dental-specific detail enhancement
                A.OneOf([
                    A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
                    A.UnsharpMask(p=1.0),
                    A.MotionBlur(blur_limit=3, p=1.0),
                ], p=0.3),
                
                # Advanced color adjustments
                A.OneOf([
                    A.HueSaturationValue(
                        hue_shift_limit=20,
                        sat_shift_limit=30,
                        val_shift_limit=20,
                        p=1.0
                    ),
                    A.RGBShift(
                        r_shift_limit=20,
                        g_shift_limit=20,
                        b_shift_limit=20,
                        p=1.0
                    ),
                ], p=0.3),
                
                # Cutout for robustness
                A.CoarseDropout(
                    max_holes=8,
                    max_height=8,
                    max_width=8,
                    fill_value=0,
                    p=0.2
                ),
                
                # Final normalization
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(
                format='pascal_voc',
                min_area=0,
                min_visibility=0.3,
                label_fields=['class_labels']  # Changed to match the transform input
            ))
        else:
            return A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]) 