import torch
from torch.utils.data import DataLoader
import logging
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Test 1: Check if files exist
logging.info("Checking files...")
train_image_path = 'preprocessed_dataset/X_train.npy'
train_label_path = 'preprocessed_dataset/y_train.npy'

if not os.path.exists(train_image_path):
    logging.error(f"Training image file not found: {train_image_path}")
if not os.path.exists(train_label_path):
    logging.error(f"Training label file not found: {train_label_path}")

# Test 2: Dataset
logging.info("\nTesting Dataset...")
try:
    from src.data.dataset import DentalCariesDataset
    
    dataset = DentalCariesDataset(
        image_path=train_image_path,
        label_path=train_label_path,
        is_training=True
    )
    
    # Test single item
    sample = dataset[0]
    logging.info(f"Sample type: {type(sample)}")
    logging.info(f"Sample keys: {sample.keys()}")
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            logging.info(f"{k} shape: {v.shape}")
        else:
            logging.info(f"{k} type: {type(v)}")
    logging.info("Dataset test passed!")

except Exception as e:
    logging.error(f"Dataset test failed: {str(e)}")
    raise

# Test 3: DataLoader
logging.info("\nTesting DataLoader...")
try:
    def collate_fn(batch):
        return batch

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    # Test first batch
    for batch in loader:
        logging.info(f"Batch size: {len(batch)}")
        first_sample = batch[0]
        logging.info(f"First sample keys: {first_sample.keys()}")
        for k, v in first_sample.items():
            if isinstance(v, torch.Tensor):
                logging.info(f"{k} shape: {v.shape}")
            else:
                logging.info(f"{k} type: {type(v)}")
        break
    logging.info("DataLoader test passed!")

except Exception as e:
    logging.error(f"DataLoader test failed: {str(e)}")
    raise

# Test 4: Model
logging.info("\nTesting Model...")
try:
    from src.models.mask_rcnn import DentalCariesMaskRCNN
    from src.configs.model_config import ModelConfig
    
    config = ModelConfig()
    model = DentalCariesMaskRCNN(num_classes=config.num_classes)
    model.train()
    
    # Test with real data from loader
    for batch in loader:
        logging.info("Running model forward pass...")
        # Extract images and targets from batch
        images = [item['image'] for item in batch]  # List of tensors
        
        targets = []
        for item in batch:
            target = {
                'boxes': item['boxes'],
                'labels': item['labels'],
                'masks': item['masks']
            }
            targets.append(target)
        
        outputs = model(images, targets)
        logging.info(f"Model output type: {type(outputs)}")
        if isinstance(outputs, dict):
            logging.info(f"Model output keys: {outputs.keys()}")
            for k, v in outputs.items():
                if isinstance(v, torch.Tensor):
                    logging.info(f"{k} shape: {v.shape}")
                elif isinstance(v, (int, float)):
                    logging.info(f"{k} value: {v}")
                else:
                    logging.info(f"{k} type: {type(v)}")
        else:
            logging.info(f"Model output is not a dictionary, type: {type(outputs)}")
        break
    logging.info("Model test passed!")

except Exception as e:
    logging.error(f"Model test failed: {str(e)}")
    raise 