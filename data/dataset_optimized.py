import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os
import albumentations as A
from typing import Tuple, Dict, Any
import json
from pathlib import Path
import cv2
import logging

logger = logging.getLogger(__name__)

class BaselineDatasetOptimized(Dataset):
    def __init__(self, 
                 data_dir: str,
                 transform: A.Compose = None,
                 is_train: bool = True,
                 target_size: Tuple[int, int] = (768, 1024)):  # (width, height)
        """
        Optimized Dataset for baseline detection with faster augmentations.
        
        Args:
            data_dir: Directory containing the dataset
            transform: Albumentations transform pipeline
            is_train: Whether this is training data
            target_size: Target size for images (width, height)
        """
        self.data_dir = Path(data_dir)
        self.is_train = is_train
        self.target_size = target_size  # (width, height)
        
        # Get all image files
        self.image_files = sorted(list((self.data_dir / 'images').glob('*.png')))
        
        # Load metadata
        self.metadata = {}
        for img_file in self.image_files:
            meta_file = self.data_dir / 'metadata' / f"{img_file.stem}.json"
            if meta_file.exists():
                with open(meta_file, 'r') as f:
                    self.metadata[img_file.stem] = json.load(f)
        
        # Default transform if none provided
        if transform is None and is_train:
            self.transform = A.Compose([
                # Remove flips and random rotation
                # Keep only small rotations that preserve text orientation
                A.Affine(
                    scale=(0.9, 1.1),
                    translate_percent=(-0.1, 0.1),
                    rotate=(-5, 5),  # Reduced rotation range
                    p=0.5,
                    keep_ratio=True
                ),
                # Optimized elastic transform for slight line variations
                A.ElasticTransform(
                    alpha=30.0,  # Reduced from 60.0
                    sigma=30.0,  # Reduced from 60.0
                    p=0.3
                ),
                # Optimized grid distortion for slight line variations
                A.GridDistortion(
                    num_steps=3,
                    distort_limit=0.1,  # Reduced from 0.2
                    p=0.3
                ),
                # Add slight perspective transform
                A.Perspective(
                    scale=(0.02, 0.05),  # Reduced from (0.05, 0.1)
                    p=0.3
                ),
                # Add random brightness/contrast
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                # Add random gamma
                A.RandomGamma(
                    gamma_limit=(80, 120),
                    p=0.5
                ),
                # Add slight blur effect
                A.GaussianBlur(
                    blur_limit=(3, 5),  # Reduced from (3, 7)
                    p=0.3
                ),
            ], is_check_shapes=False)  # Disable shape checking
        else:
            self.transform = transform
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Load image
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('L')
        image = np.array(image)
        
        # Load label
        label_path = self.data_dir / 'labels' / f"{img_path.stem}.npy"
        label = np.load(label_path)
        
        # Ensure label has correct shape (3, H, W)
        if len(label.shape) == 2:
            label = np.expand_dims(label, axis=0)
        
        # Resize image and label to target size
        image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
        label = np.stack([
            cv2.resize(channel, self.target_size, interpolation=cv2.INTER_NEAREST)
            for channel in label
        ])
        
        # Apply transformations
        if self.transform is not None and self.is_train:
            # Convert label to format expected by albumentations (H, W, C)
            label_for_transform = np.transpose(label, (1, 2, 0))
            
            transformed = self.transform(image=image, mask=label_for_transform)
            image = transformed['image']
            label = transformed['mask']
            
            # Convert label back to (C, H, W)
            label = np.transpose(label, (2, 0, 1))
            
            # Ensure dimensions are correct after augmentation
            if image.shape != self.target_size[::-1]:  # If dimensions were swapped
                image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
                label = np.stack([
                    cv2.resize(channel, self.target_size, interpolation=cv2.INTER_NEAREST)
                    for channel in label
                ])
        
        # Convert to tensor and ensure correct shape (C, H, W)
        image = torch.from_numpy(image).float().unsqueeze(0) / 255.0
        label = torch.from_numpy(label).float()
        
        # Split label into baseline and polygon masks
        baseline_mask = label[:3]  # First 3 channels for baseline
        polygon_mask = label[3:4]  # Next channel for polygon
        
        # Create targets dictionary
        targets = {
            'baseline': baseline_mask,
            'polygon': polygon_mask
        }
        
        # Verify shapes before returning
        assert image.shape[1:] == self.target_size[::-1], f"Image shape mismatch: {image.shape[1:]} vs {self.target_size[::-1]}"
        assert baseline_mask.shape[1:] == self.target_size[::-1], f"Baseline mask shape mismatch: {baseline_mask.shape[1:]} vs {self.target_size[::-1]}"
        assert polygon_mask.shape[1:] == self.target_size[::-1], f"Polygon mask shape mismatch: {polygon_mask.shape[1:]} vs {self.target_size[::-1]}"
        
        return image, targets
    
    def get_metadata(self, idx: int) -> Dict[str, Any]:
        """Get metadata for a specific sample"""
        img_path = self.image_files[idx]
        return self.metadata.get(img_path.stem, {})

def get_dataloaders_optimized(data_dir: str,
                            batch_size: int,
                            num_workers: int,
                            train_val_split: float = 0.8,
                            target_size: Tuple[int, int] = (768, 1024)) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders with optimized augmentations.
    
    Args:
        data_dir: Directory containing the dataset
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        train_val_split: Fraction of data to use for training
        target_size: Target size for images (width, height)
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    logger.info("Loading dataset with optimized augmentations...")
    # Create datasets
    dataset = BaselineDatasetOptimized(data_dir, is_train=True, target_size=target_size)
    logger.info(f"Total dataset size: {len(dataset)} images")
    
    # Split into train and validation
    train_size = int(len(dataset) * train_val_split)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    logger.info(f"Training set size: {train_size} images")
    logger.info(f"Validation set size: {val_size} images")
    
    logger.info("Creating dataloaders...")
    # Create dataloaders with persistent workers
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    logger.info("Dataloaders created successfully")
    return train_loader, val_loader 