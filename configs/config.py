from dataclasses import dataclass
from typing import Tuple, List, Optional
import torch
import os

class Config:
    def __init__(self):
        # Data parameters
        self.data_dir = 'data/sam_41_mss'
        self.batch_size = 2
        self.num_workers = 2
        self.train_val_split = 0.8
        self.target_size = (768, 1024)  # (width, height)
        
        # Model parameters
        self.in_channels = 1
        self.out_channels = 3  # 3 channels for baseline detection
        self.base_channels = 32
        self.depth = 3
        
        # Training parameters
        self.num_epochs = 100
        self.learning_rate = 0.0001
        self.weight_decay = 0.0001
        self.early_stopping_patience = 10
        self.save_interval = 5
        
        # Device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Checkpoint paths
        self.checkpoint_dir = 'checkpoints'
        self.last_model_path = os.path.join(self.checkpoint_dir, 'last_model.pth')
        self.best_model_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
        
        # Model parameters
        self.warmup_epochs = 5
        self.min_lr = 1e-6
        self.max_lr = 0.001
        self.lr_cycle_momentum = True
        self.lr_cycle_momentum_max = 0.95
        self.lr_cycle_momentum_min = 0.85
        
        # Learning rate schedule
        self.best_models_dir = 'checkpoints/best_models'  # Directory for all best models
        
        # Augmentation parameters
        self.use_augmentation = True
        self.rotation_range = 45.0
        self.width_shift_range = 0.1
        self.height_shift_range = 0.1
        self.horizontal_flip = True
        self.vertical_flip = True
        
        # Add new augmentations for continuous lines
        self.elastic_transform_alpha = 120.0
        self.elastic_transform_sigma = 120.0
        self.grid_distortion_num_steps = 5
        self.grid_distortion_distort_limit = 0.3
        
        # Inference parameters
        self.confidence_threshold = 0.5
        self.min_baseline_length = 50
        self.max_baseline_length = 1000
        
        # Logging
        self.log_interval = 10

    def __post_init__(self):
        # Create directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.best_models_dir, exist_ok=True)
