import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import time
from typing import Dict, Tuple, List
import argparse
from torch.amp import autocast, GradScaler
import torch.nn.functional as F
import os

from models.unet import BaselineDetectionModel
from data.dataset import get_dataloaders
from configs.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class ContinuousLineLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        # Focus on the baseline channel (index 2)
        pred_baseline = pred[:, 2]
        target_baseline = target[:, 2]
        
        # Calculate horizontal continuity
        pred_h = torch.abs(pred_baseline[:, :, 1:] - pred_baseline[:, :, :-1])
        target_h = torch.abs(target_baseline[:, :, 1:] - target_baseline[:, :, :-1])
        
        # Calculate vertical continuity
        pred_v = torch.abs(pred_baseline[:, 1:, :] - pred_baseline[:, :-1, :])
        target_v = torch.abs(target_baseline[:, 1:, :] - target_baseline[:, :-1, :])
        
        # Combine horizontal and vertical continuity
        continuity_loss = F.mse_loss(pred_h, target_h) + F.mse_loss(pred_v, target_v)
        
        return continuity_loss

class BaselineDetectionTrainer:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize model
        self.model = BaselineDetectionModel(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            base_channels=config.base_channels,
            depth=config.depth
        ).to(self.device)
        
        # Try to enable gradient checkpointing
        try:
            self.model.use_checkpointing()
            logger.info("Gradient checkpointing enabled")
        except Exception as e:
            logger.warning(f"Could not enable gradient checkpointing: {e}")
        
        # Create best_models directory
        self.best_models_dir = os.path.join(config.checkpoint_dir, 'best_models')
        os.makedirs(self.best_models_dir, exist_ok=True)
        
        # Initialize optimizer with lower initial learning rate
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Initialize loss functions with weighted BCE
        # Higher weights for start/end points and baseline to handle class imbalance
        pos_weights = torch.tensor([2.0, 2.0, 3.0]).view(1, 3, 1, 1).to(self.device)  # Increased weight for baseline
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        self.dice_loss = DiceLoss()
        self.continuous_line_loss = ContinuousLineLoss()
        
        # Loss weights - adjusted to make training more challenging
        self.bce_weight = 0.5  # Increased from 0.4
        self.dice_weight = 0.3  # Kept the same
        self.continuous_weight = 0.2  # Decreased from 0.3
        
        # Initialize metrics
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Initialize tensorboard
        self.writer = SummaryWriter(log_dir='runs/baseline_detection')
        
        # Initialize gradient scaler for mixed precision training
        self.scaler = GradScaler()
        
        # Gradient accumulation steps
        self.accumulation_steps = 4
        
        # Scheduler will be initialized in train method
        self.scheduler = None
        
        # Gradient clipping value
        self.clip_value = 1.0
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        epoch_loss = 0
        epoch_dice = 0
        
        # Initialize optimizer
        self.optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc='Training', leave=True)
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            baseline_targets = targets['baseline'].to(self.device)
            polygon_targets = targets['polygon'].to(self.device)
            
            # Ensure tensors have gradients
            images.requires_grad_(True)
            
            # Forward pass with mixed precision
            with autocast(device_type='cuda', dtype=torch.float16):
                output = self.model(images)
                
                # Calculate losses
                bce_loss = self.bce_loss(output, baseline_targets)
                dice_loss = self.dice_loss(output, baseline_targets)
                continuous_loss = self.continuous_line_loss(output, baseline_targets)
                
                # Combine losses with weights
                loss = (self.bce_weight * bce_loss + 
                       self.dice_weight * dice_loss + 
                       self.continuous_weight * continuous_loss) / self.accumulation_steps
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            
            # Update weights if we've accumulated enough gradients
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # Unscale gradients for clipping
                self.scaler.unscale_(self.optimizer)
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_value)
                # Step optimizer and update scaler
                self.scaler.step(self.optimizer)
                self.scaler.update()
                # Update learning rate after optimizer step
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            # Update metrics
            epoch_loss += loss.item() * self.accumulation_steps
            epoch_dice += (1 - dice_loss.item())
            
            # Update progress bar with learning rate and individual losses
            current_lr = self.scheduler.get_last_lr()[0]
            pbar.set_postfix({
                'loss': loss.item() * self.accumulation_steps,
                'bce': bce_loss.item(),
                'dice': dice_loss.item(),
                'cont': continuous_loss.item(),
                'lr': f'{current_lr:.2e}'
            })
            pbar.update()
        
        # Calculate average metrics
        avg_loss = epoch_loss / len(train_loader)
        avg_dice = epoch_dice / len(train_loader)
        
        return {
            'loss': avg_loss,
            'dice': avg_dice
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        val_loss = 0
        val_dice = 0
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc='Validation'):
                images = images.to(self.device)
                baseline_targets = targets['baseline'].to(self.device)
                polygon_targets = targets['polygon'].to(self.device)
                
                # Forward pass with mixed precision
                with autocast(device_type='cuda', dtype=torch.float16):
                    output = self.model(images)  # [B, 3, H, W]
                    baseline_loss = self.bce_loss(output, baseline_targets)
                    loss = baseline_loss
                
                # Update metrics
                val_loss += loss.item()
                val_dice += (1 - loss.item())
        
        # Calculate average metrics
        avg_loss = val_loss / len(val_loader)
        avg_dice = val_dice / len(val_loader)
        
        return {
            'loss': avg_loss,
            'dice': avg_dice
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        # Initialize scheduler with more conservative settings
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            epochs=self.config.num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,  # Longer warmup (30% of training)
            div_factor=10,  # Less aggressive initial lr reduction
            final_div_factor=100,  # Less aggressive final lr reduction
            anneal_strategy='cos'  # Smoother learning rate changes
        )
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Log metrics
            self.writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            self.writer.add_scalar('Dice/train', train_metrics['dice'], epoch)
            self.writer.add_scalar('Dice/val', val_metrics['dice'], epoch)
            
            # Save model
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(epoch, is_best=False)
            
            # Early stopping
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.early_stopping_patience:
                    logger.info("Early stopping triggered")
                    break
            
            # Log progress
            logger.info(f"Train Loss: {train_metrics['loss']:.4f}, Train Dice: {train_metrics['dice']:.4f}")
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Val Dice: {val_metrics['dice']:.4f}")
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'scaler_state_dict': self.scaler.state_dict()
        }
        
        # Save last checkpoint
        torch.save(checkpoint, self.config.last_model_path)
        logger.info(f"Saved checkpoint to {self.config.last_model_path}")
        
        # Save best checkpoint
        if is_best:
            # Save with epoch number in the best_models directory
            best_model_path = os.path.join(
                self.best_models_dir,
                f'best_model_epoch_{epoch}_loss_{self.best_val_loss:.4f}.pth'
            )
            torch.save(checkpoint, best_model_path)
            logger.info(f"Saved best model to {best_model_path}")
            
            # Also save as the current best model
            torch.save(checkpoint, self.config.best_model_path)
            logger.info(f"Updated current best model at {self.config.best_model_path}")

def main():
    parser = argparse.ArgumentParser(description='Train baseline detection model')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Path to the processed dataset directory containing images, labels, and metadata folders')
    parser.add_argument('--batch_size', type=int, default=2,
                      help='Batch size for training (default: 2 for 12GB GPU)')
    parser.add_argument('--num_workers', type=int, default=2,
                      help='Number of workers for data loading')
    parser.add_argument('--num_epochs', type=int, default=100,
                      help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001,  # Lower default learning rate
                      help='Learning rate')
    parser.add_argument('--target_width', type=int, default=768,
                      help='Target width for images')
    parser.add_argument('--target_height', type=int, default=1024,
                      help='Target height for images')
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    
    # Update config with command line arguments
    config.data_dir = args.data_dir
    config.batch_size = args.batch_size
    config.num_workers = args.num_workers
    config.num_epochs = args.num_epochs
    config.learning_rate = args.learning_rate
    
    # Create checkpoint directory
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    logger.info(f"Checkpoints will be saved to: {checkpoint_dir.absolute()}")
    
    logger.info("Initializing training...")
    logger.info(f"Configuration:")
    logger.info(f"  - Batch size: {config.batch_size}")
    logger.info(f"  - Learning rate: {config.learning_rate}")
    logger.info(f"  - Number of epochs: {config.num_epochs}")
    logger.info(f"  - Target image size: {args.target_width}x{args.target_height}")
    logger.info(f"  - Number of workers: {config.num_workers}")
    
    # Create dataloaders with persistent workers
    train_loader, val_loader = get_dataloaders(
        config.data_dir,
        config.batch_size,
        config.num_workers,
        config.train_val_split,
        target_size=(args.target_width, args.target_height)
    )
    
    logger.info("Initializing model and trainer...")
    # Initialize trainer
    trainer = BaselineDetectionTrainer(config)
    
    logger.info("Starting training...")
    # Train model
    trainer.train(train_loader, val_loader)
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main() 