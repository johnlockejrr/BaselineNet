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
import psutil
import signal
import sys

from models.unet import BaselineDetectionModel
from data.dataset_optimized import get_dataloaders_optimized
from configs.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
        self.batch_counter = 0
        self.eps = 1e-6  # Small epsilon for numerical stability
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        
        # Calculate intersection and union
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        
        # Add small epsilon to avoid division by zero and improve numerical stability
        dice = (2. * intersection + self.smooth) / (union + self.smooth + self.eps)
        
        # Calculate mean Dice score per channel
        dice_per_channel = dice.mean(dim=0)
        
        # Log channel-wise Dice scores every 50 batches
        if self.training:
            self.batch_counter += 1
            if self.batch_counter % 50 == 0:
                logger.debug(f"Dice scores per channel - Points: {dice_per_channel[0]:.4f}, Baseline: {dice_per_channel[1]:.4f}")
        
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

def cleanup_existing_processes():
    """Clean up any existing training processes"""
    current_pid = os.getpid()
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'train_optimized.py' in ' '.join(proc.info['cmdline'] or []) and proc.info['pid'] != current_pid:
                logger.warning(f"Found existing training process (PID: {proc.info['pid']}). Terminating...")
                os.kill(proc.info['pid'], signal.SIGTERM)
                time.sleep(1)  # Give it time to clean up
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

def signal_handler(signum, frame):
    """Handle termination signals gracefully"""
    logger.info("Received termination signal. Cleaning up...")
    # Clean up GPU memory
    torch.cuda.empty_cache()
    # Close tensorboard writer
    if hasattr(trainer, 'writer'):
        trainer.writer.close()
    sys.exit(0)

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
        pos_weights = torch.tensor([10.0, 10.0, 20.0]).view(1, 3, 1, 1).to(self.device)  # Reduced from [20, 20, 50]
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        self.dice_loss = DiceLoss()
        self.continuous_line_loss = ContinuousLineLoss()
        
        # Adjust loss weights to focus more on actual predictions
        self.bce_weight = 0.2  # Reduced from 0.3
        self.dice_weight = 0.6  # Increased from 0.5
        self.continuous_weight = 0.2  # Kept the same
        
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
        self.clip_value = 0.5
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        epoch_loss = 0
        epoch_dice = 0
        
        try:
            # Initialize optimizer
            self.optimizer.zero_grad()
            
            # Get initial learning rate from optimizer
            current_lr = self.optimizer.param_groups[0]['lr']
            
            pbar = tqdm(train_loader, desc='Training', leave=True)
            for batch_idx, (images, targets) in enumerate(pbar):
                try:
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
                    
                    # Check for NaN loss
                    if torch.isnan(loss):
                        logger.error("NaN loss detected! Skipping batch...")
                        continue
                    
                    # Backward pass with gradient scaling
                    self.scaler.scale(loss).backward()
                    
                    # Update weights if we've accumulated enough gradients
                    if (batch_idx + 1) % self.accumulation_steps == 0:
                        # Unscale gradients for clipping
                        self.scaler.unscale_(self.optimizer)
                        # Clip gradients
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_value)
                        
                        # First step the optimizer
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        
                        # Then step the scheduler AFTER optimizer step
                        self.scheduler.step()
                        
                        # Update learning rate from optimizer after scheduler step
                        current_lr = self.optimizer.param_groups[0]['lr']
                        
                        # Finally zero the gradients
                        self.optimizer.zero_grad()
                    
                    # Update metrics
                    epoch_loss += loss.item() * self.accumulation_steps
                    epoch_dice += (1 - dice_loss.item())
                    
                    # Update progress bar with learning rate and individual losses
                    pbar.set_postfix({
                        'loss': loss.item() * self.accumulation_steps,
                        'bce': bce_loss.item(),
                        'dice': dice_loss.item(),
                        'cont': continuous_loss.item(),
                        'lr': f'{current_lr:.2e}'
                    })
                    pbar.update()
                except Exception as e:
                    logger.error(f"Error in batch {batch_idx}: {str(e)}")
            
            # Calculate average metrics
            avg_loss = epoch_loss / len(train_loader)
            avg_dice = epoch_dice / len(train_loader)
            
            return {
                'loss': avg_loss,
                'dice': avg_dice
            }
        except Exception as e:
            logger.error(f"Training failed with error: {str(e)}")
            raise
    
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
                    
                    # Calculate all losses like in training
                    bce_loss = self.bce_loss(output, baseline_targets)
                    dice_loss = self.dice_loss(output, baseline_targets)
                    continuous_loss = self.continuous_line_loss(output, baseline_targets)
                    
                    # Combine losses with weights
                    loss = (self.bce_weight * bce_loss + 
                           self.dice_weight * dice_loss + 
                           self.continuous_weight * continuous_loss)
                
                # Update metrics
                val_loss += loss.item()
                val_dice += (1 - dice_loss.item())
        
        # Calculate average metrics
        avg_loss = val_loss / len(val_loader)
        avg_dice = val_dice / len(val_loader)
        
        return {
            'loss': avg_loss,
            'dice': avg_dice
        }
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load checkpoint and return the epoch number"""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if it exists
        if checkpoint['scheduler_state_dict'] is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler state
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Load best validation loss
        self.best_val_loss = checkpoint['best_val_loss']
        
        # Return the epoch number
        return checkpoint['epoch']

    def train(self, train_loader: DataLoader, val_loader: DataLoader, start_epoch: int = 0):
        # Calculate exact number of steps
        steps_per_epoch = len(train_loader) // self.accumulation_steps
        total_steps = self.config.num_epochs * steps_per_epoch
        
        # Initialize scheduler with correct number of steps
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate * 0.5,  # Reduced max learning rate
            total_steps=total_steps,
            pct_start=0.3,
            div_factor=25,  # Increased from 10 for more conservative start
            final_div_factor=100,
            anneal_strategy='cos',
            three_phase=True
        )
        
        # Do a dummy optimizer step before scheduler step
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        
        logger.info(f"Initialized scheduler with {total_steps} total steps "
                   f"({steps_per_epoch} steps per epoch with accumulation)")
        
        try:
            for epoch in range(start_epoch, self.config.num_epochs):
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
                
                # Clear GPU cache periodically
                if epoch % 5 == 0:
                    torch.cuda.empty_cache()
                    
        except Exception as e:
            logger.error(f"Training failed with error: {str(e)}")
            raise
        finally:
            # Clean up
            self.writer.close()
            torch.cuda.empty_cache()
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        # Save only essential information
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'scaler_state_dict': self.scaler.state_dict()
        }
        
        # Save last checkpoint with reduced precision
        torch.save(checkpoint, self.config.last_model_path, _use_new_zipfile_serialization=False)
        logger.info(f"Saved checkpoint to {self.config.last_model_path}")
        
        # Save best checkpoint
        if is_best:
            # Save with epoch number in the best_models directory
            best_model_path = os.path.join(
                self.best_models_dir,
                f'best_model_epoch_{epoch}_loss_{self.best_val_loss:.4f}.pth'
            )
            # Save best model with reduced precision
            torch.save(checkpoint, best_model_path, _use_new_zipfile_serialization=False)
            logger.info(f"Saved best model to {best_model_path}")
            
            # Also save as the current best model
            torch.save(checkpoint, self.config.best_model_path, _use_new_zipfile_serialization=False)
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
    parser.add_argument('--resume', action='store_true',
                      help='Resume training from the last checkpoint')
    
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
    train_loader, val_loader = get_dataloaders_optimized(
        config.data_dir,
        config.batch_size,
        config.num_workers,
        config.train_val_split,
        target_size=(args.target_width, args.target_height)
    )
    
    logger.info("Initializing model and trainer...")
    # Initialize trainer
    trainer = BaselineDetectionTrainer(config)
    
    # Resume from checkpoint if requested
    start_epoch = 0
    if args.resume and os.path.exists(config.last_model_path):
        start_epoch = trainer.load_checkpoint(config.last_model_path)
        logger.info(f"Resuming training from epoch {start_epoch + 1}")
    
    logger.info("Starting training...")
    # Train model
    trainer.train(train_loader, val_loader, start_epoch)
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main() 
