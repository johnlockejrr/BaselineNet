import torch
import argparse
from models.unet import BaselineDetectionModel
from configs.config import Config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_checkpoint(checkpoint_path: str):
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Print basic checkpoint info
    logger.info(f"\nCheckpoint Information:")
    logger.info(f"Epoch: {checkpoint['epoch']}")
    logger.info(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
    
    # Print model state dict info
    state_dict = checkpoint['model_state_dict']
    logger.info(f"\nModel Architecture Information:")
    logger.info(f"Number of parameters: {sum(p.numel() for p in state_dict.values()):,}")
    logger.info(f"Number of layers: {len(state_dict)}")
    
    # Print layer-wise information
    logger.info("\nLayer-wise Information:")
    for name, param in state_dict.items():
        logger.info(f"Layer: {name}")
        logger.info(f"  Shape: {param.shape}")
        logger.info(f"  Parameters: {param.numel():,}")
        
        # Only calculate statistics for floating point parameters
        if param.dtype in [torch.float32, torch.float16, torch.float64]:
            logger.info(f"  Mean: {param.mean().item():.4f}")
            logger.info(f"  Std: {param.std().item():.4f}")
            logger.info(f"  Min: {param.min().item():.4f}")
            logger.info(f"  Max: {param.max().item():.4f}")
        else:
            logger.info(f"  Type: {param.dtype}")
            if param.numel() == 1:
                logger.info(f"  Value: {param.item()}")
    
    # Print optimizer state if available
    if 'optimizer_state_dict' in checkpoint:
        logger.info("\nOptimizer Information:")
        opt_state = checkpoint['optimizer_state_dict']
        logger.info(f"Optimizer type: {type(opt_state).__name__}")
        logger.info(f"Learning rate: {opt_state['param_groups'][0]['lr']:.2e}")
        logger.info(f"Weight decay: {opt_state['param_groups'][0]['weight_decay']:.2e}")
    
    # Print scheduler state if available
    if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
        logger.info("\nScheduler Information:")
        scheduler_state = checkpoint['scheduler_state_dict']
        logger.info(f"Scheduler type: {type(scheduler_state).__name__}")
        if 'last_epoch' in scheduler_state:
            logger.info(f"Last epoch: {scheduler_state['last_epoch']}")

def main():
    parser = argparse.ArgumentParser(description='Analyze model checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to the model checkpoint file')
    args = parser.parse_args()
    
    analyze_checkpoint(args.checkpoint)

if __name__ == "__main__":
    main() 