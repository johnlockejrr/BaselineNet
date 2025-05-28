import torch
import numpy as np
import cv2
from PIL import Image
import argparse
from pathlib import Path
import logging
from models.unet import BaselineDetectionModel
from configs.config import Config
import matplotlib.pyplot as plt
from torchvision import transforms
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_path: str, config: Config) -> BaselineDetectionModel:
    """Load the trained model."""
    model = BaselineDetectionModel(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        base_channels=config.base_channels,
        depth=config.depth
    )
    
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def process_image(image_path: str, target_size: tuple) -> torch.Tensor:
    """Process input image for model prediction."""
    # Load and preprocess image
    image = Image.open(image_path).convert('L')
    image = np.array(image)
    
    # Resize image
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Normalize and convert to tensor
    image = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0) / 255.0
    return image

def visualize_predictions(image: np.ndarray, predictions: torch.Tensor, output_path: str):
    """Visualize model predictions with multiple subplots."""
    # Convert predictions to numpy
    pred_np = predictions.squeeze().cpu().numpy()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle('Model Predictions', fontsize=16)
    
    # Original image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Points channel
    im1 = axes[0, 1].imshow(pred_np[0], cmap='hot')
    axes[0, 1].set_title('Points Channel')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # Baseline channel
    im2 = axes[1, 0].imshow(pred_np[1], cmap='hot')
    axes[1, 0].set_title('Baseline Channel')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # Overlay both points and baselines on original image
    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Apply thresholds
    points = (pred_np[0] > 0.3).astype(np.uint8) * 255  # Higher threshold for points
    baselines = (pred_np[1] > 0.2).astype(np.uint8) * 255  # Lower threshold for baselines
    
    # Color points in green and baselines in red
    overlay[points > 0] = [0, 255, 0]  # Green for points
    overlay[baselines > 0] = [255, 0, 0]  # Red for baselines
    
    # Where both points and baselines are detected, use yellow
    both = np.logical_and(points > 0, baselines > 0)
    overlay[both] = [255, 255, 0]  # Yellow for overlap
    
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('Overlay (Green: Points, Red: Baselines, Yellow: Both)')
    axes[1, 1].axis('off')
    
    # Print prediction statistics
    logger.info(f"Prediction statistics:")
    logger.info(f"Points channel - min: {pred_np[0].min():.3f}, max: {pred_np[0].max():.3f}, mean: {pred_np[0].mean():.3f}")
    logger.info(f"Baseline channel - min: {pred_np[1].min():.3f}, max: {pred_np[1].max():.3f}, mean: {pred_np[1].mean():.3f}")
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize model predictions')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the trained model checkpoint')
    parser.add_argument('--image_path', type=str, required=True,
                      help='Path to the input image')
    parser.add_argument('--output_dir', type=str, default='predictions',
                      help='Directory to save visualization results')
    parser.add_argument('--target_width', type=int, default=768,
                      help='Target width for images')
    parser.add_argument('--target_height', type=int, default=1024,
                      help='Target height for images')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load configuration
    config = Config()
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = load_model(args.model_path, config)
    
    # Process image
    logger.info(f"Processing image {args.image_path}")
    image = process_image(args.image_path, (args.target_width, args.target_height))
    
    # Get predictions
    with torch.no_grad():
        predictions = model(image)
        predictions = torch.sigmoid(predictions)
    
    # Load original image for visualization
    original_image = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)
    original_image = cv2.resize(original_image, (args.target_width, args.target_height))
    
    # Generate output path
    output_path = output_dir / f"{Path(args.image_path).stem}_prediction.png"
    
    # Visualize and save
    logger.info(f"Saving visualization to {output_path}")
    visualize_predictions(original_image, predictions, str(output_path))
    
    logger.info("Done!")

if __name__ == "__main__":
    main() 