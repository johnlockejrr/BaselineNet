import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def visualize_mask(image_path: str, mask_path: str, output_path: str = None):
    """
    Visualize the mask file and overlay it on the original image.
    The image is automatically resized to match the mask shape.
    
    Args:
        image_path: Path to the original image
        mask_path: Path to the .npy mask file
        output_path: Optional path to save the visualization
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert to RGB for better visualization
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Load mask
    mask = np.load(mask_path)
    
    # Print dimensions for debugging
    print(f"Original image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    
    # Resize image to match mask shape (mask is in HWC format)
    mask_height, mask_width = mask.shape[1], mask.shape[2]
    image = cv2.resize(image, (mask_width, mask_height))
    print(f"Resized image shape: {image.shape}")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # Plot original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title(f'Original Image (Resized to {mask_width}x{mask_height})')
    axes[0, 0].axis('off')
    
    # Plot start points mask
    axes[0, 1].imshow(mask[0], cmap='hot')
    axes[0, 1].set_title('Start Points Mask')
    axes[0, 1].axis('off')
    
    # Plot end points mask
    axes[1, 0].imshow(mask[1], cmap='hot')
    axes[1, 0].set_title('End Points Mask')
    axes[1, 0].axis('off')
    
    # Plot baseline mask
    axes[1, 1].imshow(mask[2], cmap='hot')
    axes[1, 1].set_title('Baseline Mask')
    axes[1, 1].axis('off')
    
    # Create overlay
    overlay = image.copy()
    
    # Scale the mask values to be more visible
    start_points = (mask[0] > 0.5).astype(np.uint8)
    end_points = (mask[1] > 0.5).astype(np.uint8)
    baselines = (mask[2] > 0.5).astype(np.uint8)
    
    # Add start points in red
    overlay[start_points > 0] = [255, 0, 0]
    # Add end points in blue
    overlay[end_points > 0] = [0, 0, 255]
    # Add baselines in green
    overlay[baselines > 0] = [0, 255, 0]
    
    # Create a new figure for the overlay
    plt.figure(figsize=(15, 15))
    plt.imshow(overlay)
    plt.title('Overlay: Red=Start, Blue=End, Green=Baseline')
    plt.axis('off')
    
    if output_path:
        # Save the visualization
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
        print(f"Saved visualization to {output_path}")
    else:
        # Show the plots
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize mask files')
    parser.add_argument('--image_dir', required=True, help='Directory containing images')
    parser.add_argument('--mask_dir', required=True, help='Directory containing .npy mask files')
    parser.add_argument('--output_dir', help='Directory to save visualizations')
    parser.add_argument('--image_name', help='Specific image to visualize (without extension)')
    
    args = parser.parse_args()
    
    # Create output directory if specified
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get list of mask files
    mask_dir = Path(args.mask_dir)
    if args.image_name:
        mask_files = [mask_dir / f"{args.image_name}.npy"]
    else:
        mask_files = list(mask_dir.glob('*.npy'))
    
    # Process each mask file
    for mask_file in mask_files:
        # Get corresponding image file
        image_name = mask_file.stem
        image_file = Path(args.image_dir) / f"{image_name}.png"
        
        if not image_file.exists():
            print(f"Warning: Image file not found for {image_name}")
            continue
        
        # Set output path if specified
        output_path = None
        if args.output_dir:
            output_path = Path(args.output_dir) / f"{image_name}_visualization.png"
        
        try:
            visualize_mask(str(image_file), str(mask_file), str(output_path) if output_path else None)
        except Exception as e:
            print(f"Error processing {image_name}: {str(e)}")

if __name__ == "__main__":
    main() 