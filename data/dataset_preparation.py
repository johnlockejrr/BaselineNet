import os
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import cv2
from typing import List, Tuple, Dict, Optional
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_page_xml(xml_path: str) -> List[Dict]:
    """
    Parse PAGE-XML file and extract baseline information.
    Returns a list of dictionaries containing baseline coordinates and text line information.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Get the namespace from the root element
    ns_uri = root.tag.split('}')[0].strip('{')
    ns = {'page': ns_uri}
    
    logger.info(f"Using namespace: {ns_uri}")
    
    baselines = []
    try:
        # Find all TextRegions
        text_regions = root.findall('.//page:TextRegion', ns)
        logger.info(f"Found {len(text_regions)} text regions")
        
        for region in text_regions:
            # Find all TextLines in this region
            text_lines = region.findall('.//page:TextLine', ns)
            logger.info(f"Found {len(text_lines)} text lines in region {region.get('id', 'unknown')}")
            
            for text_line in text_lines:
                # Get baseline
                baseline = text_line.find('.//page:Baseline', ns)
                if baseline is not None:
                    points = baseline.get('points', '').split()
                    coords = []
                    for point in points:
                        x, y = map(float, point.split(','))
                        coords.append([x, y])
                    
                    # Get text line polygon
                    coords_elem = text_line.find('.//page:Coords', ns)
                    if coords_elem is not None:
                        polygon_points = coords_elem.get('points', '').split()
                        polygon = []
                        for point in polygon_points:
                            x, y = map(float, point.split(','))
                            polygon.append([x, y])
                    else:
                        polygon = None
                    
                    # Get text content
                    text_elem = text_line.find('.//page:TextEquiv/page:Unicode', ns)
                    text = text_elem.text if text_elem is not None else ''
                    
                    baselines.append({
                        'baseline': np.array(coords),
                        'polygon': np.array(polygon) if polygon else None,
                        'text': text
                    })
                else:
                    logger.warning(f"No baseline found for text line in region {region.get('id', 'unknown')}")
    
    except Exception as e:
        logger.error(f"Error parsing XML file {xml_path}: {str(e)}")
        # Print the structure of the XML file for debugging
        logger.warning("XML structure:")
        for elem in root.iter():
            logger.warning(f"Tag: {elem.tag}, Attributes: {elem.attrib}")
    
    if not baselines:
        logger.warning(f"No baselines found in {xml_path}")
        # Print the first few lines of the XML file for debugging
        with open(xml_path, 'r') as f:
            logger.warning(f"First 5 lines of XML file:\n{''.join(f.readlines()[:5])}")
    
    return baselines

def create_baseline_mask(image_size: Tuple[int, int], baselines: List[Dict]) -> np.ndarray:
    """
    Create a multi-channel mask for baseline detection.
    Channels: [start points, end points, baseline]
    
    Args:
        image_size: Tuple of (height, width)
        baselines: List of baseline dictionaries
    """
    mask = np.zeros((3, image_size[0], image_size[1]), dtype=np.uint8)
    
    for baseline in baselines:
        coords = baseline['baseline']
        
        # Draw baseline
        cv2.polylines(mask[2], [coords.astype(np.int32)], False, 1, thickness=2)
        
        # Mark start and end points
        cv2.circle(mask[0], tuple(coords[0].astype(np.int32)), 3, 1, -1)
        cv2.circle(mask[1], tuple(coords[-1].astype(np.int32)), 3, 1, -1)
    
    return mask

def find_image_file(xml_path: Path, image_dir: Path) -> Optional[Path]:
    """
    Find corresponding image file for an XML file.
    Checks for both .jpg and .png extensions.
    """
    base_name = xml_path.stem
    for ext in ['.jpg', '.png']:
        image_path = image_dir / f"{base_name}{ext}"
        if image_path.exists():
            return image_path
    return None

def prepare_dataset(xml_dir: str, image_dir: str, output_dir: str, max_height: int = 1024, max_width: int = 768):
    """
    Prepare dataset from PAGE-XML files and corresponding images.
    Maintains aspect ratio while resizing to fit within max dimensions.
    
    Args:
        xml_dir: Directory containing PAGE-XML files
        image_dir: Directory containing corresponding images
        output_dir: Output directory for processed dataset
        max_height: Maximum height of processed images
        max_width: Maximum width of processed images
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / 'images'
    labels_dir = output_dir / 'labels'
    metadata_dir = output_dir / 'metadata'
    
    # Create directories
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each XML file
    for xml_file in Path(xml_dir).glob('*.xml'):
        try:
            # Find corresponding image file
            image_file = find_image_file(xml_file, Path(image_dir))
            if image_file is None:
                logger.warning(f"Image file not found for {xml_file}")
                continue
            
            # Parse XML
            baselines = parse_page_xml(str(xml_file))
            
            # Read image
            image = Image.open(image_file).convert('L')
            original_size = image.size  # (width, height)
            
            # Calculate new size maintaining aspect ratio
            width, height = original_size
            aspect_ratio = width / height
            
            if width > max_width or height > max_height:
                if aspect_ratio > 1:  # Wider than tall
                    new_width = max_width
                    new_height = int(new_width / aspect_ratio)
                else:  # Taller than wide
                    new_height = max_height
                    new_width = int(new_height * aspect_ratio)
                
                # Ensure we don't exceed max dimensions
                if new_width > max_width:
                    new_width = max_width
                    new_height = int(new_width / aspect_ratio)
                if new_height > max_height:
                    new_height = max_height
                    new_width = int(new_height * aspect_ratio)
                
                new_size = (new_width, new_height)
            else:
                new_size = original_size
            
            # Calculate scaling factors
            scale_x = new_size[0] / original_size[0]
            scale_y = new_size[1] / original_size[1]
            
            # Scale baseline coordinates
            scaled_baselines = []
            for baseline in baselines:
                scaled_coords = baseline['baseline'].copy()
                scaled_coords[:, 0] *= scale_x  # Scale x coordinates
                scaled_coords[:, 1] *= scale_y  # Scale y coordinates
                
                scaled_polygon = None
                if baseline['polygon'] is not None:
                    scaled_polygon = baseline['polygon'].copy()
                    scaled_polygon[:, 0] *= scale_x
                    scaled_polygon[:, 1] *= scale_y
                
                scaled_baselines.append({
                    'baseline': scaled_coords,
                    'polygon': scaled_polygon,
                    'text': baseline['text']
                })
            
            # Resize image
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Create mask with correct dimensions (height, width)
            mask = create_baseline_mask((new_size[1], new_size[0]), scaled_baselines)
            
            # Save processed data
            image.save(images_dir / f"{xml_file.stem}.png")
            np.save(labels_dir / f"{xml_file.stem}.npy", mask)
            
            # Save metadata
            metadata = {
                'original_size': original_size,
                'processed_size': new_size,
                'baselines': [{
                    'baseline': bl['baseline'].tolist(),
                    'polygon': bl['polygon'].tolist() if bl['polygon'] is not None else None,
                    'text': bl['text']
                } for bl in scaled_baselines]
            }
            with open(metadata_dir / f"{xml_file.stem}.json", 'w') as f:
                json.dump(metadata, f)
            
            logger.info(f"Processed {xml_file.stem} - Original: {original_size}, New: {new_size}")
            
        except Exception as e:
            logger.error(f"Error processing {xml_file}: {str(e)}")
            continue

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare dataset from PAGE-XML files')
    parser.add_argument('--xml_dir', required=True, help='Directory containing PAGE-XML files')
    parser.add_argument('--image_dir', required=True, help='Directory containing corresponding images')
    parser.add_argument('--output_dir', required=True, help='Output directory for processed dataset')
    parser.add_argument('--max_height', type=int, default=1024, help='Maximum height of processed images')
    parser.add_argument('--max_width', type=int, default=768, help='Maximum width of processed images')
    
    args = parser.parse_args()
    
    prepare_dataset(
        args.xml_dir,
        args.image_dir,
        args.output_dir,
        args.max_height,
        args.max_width
    ) 