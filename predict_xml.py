import torch
import cv2
import numpy as np
from pathlib import Path
import argparse
from models.unet import BaselineDetectionModel
from configs.config import Config
import logging
from typing import List, Tuple, Optional, Dict, Sequence, Union, Any
import json
from PIL import Image
from skimage.morphology import skeletonize
from skimage.measure import label
import networkx as nx
from shapely import geometry as geom
from scipy.ndimage import gaussian_filter, sobel
from scipy.signal import convolve2d
from skimage.graph import MCP_Connect
from utils.page_xml_generator import PAGE_XML_Generator
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LineMCP(MCP_Connect):
    """Modified MCP_Connect class for line following."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.connections = []

    def create_connection(self, id1, id2, pos1, pos2, cost1, cost2):
        self.connections.append((pos1, pos2))

    def get_connections(self):
        return self.connections

    def goal_reached(self, int_index, float_cumcost):
        return True

def _calc_seam(baseline, polygon, angle, im_feats, bias=150):
    """Calculate optimal path between baselines using seam carving."""
    # Create direction vectors
    dir_vec = np.array([np.cos(angle), np.sin(angle)])
    perp_vec = np.array([-np.sin(angle), np.cos(angle)])
    
    # Calculate ROI bounds
    bounds = polygon.bounds
    min_x, min_y, max_x, max_y = bounds
    
    # Create cost image
    cost = np.ones_like(im_feats, dtype=np.float32)
    
    # Add gradient cost
    cost[im_feats > 0] = 0.1
    
    # Create distance transform from baseline
    mask = np.ones_like(cost)
    for i in range(len(baseline)-1):
        cv2.line(mask, tuple(baseline[i]), tuple(baseline[i+1]), 0, 1)
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    
    # Add distance cost with bias
    # For positive bias, push up; for negative bias, push down
    if bias > 0:
        cost += dist * bias / np.max(dist)
    else:
        cost -= dist * abs(bias) / np.max(dist)
    
    # Add directional bias using perpendicular vector
    center = np.mean(baseline, axis=0)
    y, x = np.ogrid[:cost.shape[0], :cost.shape[1]]
    dx = x - center[0]
    dy = y - center[1]
    norm = np.sqrt(dx*dx + dy*dy)
    
    # Use perpendicular vector to push in opposite directions
    if bias > 0:
        # Push upward
        dot_product = (dx * perp_vec[0] + dy * perp_vec[1]) / (norm + 1e-6)
        cost += (1 - dot_product) * 50
    else:
        # Push downward
        dot_product = (dx * perp_vec[0] + dy * perp_vec[1]) / (norm + 1e-6)
        cost += (1 + dot_product) * 50
    
    # Ensure start and end points are within bounds
    start = np.array([int(min_x), int(min_y)])
    end = np.array([int(max_x), int(max_y)])
    
    # Clip coordinates to image bounds
    start = np.clip(start, [0, 0], [cost.shape[1]-1, cost.shape[0]-1])
    end = np.clip(end, [0, 0], [cost.shape[1]-1, cost.shape[0]-1])
    
    try:
        # Initialize MCP with more lenient parameters
        mcp = LineMCP(cost, fully_connected=True)
        
        # Calculate path
        mcp.find_costs([start], [end])
        path = mcp.traceback(end)
        
        if len(path) < 2:
            # If path is too short, create a simple offset
            offset = 20
            # Use bias to determine direction
            if bias > 0:
                # Push upward
                path = baseline + perp_vec * offset
            else:
                # Push downward
                path = baseline - perp_vec * offset
            path = np.clip(path, [0, 0], [cost.shape[1]-1, cost.shape[0]-1])
        
        # Smooth the path
        path = np.array(path)
        if len(path) > 2:
            path = cv2.approxPolyDP(path.reshape(-1, 1, 2), 2, False).reshape(-1, 2)
        
        return path
    except Exception as e:
        # Fallback: create a simple offset path
        offset = 20
        # Use bias to determine direction
        if bias > 0:
            # Push upward
            path = baseline + perp_vec * offset
        else:
            # Push downward
            path = baseline - perp_vec * offset
        path = np.clip(path, [0, 0], [cost.shape[1]-1, cost.shape[0]-1])
        return path

def _extract_patch(env_up, env_bottom, baseline, offset_baseline, end_points, dir_vec, topline, offset, im_feats, bounds):
    """Extract patch for polygon generation."""
    # Calculate patch bounds
    min_x, min_y, max_x, max_y = bounds
    
    # Create patch mask
    patch = np.zeros_like(im_feats, dtype=np.uint8)
    
    # Ensure points are in correct format for OpenCV
    baseline = baseline.reshape(-1, 1, 2).astype(np.int32)
    offset_baseline = offset_baseline.reshape(-1, 1, 2).astype(np.int32)
    
    # Draw baseline and offset lines
    cv2.polylines(patch, [baseline], False, 1, 1)
    cv2.polylines(patch, [offset_baseline], False, 1, 1)
    
    # Create polygon points
    poly_points = np.vstack([baseline.reshape(-1, 2), offset_baseline.reshape(-1, 2)[::-1]])
    poly_points = poly_points.reshape(-1, 1, 2).astype(np.int32)
    
    # Fill between lines
    cv2.fillPoly(patch, [poly_points], 1)
    
    # Apply mask to features
    patch_feats = im_feats * patch
    
    return patch_feats

def _calc_roi(line, bounds, baselines, suppl_obj, p_dir):
    """Calculate region of interest for polygon generation."""
    # Get line bounds
    min_x, min_y, max_x, max_y = bounds
    
    # Calculate ROI size
    width = max_x - min_x
    height = max_y - min_y
    
    # Add padding
    pad = 10
    roi = (int(min_x - pad), int(min_y - pad),
           int(width + 2*pad), int(height + 2*pad))
    
    return roi

def calculate_polygonal_environment(im: Image.Image,
                                 baselines: Sequence[Sequence[Tuple[int, int]]],
                                 scale: Tuple[int, int] = None,
                                 topline: bool = False) -> List[Optional[np.ndarray]]:
    """Calculate polygonal environment around baselines."""
    # Process image for feature extraction
    im_array = np.array(im)
    if len(im_array.shape) == 3:
        im_array = cv2.cvtColor(im_array, cv2.COLOR_RGB2GRAY)
    im_feats = gaussian_filter(im_array, sigma=1.0)
    im_feats = sobel(im_feats)
    
    # Process each baseline
    polygons = []
    for i, baseline in enumerate(baselines):
        try:
            # Convert baseline to numpy array
            baseline = np.array(baseline)
            
            # Calculate baseline angle
            if len(baseline) > 1:
                angle = np.arctan2(baseline[-1][1] - baseline[0][1],
                                 baseline[-1][0] - baseline[0][0])
            else:
                angle = 0
            
            # Create buffer around baseline
            baseline_poly = geom.LineString(baseline).buffer(10)
            
            # Calculate top and bottom seams
            top_seam = _calc_seam(baseline, baseline_poly, angle, im_feats, bias=200)
            bottom_seam = _calc_seam(baseline, baseline_poly, angle, im_feats, bias=-200)
            
            # Ensure seams are on correct side
            top_dir = _ensure_correct_side(baseline, top_seam)
            bottom_dir = _ensure_correct_side(baseline, bottom_seam)
            
            if top_dir > bottom_dir:
                top_seam, bottom_seam = bottom_seam, top_seam
            
            if len(top_seam) > 2 and len(bottom_seam) > 2:
                # Reshape seams for OpenCV
                baseline = baseline.reshape(-1, 1, 2).astype(np.int32)
                top_seam = top_seam.reshape(-1, 1, 2).astype(np.int32)
                bottom_seam = bottom_seam.reshape(-1, 1, 2).astype(np.int32)
                
                # Create polygon points
                poly_points = np.vstack([
                    top_seam.reshape(-1, 2),
                    bottom_seam.reshape(-1, 2)[::-1]
                ])
                
                # Ensure polygon is closed
                if not np.array_equal(poly_points[0], poly_points[-1]):
                    poly_points = np.vstack([poly_points, poly_points[0]])
                
                # Reshape for OpenCV
                poly_points = poly_points.reshape(-1, 1, 2).astype(np.int32)
                polygons.append(poly_points)
            else:
                # Fallback: create simple offset polygon
                offset = 20
                perp_vec = np.array([-np.sin(angle), np.cos(angle)])
                top = baseline - perp_vec * offset
                bottom = baseline + perp_vec * offset
                
                # Create polygon points
                poly_points = np.vstack([top, bottom[::-1]])
                
                # Ensure polygon is closed
                if not np.array_equal(poly_points[0], poly_points[-1]):
                    poly_points = np.vstack([poly_points, poly_points[0]])
                
                # Reshape for OpenCV
                poly_points = poly_points.reshape(-1, 1, 2).astype(np.int32)
                polygons.append(poly_points)
                
        except Exception as e:
            logger.warning(f"Polygonizer failed on line {i}: {e}")
            # Fallback: create simple offset polygon
            offset = 20
            perp_vec = np.array([-np.sin(angle), np.cos(angle)])
            top = baseline - perp_vec * offset
            bottom = baseline + perp_vec * offset
            
            # Create polygon points
            poly_points = np.vstack([top, bottom[::-1]])
            
            # Ensure polygon is closed
            if not np.array_equal(poly_points[0], poly_points[-1]):
                poly_points = np.vstack([poly_points, poly_points[0]])
            
            # Reshape for OpenCV
            poly_points = poly_points.reshape(-1, 1, 2).astype(np.int32)
            polygons.append(poly_points)
    
    return polygons

class BaselineDetector:
    def __init__(self, model_path: str, target_size: tuple = (1024, 768)):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_size = target_size
        
        # Load configuration
        self.config = Config()
        
        # Initialize model
        self.model = BaselineDetectionModel(
            in_channels=self.config.in_channels,
            out_channels=self.config.out_channels,
            base_channels=self.config.base_channels,
            depth=self.config.depth
        ).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Using device: {self.device}")

    def ensure_consistent_direction(self, baselines: List[np.ndarray], polygons: List[np.ndarray], 
                                  direction: str = 'ltr') -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Ensure all baselines and polygons follow specified direction."""
        processed_baselines = []
        processed_polygons = []
    
        for baseline, polygon in zip(baselines, polygons):
            # Process baseline
            if len(baseline) > 1:
                x_coords = baseline[:, 0]
                if (direction == 'ltr' and x_coords[0] > x_coords[-1]) or \
                   (direction == 'rtl' and x_coords[0] < x_coords[-1]):
                    baseline = np.flip(baseline, axis=0)
        
            # Process polygon (ensure it matches baseline direction)
            if polygon is not None and len(polygon) > 1:
                poly_points = polygon.reshape(-1, 2)
                if (direction == 'ltr' and poly_points[0,0] > poly_points[-1,0]) or \
                   (direction == 'rtl' and poly_points[0,0] < poly_points[-1,0]):
                    polygon = np.flip(polygon, axis=0)
        
            processed_baselines.append(baseline)
            processed_polygons.append(polygon)
    
        return processed_baselines, processed_polygons

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize image
        image = cv2.resize(image, self.target_size)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        # Add batch and channel dimensions
        image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
        
        return image.to(self.device)
    
    def postprocess_output(self, output: torch.Tensor) -> tuple:
        """Convert model output to binary masks."""
        # Move to CPU and convert to numpy
        output = output.cpu().detach().numpy()
        
        # Get individual channels
        start_points = output[0, 0]  # First channel
        end_points = output[0, 1]    # Second channel
        baseline = output[0, 2]      # Third channel
        
        # Convert to binary masks
        start_points = (start_points > 0.5).astype(np.uint8)
        end_points = (end_points > 0.5).astype(np.uint8)
        baseline = (baseline > 0.5).astype(np.uint8)
        
        return start_points, end_points, baseline
    
    def remove_spurs(self, skeleton: np.ndarray, max_length: int = 3) -> np.ndarray:
        """Remove small branches (spurs) from the skeleton."""
        # Create a copy of the skeleton
        cleaned = skeleton.copy()
        
        # Find endpoints (pixels with exactly one neighbor)
        kernel = np.array([[1, 1, 1],
                          [1, 10, 1],
                          [1, 1, 1]], dtype=np.uint8)
        
        for _ in range(max_length):
            # Count neighbors for each pixel
            neighbors = cv2.filter2D(cleaned.astype(np.uint8), -1, kernel)
            # Endpoints have exactly one neighbor (value = 11)
            endpoints = (neighbors == 11)
            if not np.any(endpoints):
                break
            # Remove endpoints
            cleaned[endpoints] = 0
        
        return cleaned
    
    def simplify_baseline(self, baseline: np.ndarray, epsilon: float = 1.0) -> np.ndarray:
        """Simplify baseline coordinates using Douglas-Peucker algorithm."""
        # Ensure baseline is in the correct format for cv2.approxPolyDP
        baseline = baseline.reshape(-1, 1, 2).astype(np.float32)
        
        # Apply Douglas-Peucker algorithm
        simplified = cv2.approxPolyDP(baseline, epsilon, False)
        
        # Reshape back to (N, 2) format
        return simplified.reshape(-1, 2).astype(np.int32)
    
    def extract_baselines(self, mask: np.ndarray, min_length: int = 10) -> List[np.ndarray]:
        """Extract robust, non-looping, non-duplicated baselines from a binary mask using skeleton graph analysis."""
        # Skeletonize the mask
        skeleton = skeletonize(mask)
        
        # Remove spurs
        skeleton = self.remove_spurs(skeleton)
        
        # Convert to binary
        skel = skeleton.astype(np.uint8)
        h, w = skel.shape
        
        # Build graph: nodes are (y, x) pixel coordinates
        G = nx.Graph()
        for y in range(h):
            for x in range(w):
                if skel[y, x]:
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0:
                                continue
                            ny, nx_ = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx_ < w and skel[ny, nx_]:
                                G.add_edge((y, x), (ny, nx_))
        
        # Find connected components
        baselines = []
        for component in nx.connected_components(G):
            subgraph = G.subgraph(component)
            
            # Find endpoints (degree 1 nodes)
            endpoints = [n for n in subgraph.nodes if subgraph.degree[n] == 1]
            
            if len(endpoints) < 2:
                # Closed loop or small blob, skip
                continue
            
            # Find the longest simple path between any two endpoints
            max_path = []
            for i in range(len(endpoints)):
                for j in range(i+1, len(endpoints)):
                    try:
                        path = nx.shortest_path(subgraph, endpoints[i], endpoints[j])
                        if len(path) > len(max_path):
                            max_path = path
                    except nx.NetworkXNoPath:
                        continue
            
            if len(max_path) >= min_length:
                # Convert (y, x) to (x, y) for OpenCV compatibility
                baseline = np.array([(x, y) for (y, x) in max_path], dtype=np.int32)
                # Simplify the baseline
                baseline = self.simplify_baseline(baseline)
                baselines.append(baseline)
        
        return baselines
    
    def predict(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray], List[Optional[np.ndarray]]]:
        """Predict start points, end points, baselines, and polygons."""
        # Preprocess
        input_tensor = self.preprocess_image(image)
        
        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Postprocess
        start_points, end_points, baseline = self.postprocess_output(output)
        
        # Resize masks to match original image size
        h, w = image.shape[:2]
        baseline = cv2.resize(baseline, (w, h))
        
        # Extract baselines
        baselines = self.extract_baselines(baseline)
        
        # Convert image to PIL Image for polygon generation
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Generate polygons
        polygons = calculate_polygonal_environment(
            im=pil_image,
            baselines=baselines,
            scale=(w, h),
            topline=False
        )
        
        return start_points, end_points, baselines, polygons

    def generate_page_xml(self, image_path: str, baselines: List[np.ndarray], 
                         polygons: List[Optional[np.ndarray]], line_texts: Optional[List[str]] = None) -> str:
        """Generate PAGE-XML format output."""
        page_gen = PAGE_XML_Generator()

        # Read image to get dimensions
        image = cv2.imread(image_path)
        h, w = image.shape[:2]

        # Prepare polygons and baselines in the correct format
        formatted_polygons = []
        formatted_baselines = []
        for polygon, baseline in zip(polygons, baselines):
            if polygon is not None:
                formatted_polygons.append(polygon.reshape(-1, 2))
            else:
                formatted_polygons.append(np.array([[0, 0], [0, 0], [0, 0], [0, 0]]))
        
            # Add baseline points
            formatted_baselines.append(baseline.reshape(-1, 2))

        # Create region that encompasses all lines
        region = {
            'img_shape': (h, w),
            'region_name': 'paragraph',
            'region_coords': self._calculate_region_bounds(formatted_polygons),
            'lines': formatted_polygons,
            'baselines': formatted_baselines
        }

        # If no line texts provided, use empty strings
        if line_texts is None:
            line_texts = [''] * len(baselines)

        # Generate PAGE-XML
        xml_content = page_gen.create_page_xml(
            image_path=str(Path(image_path).name),
            regions=[region],
            lines=[],  # Lines are included in regions
            line_texts=line_texts
        )

        return xml_content

    def _calculate_region_bounds(self, polygons: List[Optional[np.ndarray]]) -> np.ndarray:
        """Calculate bounding polygon that encompasses all text lines."""
        # Collect all points from all polygons
        all_points = []
        for polygon in polygons:
            if polygon is not None:
                all_points.extend(polygon.reshape(-1, 2))
        
        if not all_points:
            return np.array([[0, 0]])
        
        # Convert to numpy array
        points = np.array(all_points)
        
        # Calculate convex hull
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points)
        
        # Get hull vertices in order
        region_coords = points[hull.vertices]
        
        return region_coords

    def save_results(self, image_path: str, start_points: np.ndarray, end_points: np.ndarray, 
                    baselines: List[np.ndarray], polygons: List[np.ndarray], 
                    output_path: str, line_texts: Optional[List[str]] = None,
                    baseline_direction: str = 'ltr'):
        """Save results with consistent direction."""
        # Ensure consistent direction
        baselines, polygons = self.ensure_consistent_direction(baselines, polygons, baseline_direction)
    
        # Save JSON
        self._save_json_results(image_path, start_points, end_points, baselines, polygons, output_path)
    
        # Generate and save PAGE-XML
        xml_content = self.generate_page_xml(image_path, baselines, polygons, line_texts)
        xml_path = str(Path(output_path).with_suffix('.xml'))
    
        with open(xml_path, 'w', encoding='utf-8') as f:
            f.write(xml_content)
    
        logger.info(f"Saved PAGE-XML to {xml_path}")

    def _save_json_results(self, image_path: str, start_points: np.ndarray, end_points: np.ndarray, 
                          baselines: List[np.ndarray], polygons: List[Optional[np.ndarray]], 
                          output_path: str):
        """Save results to JSON file."""
        # Get image information
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        
        # Prepare results dictionary
        results = {
            'image_info': {
                'path': str(Path(image_path).absolute()),
                'width': w,
                'height': h,
                'filename': Path(image_path).name
            },
            'lines': []
        }
        
        # Add each line's information
        for i, (baseline, polygon) in enumerate(zip(baselines, polygons)):
            line_info = {
                'id': i,
                'baseline': baseline.tolist(),
                'polygon': polygon.tolist() if polygon is not None else None
            }
            results['lines'].append(line_info)
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved JSON results to {output_path}")

def _ensure_correct_side(baseline: np.ndarray, seam: np.ndarray) -> float:
    """Ensure seam is on the correct side of the baseline.
    Returns vertical distance: positive means seam is below baseline."""
    baseline_center = np.mean(baseline, axis=0)
    seam_center = np.mean(seam, axis=0)
    direction = seam_center - baseline_center
    return direction[1]  # vertical distance: positive means seam is below baseline

def main():
    parser = argparse.ArgumentParser(description='Predict baselines and polygons')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the trained model checkpoint')
    parser.add_argument('--image_path', type=str, required=True,
                      help='Path to the input image')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save outputs')
    parser.add_argument('--target_width', type=int, default=768,
                      help='Target width for images')
    parser.add_argument('--target_height', type=int, default=1024,
                      help='Target height for images')
    parser.add_argument('--topline', action='store_true',
                      help='Use baseline as top line for polygon generation')
    parser.add_argument('--baseline-direction', type=str, default='ltr',
                       choices=['ltr', 'rtl'],
                      help='Direction for baselines and polygons (ltr: left-to-right, rtl: right-to-left)')
    parser.add_argument('--visualization', action='store_true',
                      help='Enable visualization output')

    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize detector
    detector = BaselineDetector(
        model_path=args.model_path,
        target_size=(args.target_width, args.target_height)
    )
    
    # Load and process image
    image = cv2.imread(args.image_path)
    if image is None:
        raise ValueError(f"Could not load image from {args.image_path}")
    
    # Get predictions
    start_points, end_points, baselines, polygons = detector.predict(image)
    
    # Save results (now saves both JSON and PAGE-XML)
    output_json_path = output_dir / f"{Path(args.image_path).stem}.json"
    detector.save_results(
        args.image_path,
        start_points,
        end_points,
        baselines,
        polygons,
        str(output_json_path),
        baseline_direction=args.baseline_direction
    )

    if args.visualization:
        # Create visualization
        vis_image = image.copy()
    
        # Draw polygons in semi-transparent green first
        for polygon in polygons:
            if polygon is not None:
                polygon = polygon.reshape(-1, 1, 2).astype(np.int32)
                overlay = vis_image.copy()
                cv2.fillPoly(overlay, [polygon], (0, 255, 0))
                cv2.addWeighted(overlay, 0.3, vis_image, 0.7, 0, vis_image)
    
        # Draw baselines in red on top
        for baseline in baselines:
            baseline = baseline.reshape(-1, 1, 2).astype(np.int32)
            cv2.polylines(vis_image, [baseline], False, (0, 0, 255), 1)
    
        # Save visualization
        vis_path = output_dir / f"{Path(args.image_path).stem}_visualization.png"
        cv2.imwrite(str(vis_path), vis_image)
        logger.info(f"Saved visualization to {vis_path}")

if __name__ == "__main__":
    main()
