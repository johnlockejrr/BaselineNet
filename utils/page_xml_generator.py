import uuid
import xml.etree.ElementTree as ET
from xml.dom import minidom
from datetime import datetime, timezone
from typing import List, Dict, Optional
import numpy as np
from scipy.spatial import ConvexHull

class PAGE_XML_Generator:
    def __init__(self):
        self.namespace = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
        self.schema_location = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15/pagecontent.xsd"
    
    def generate_id(self, prefix: str) -> str:
        """Generate unique ID with prefix."""
        return f"{prefix}_{uuid.uuid4().hex[:8]}"

    def detect_direction(self, points: np.ndarray) -> str:
        """Detect if points are ordered Left-to-Right or Right-to-Left."""
        if len(points) < 2:
            return 'ltr'
    
        x_coords = points[:, 0]
        if x_coords[0] < x_coords[-1]:
            return 'ltr'
        return 'rtl'

    def ensure_direction(self, points: np.ndarray, target_direction: str = 'ltr') -> np.ndarray:
        """Ensure points follow specified direction."""
        if len(points) < 2:
            return points
    
        current_direction = self.detect_direction(points)
        if current_direction != target_direction:
            return np.flip(points, axis=0)
        return points

    def create_page_xml(
        self,
        image_path: str,
        regions: List[Dict],
        lines: List[Dict],
        line_texts: List[str],
        output_path: Optional[str] = None
    ) -> str:
        """
        Create PAGE-XML structure from detection results.
        """
        # Create root element with namespaces
        root = ET.Element("PcGts")
        root.set("xmlns", self.namespace)
        root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
        root.set("xsi:schemaLocation", f"{self.namespace} {self.schema_location}")

        # Add metadata
        metadata = ET.SubElement(root, "Metadata")
        creator = ET.SubElement(metadata, "Creator")
        creator.text = "U-Net-BaseLineSeg"
        
        current_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "+00:00"
        created = ET.SubElement(metadata, "Created")
        created.text = current_time
        last_change = ET.SubElement(metadata, "LastChange")
        last_change.text = current_time

        # Create Page element
        page = ET.SubElement(root, "Page", 
                           imageFilename=image_path,
                           imageWidth=str(regions[0]['img_shape'][1]),
                           imageHeight=str(regions[0]['img_shape'][0]))

        # Add regions with their lines
        for region in regions:
            region_id = self.generate_id("uNet_textblock")
            region_elem = ET.SubElement(page, "TextRegion", 
                                      id=region_id,
                                      custom=f"structure {{type:{region['region_name']};}}")
            
            # Add region coordinates
            coords = ET.SubElement(region_elem, "Coords")
            coords.set("points", self.polygon_to_pagexml(region['region_coords']))

            # Add lines belonging to this region
            for line_idx, line in enumerate(region['lines']):
                line_id = self.generate_id("uNet_line")
                line_elem = ET.SubElement(region_elem, "TextLine", 
                                        id=line_id,
                                        custom="structure {type:text_line;}")
                
                # Line coordinates
                line_coords = ET.SubElement(line_elem, "Coords")
                line_coords.set("points", self.polygon_to_pagexml(line))
                
                # Baseline (approximate from polygon)
                baseline_elem = ET.SubElement(line_elem, "Baseline")
                baseline_elem.set("points", self.polygon_to_pagexml(region['baselines'][line_idx]))
                
                # Text content
                if line_idx < len(line_texts):
                    text_equiv = ET.SubElement(line_elem, "TextEquiv")
                    unicode_elem = ET.SubElement(text_equiv, "Unicode")
                    unicode_elem.text = line_texts[line_idx]

        # Convert to pretty-printed XML
        xml_str = ET.tostring(root, encoding="utf-8")
        xml_pretty = minidom.parseString(xml_str).toprettyxml(indent="  ")
        
        # Save to file if output path provided
        if output_path:
            with open(output_path, "w") as f:
                f.write(xml_pretty)
        
        return xml_pretty

    def polygon_to_pagexml(self, polygon: np.ndarray) -> str:
        """Convert polygon array to PAGE-XML coordinate string."""
        return " ".join([f"{int(x)},{int(y)}" for x, y in polygon])
 