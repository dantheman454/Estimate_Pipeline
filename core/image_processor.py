"""
Image Processing Module for Blueprint Pipeline
Handles legend extraction and floor plan separation from combined images
"""

import numpy as np
import sys
from pathlib import Path

# Workaround for OpenCV import path issue
try:
    import cv2
except ImportError:
    # Add the virtual environment site-packages path
    venv_path = Path(__file__).parent.parent / "venv_cli" / "lib" / "python3.13" / "site-packages"
    if venv_path.exists():
        sys.path.insert(0, str(venv_path))
        import cv2
    else:
        raise ImportError("OpenCV not found. Please install opencv-python in the virtual environment.")

try:
    import pytesseract
    HAS_PYTESSERACT = True
except ImportError:
    HAS_PYTESSERACT = False
    
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
from typing import Tuple, Dict, List, Optional
import logging

# Remove rich dependencies for now
# from rich.console import Console
# from rich.progress import Progress, SpinnerColumn, TextColumn

logger = logging.getLogger(__name__)
# console = Console()


class ImageProcessor:
    """
    Advanced image processor for separating electrical legends from floor plans.
    
    Features:
    - Text-based legend detection using OCR
    - Layout-based legend detection as fallback
    - Region cropping and extraction
    - Image enhancement for better processing
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.legend_keywords = [
            'ELECTRICAL', 'LEGEND', 'SYMBOLS', 'KEY', 
            'SCHEDULE', 'NOTES', 'ABBREVIATIONS'
        ]
    
    def extract_key_and_plan(self, image: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """
        Extract electrical key/legend and floor plan from combined image.
        
        Args:
            image (Image.Image): Combined image containing both legend and floor plan
            
        Returns:
            Tuple[Image.Image, Image.Image]: (key_image, plan_image)
        """
        if self.verbose:
            print("🔍 Analyzing image for legend separation...")
        
        # Convert PIL to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Step 1: Try text-based detection
        if self.verbose:
            print("   Detecting legend by text recognition...")
        key_region = self._find_key_region_by_text(image)
        
        # Step 2: If text detection fails, try layout-based detection
        if not key_region:
            if self.verbose:
                print("   Detecting legend by layout analysis...")
            key_region = self._find_key_region_by_layout(cv_image)
        
        # Step 3: Extract regions
        if self.verbose:
            print("   Extracting legend and floor plan regions...")
        
        if key_region:
            key_image = image.crop(key_region)
            plan_image = self._remove_key_region(image, key_region)
            
            if self.verbose:
                print(f"✅ Legend found at region: {key_region}")
        else:
            # Fallback: assume top or side region is legend
            key_image, plan_image = self._fallback_region_split(image)
            if self.verbose:
                print("⚠️  Using fallback region splitting")
        
        return key_image, plan_image
    
    def _find_key_region_by_text(self, image: Image.Image) -> Optional[Tuple[int, int, int, int]]:
        """
        Find key region by looking for electrical legend text.
        
        Args:
            image (Image.Image): Source image to analyze
            
        Returns:
            Optional[Tuple[int, int, int, int]]: (left, top, right, bottom) or None
        """
        try:
            # Use pytesseract to find text regions
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            legend_regions = []
            
            # Look for legend-related text
            for i, text in enumerate(data['text']):
                if any(keyword in text.upper() for keyword in self.legend_keywords):
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    confidence = data['conf'][i]
                    
                    if confidence > 30:  # Filter out low-confidence detections
                        legend_regions.append((x, y, x + w, y + h, confidence))
            
            if legend_regions:
                # Find the most confident detection
                best_region = max(legend_regions, key=lambda r: r[4])
                region = best_region[:4]  # Remove confidence score
                
                # Expand region to capture full legend
                return self._expand_key_region(image, region)
            
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Text-based detection failed: {e}")
        
        return None
    
    def _find_key_region_by_layout(self, cv_image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Find key region by analyzing image layout and structure.
        
        Args:
            cv_image (np.ndarray): OpenCV image array
            
        Returns:
            Optional[Tuple[int, int, int, int]]: (left, top, right, bottom) or None
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3, L2gradient=True)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Look for rectangular regions that might be legends
            legend_candidates = []
            
            image_height, image_width = gray.shape
            
            for contour in contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by size - legends are usually smaller regions
                if (w > image_width * 0.1 and w < image_width * 0.4 and 
                    h > image_height * 0.1 and h < image_height * 0.6):
                    
                    # Calculate aspect ratio
                    aspect_ratio = w / h
                    
                    # Legends often have specific aspect ratios
                    if 0.5 < aspect_ratio < 3.0:
                        legend_candidates.append((x, y, x + w, y + h, w * h))
            
            if legend_candidates:
                # Sort by area and take the most reasonable sized region
                legend_candidates.sort(key=lambda r: r[4], reverse=True)
                
                # Take the largest reasonable region
                best_region = legend_candidates[0][:4]  # Remove area score
                return best_region
            
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Layout-based detection failed: {e}")
        
        return None
    
    def _expand_key_region(self, image: Image.Image, region: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """
        Expand the detected key region to capture the full legend.
        
        Args:
            image (Image.Image): Source image
            region (Tuple[int, int, int, int]): Initial region (x1, y1, x2, y2)
            
        Returns:
            Tuple[int, int, int, int]: Expanded region coordinates
        """
        x1, y1, x2, y2 = region
        width, height = image.size
        
        # Expand by 20% in each direction, but stay within image bounds
        expand_x = int((x2 - x1) * 0.2)
        expand_y = int((y2 - y1) * 0.2)
        
        x1 = max(0, x1 - expand_x)
        y1 = max(0, y1 - expand_y)
        x2 = min(width, x2 + expand_x)
        y2 = min(height, y2 + expand_y)
        
        return (x1, y1, x2, y2)
    
    def _remove_key_region(self, image: Image.Image, key_region: Tuple[int, int, int, int]) -> Image.Image:
        """
        Remove the key region from the image to get clean floor plan.
        
        Args:
            image (Image.Image): Source image
            key_region (Tuple[int, int, int, int]): Region to remove (x1, y1, x2, y2)
            
        Returns:
            Image.Image: Floor plan with legend area blanked out
        """
        # Create a copy of the image
        plan_image = image.copy()
        
        # Create a drawing context
        draw = ImageDraw.Draw(plan_image)
        
        # Fill the legend region with white
        draw.rectangle(key_region, fill='white')
        
        return plan_image
    
    def _fallback_region_split(self, image: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """
        Fallback method to split image when automatic detection fails.
        Assumes legend is in top-left or right portion.
        
        Args:
            image (Image.Image): Source image
            
        Returns:
            Tuple[Image.Image, Image.Image]: (key_image, plan_image)
        """
        width, height = image.size
        
        # Try different common legend positions
        
        # Option 1: Top 30% of image
        top_region = (0, 0, width, int(height * 0.3))
        
        # Option 2: Left 40% of image
        left_region = (0, 0, int(width * 0.4), height)
        
        # Option 3: Right 40% of image  
        right_region = (int(width * 0.6), 0, width, height)
        
        # For now, default to top region
        # In a more advanced implementation, we could analyze each region
        # and choose the one most likely to contain a legend
        
        key_region = top_region
        key_image = image.crop(key_region)
        plan_image = self._remove_key_region(image, key_region)
        
        return key_image, plan_image
    
    def enhance_for_symbol_detection(self, image: Image.Image) -> Image.Image:
        """
        Enhance image specifically for electrical symbol detection.
        
        Args:
            image (Image.Image): Image to enhance
            
        Returns:
            Image.Image: Enhanced image
        """
        # Convert to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Apply sharpening
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Convert back to PIL
        enhanced_pil = Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
        
        return enhanced_pil
    
    def save_debug_images(self, key_image: Image.Image, plan_image: Image.Image, 
                         output_dir: str = "debug_output"):
        """
        Save extracted regions for debugging purposes.
        
        Args:
            key_image (Image.Image): Extracted legend image
            plan_image (Image.Image): Extracted floor plan image
            output_dir (str): Directory to save debug images
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = Path().cwd().name  # Use current directory name as identifier
        
        key_path = output_path / f"extracted_key_{timestamp}.png"
        plan_path = output_path / f"extracted_plan_{timestamp}.png"
        
        key_image.save(key_path)
        plan_image.save(plan_path)
        
        if self.verbose:
            print(f"💾 Debug images saved:")
            print(f"   Legend: {key_path}")
            print(f"   Floor Plan: {plan_path}")


def main():
    """Demo and testing of the image processing module."""
    print("🖼️  Image Processing Module - Demo")
    
    # Test with the provided images
    test_images = ["key.png", "Full_plan.png"]
    
    processor = ImageProcessor(verbose=True)
    
    for img_file in test_images:
        img_path = Path(img_file)
        if img_path.exists():
            print(f"\n📄 Processing: {img_file}")
            
            try:
                # Load image
                image = Image.open(img_path)
                print(f"   Size: {image.size[0]}x{image.size[1]} pixels")
                
                # Extract key and plan
                key_image, plan_image = processor.extract_key_and_plan(image)
                
                # Save debug images
                processor.save_debug_images(key_image, plan_image)
                
                print("✅ Successfully processed!")
                
            except Exception as e:
                print(f"❌ Error processing {img_file}: {e}")
        else:
            print(f"⚠️  Test image not found: {img_file}")


if __name__ == "__main__":
    main()
