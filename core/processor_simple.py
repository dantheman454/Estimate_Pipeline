"""
Simplified processor for CLI without OpenCV dependencies
Uses basic PIL-only image processing
"""

from pdf2image import convert_from_path
from PIL import Image, ImageEnhance
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def process_blueprint_multipage(pdf_path: str) -> List[Image.Image]:
    """Convert multi-page PDF blueprint to enhanced images for AI analysis.
    
    Simplified version without OpenCV dependencies - uses PIL only.
    
    Args:
        pdf_path (str): Absolute path to the PDF blueprint file to process.
        
    Returns:
        List[Image.Image]: List of PIL Image objects, one per PDF page, enhanced
        for electrical component detection.
    """
    logger.info(f"Converting PDF to images: {pdf_path}")
    
    # Convert PDF to images at high DPI for quality
    images = convert_from_path(pdf_path, dpi=400, fmt='RGB')
    logger.info(f"Converted {len(images)} pages from PDF")
    
    # Enhance each image
    enhanced_images = []
    for i, img in enumerate(images):
        logger.info(f"Enhancing page {i+1}/{len(images)}...")
        enhanced_img = enhance_scanned_blueprint(img)
        enhanced_images.append(enhanced_img)
    
    return enhanced_images


def enhance_scanned_blueprint(image: Image.Image) -> Image.Image:
    """Enhance scanned blueprint image for better AI analysis.
    
    Simple PIL-based enhancement without OpenCV dependencies.
    
    Args:
        image (Image.Image): Original PIL Image object to enhance.
        
    Returns:
        Image.Image: Enhanced PIL Image with improved contrast and sharpness.
    """
    # Convert to RGB if not already
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Increase contrast for better symbol visibility
    contrast_enhancer = ImageEnhance.Contrast(image)
    enhanced = contrast_enhancer.enhance(1.3)  # 30% more contrast
    
    # Increase sharpness for clearer edges and text
    sharpness_enhancer = ImageEnhance.Sharpness(enhanced)
    sharpened = sharpness_enhancer.enhance(1.2)  # 20% more sharpness
    
    return sharpened


# DEPRECATED: This function will be removed in the image-based pipeline upgrade
# Use process_blueprint_multipage for simple image conversion
def process_blueprint_with_floor_plans(pdf_path: str) -> Dict:
    """
    DEPRECATED: This PDF-specific floor plan detection will be removed.
    Use the new image-based pipeline for processing.
    """
    logger.warning("process_blueprint_with_floor_plans is deprecated. Use image-based processing instead.")
    
    # Convert PDF to images
    images = process_blueprint_multipage(pdf_path)
    
    # Create simplified structure - treat each page as one floor plan
    result = {
        'pages': [],
        'total_pages': len(images),
        'total_floor_plans': len(images)
    }
    
    for i, image in enumerate(images):
        # Create a simple floor plan entry for each page
        floor_plan = {
            'title': f'FLOOR PLAN {i+1}',
            'image': image,
            'confidence': 95.0
        }
        
        page_data = {
            'page_number': i + 1,
            'floor_plans': [floor_plan]
        }
        
        result['pages'].append(page_data)
    
    return result
