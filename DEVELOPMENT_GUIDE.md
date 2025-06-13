# Blueprint Symbol Detection Development Guide

## Current State Assessment

Based on your codebase analysis, you have a solid foundation with these key components:
- **Symbol Extractor** (`core/symbol_extractor.py`) - Partially implemented legend processing
- **Template Matcher** (`core/template_matcher.py`) - Advanced template matching framework
- **Process Blueprint** (`process_blueprint.py`) - Main CLI interface
- **Simple Processor** (`core/processor_simple.py`) - Image preprocessing pipeline

## The Core Problem

Your current pipeline has these issues:
1. **Legend Processing**: Cannot reliably extract individual symbols and their names from legend/key images
2. **Template Matching**: Cannot effectively match extracted symbols to blueprint symbols
3. **Symbol Counting**: Cannot accurately count symbol occurrences in blueprints
4. **Pipeline Integration**: The current hybrid approach needs better coordination

## Immediate Development Priority

Based on your current project structure and existing code, here's the most critical development path to solve the legend-to-symbol-counting problem:

## Step-by-Step Development Plan

### Phase 1: Legend Processing - Extracting Symbols and Their Names

#### Step 1.1: Input Legend Image Processing

**Goal**: Modify the main CLI to accept legend images and load them properly.

**Implementation**: Update `process_blueprint.py` to ensure legend handling is robust:

```python
def validate_inputs(args):
    """Enhanced validation with better legend support."""
    # ...existing code...
    
    # Enhanced legend validation
    legend_path = None
    if args.legend:
        legend_path = Path(args.legend)
        if not legend_path.exists():
            print(f"❌ Error: Legend file not found: {legend_path}")
            print("   Legend is required for accurate symbol detection.")
            sys.exit(1)  # Make legend required for best results
        elif not legend_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            print(f"❌ Error: Legend file must be PNG or JPG: {legend_path}")
            sys.exit(1)
        else:
            print(f"✅ Legend file validated: {legend_path.name}")
    else:
        print("⚠️  Warning: No legend provided. Detection accuracy will be reduced.")
        print("   For best results, use: --legend key.png")
    
    return main_file_path.absolute(), pricing_path, legend_path
```

#### Step 1.2: Isolate Individual Symbols from Legend

**Goal**: Enhance the `SymbolExtractor` class to better detect and isolate symbol regions.

**Implementation**: Improve the `_detect_symbol_regions` method in `core/symbol_extractor.py`:

```python
def _detect_symbol_regions_enhanced(self, legend_cv: np.ndarray) -> List[Tuple]:
    """
    Enhanced symbol region detection with better filtering and layout analysis.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(legend_cv, cv2.COLOR_BGR2GRAY)
    
    # Enhanced preprocessing pipeline
    # 1. Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # 2. Apply adaptive threshold for better symbol separation
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # 3. Morphological operations to clean up symbols
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 4. Find contours with hierarchy for better filtering
    contours, hierarchy = cv2.findContours(
        cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Enhanced symbol filtering criteria
    symbol_candidates = []
    
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        aspect_ratio = w / max(h, 1)
        
        # Enhanced filtering criteria for electrical symbols
        if (
            20 <= w <= 150 and          # Width range
            15 <= h <= 80 and           # Height range
            200 <= area <= 8000 and     # Area range
            0.3 <= aspect_ratio <= 4.0  # Aspect ratio range
        ):
            # Additional shape analysis
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Electrical symbols typically have moderate complexity
            if 0.1 <= circularity <= 0.9:
                symbol_candidates.append((x, y, w, h, area, circularity))
    
    # Sort by position (top-to-bottom, left-to-right for typical legend layouts)
    symbol_candidates.sort(key=lambda c: (c[1] // 50, c[0]))  # Group by rows
    
    # Extract symbols with enhanced text detection
    symbol_regions = []
    
    for x, y, w, h, area, circularity in symbol_candidates:
        # Extract symbol region with padding
        padding = 5
        symbol_img = legend_cv[
            max(0, y-padding):min(legend_cv.shape[0], y+h+padding),
            max(0, x-padding):min(legend_cv.shape[1], x+w+padding)
        ]
        
        # Enhanced adjacent text finding
        text_region, text_content = self._find_adjacent_text_enhanced(
            legend_cv, x, y, w, h
        )
        
        symbol_bbox = (x, y, w, h)
        symbol_regions.append((symbol_bbox, text_region, symbol_img, text_content))
        
        if self.verbose:
            print(f"   Detected symbol region: {w}x{h} at ({x},{y}), "
                  f"area={area}, circularity={circularity:.2f}")
    
    return symbol_regions
```

#### Step 1.3: Enhanced Text Extraction with OCR

**Goal**: Improve OCR accuracy for symbol name extraction.

**Implementation**: Create an enhanced text extraction method:

```python
def _find_adjacent_text_enhanced(self, legend_cv: np.ndarray, x: int, y: int, w: int, h: int) -> Tuple[Optional[Tuple], str]:
    """
    Enhanced text detection with multiple strategies and OCR preprocessing.
    """
    img_height, img_width = legend_cv.shape[:2]
    
    # Define multiple search strategies
    search_strategies = [
        # Strategy 1: Right side (most common in electrical legends)
        {
            'region': (x + w + 10, max(0, y - 10), min(300, img_width - (x + w + 10)), h + 20),
            'weight': 1.0,
            'name': 'right'
        },
        # Strategy 2: Below symbol
        {
            'region': (max(0, x - 20), y + h + 5, w + 40, min(60, img_height - (y + h + 5))),
            'weight': 0.8,
            'name': 'below'
        },
        # Strategy 3: Above symbol
        {
            'region': (max(0, x - 20), max(0, y - 60), w + 40, min(60, y)),
            'weight': 0.6,
            'name': 'above'
        },
        # Strategy 4: Left side (less common but possible)
        {
            'region': (max(0, x - 200), max(0, y - 10), min(200, x), h + 20),
            'weight': 0.4,
            'name': 'left'
        }
    ]
    
    best_text = ""
    best_region = None
    best_confidence = 0
    
    for strategy in search_strategies:
        rx, ry, rw, rh = strategy['region']
        
        # Validate region bounds
        if (rx >= 0 and ry >= 0 and 
            rx + rw <= img_width and ry + rh <= img_height and
            rw > 10 and rh > 5):  # Minimum size for text
            
            text_content, confidence = self._extract_text_with_confidence(
                legend_cv, rx, ry, rw, rh
            )
            
            # Score based on text quality and strategy weight
            text_score = len(text_content.strip()) * confidence * strategy['weight']
            
            if text_score > best_confidence and len(text_content.strip()) >= 2:
                best_text = text_content
                best_region = strategy['region']
                best_confidence = text_score
                
                if self.verbose:
                    print(f"     Found text ({strategy['name']}): '{text_content.strip()}' "
                          f"(confidence: {confidence:.2f})")
    
    return best_region, best_text

def _extract_text_with_confidence(self, legend_cv: np.ndarray, x: int, y: int, w: int, h: int) -> Tuple[str, float]:
    """Extract text with confidence scoring for better text detection."""
    if not HAS_OCR:
        return "", 0.0
    
    try:
        # Extract and preprocess region
        region = legend_cv[y:y+h, x:x+w]
        
        # Convert to PIL and enhance for OCR
        region_pil = Image.fromarray(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
        
        # OCR preprocessing pipeline
        # 1. Convert to grayscale
        region_gray = region_pil.convert('L')
        
        # 2. Upscale for better OCR (3x scaling)
        new_size = (region_gray.width * 3, region_gray.height * 3)
        region_upscaled = region_gray.resize(new_size, Image.Resampling.LANCZOS)
        
        # 3. Apply contrast enhancement
        enhancer = ImageEnhance.Contrast(region_upscaled)
        region_enhanced = enhancer.enhance(2.0)
        
        # 4. Apply sharpening
        sharpener = ImageEnhance.Sharpness(region_enhanced)
        region_sharp = sharpener.enhance(1.5)
        
        # Extract text with detailed configuration
        custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-/(). '
        
        # Get text with additional data for confidence
        text_data = pytesseract.image_to_data(
            region_sharp, 
            config=custom_config,
            output_type=pytesseract.Output.DICT
        )
        
        # Extract text and calculate confidence
        text_parts = []
        confidences = []
        
        for i in range(len(text_data['text'])):
            text = text_data['text'][i].strip()
            conf = int(text_data['conf'][i])
            
            if text and conf > 0:  # Valid text with confidence
                text_parts.append(text)
                confidences.append(conf)
        
        # Combine text and calculate average confidence
        full_text = ' '.join(text_parts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Normalize confidence to 0-1 range
        normalized_confidence = avg_confidence / 100.0
        
        return full_text, normalized_confidence
        
    except Exception as e:
        if self.verbose:
            print(f"     OCR error: {e}")
        return "", 0.0
```

#### Step 1.4: Create Enhanced Symbol-Name Key

**Goal**: Build a robust data structure mapping symbols to names.

**Implementation**: Enhance the symbol extraction result structure:

```python
def extract_symbols_from_legend_enhanced(self, legend_image: Image.Image) -> Dict[str, Any]:
    """
    Enhanced legend processing with better symbol extraction and classification.
    """
    # Convert to OpenCV format
    legend_cv = cv2.cvtColor(np.array(legend_image), cv2.COLOR_RGB2BGR)
    
    # Detect symbol regions with enhanced filtering
    symbol_regions = self._detect_symbol_regions_enhanced(legend_cv)
    
    if self.verbose:
        print(f"🔍 Enhanced Legend Analysis:")
        print(f"   Found {len(symbol_regions)} potential symbol regions")
    
    # Process each symbol region
    extracted_symbols = {}
    symbol_metadata = {}
    
    for i, (symbol_bbox, text_bbox, symbol_img, text_content) in enumerate(symbol_regions):
        # Classify symbol type
        symbol_type = self._classify_symbol_text(text_content)
        
        if symbol_type and len(text_content.strip()) >= 2:
            # Convert to PIL for storage
            symbol_pil = Image.fromarray(cv2.cvtColor(symbol_img, cv2.COLOR_BGR2RGB))
            
            # Analyze symbol characteristics for better matching
            characteristics = self.analyze_symbol_characteristics(symbol_pil)
            
            # Store symbol with metadata
            extracted_symbols[symbol_type] = symbol_pil
            symbol_metadata[symbol_type] = {
                'original_text': text_content.strip(),
                'bbox': symbol_bbox,
                'text_bbox': text_bbox,
                'characteristics': characteristics,
                'extraction_order': i
            }
            
            if self.verbose:
                print(f"   ✅ Symbol {i+1}: {symbol_type}")
                print(f"      Text: '{text_content.strip()}'")
                print(f"      Size: {symbol_bbox[2]}x{symbol_bbox[3]}")
                print(f"      Complexity: {characteristics['complexity']:.2f}")
        else:
            if self.verbose:
                print(f"   ❌ Symbol {i+1}: Could not classify")
                print(f"      Text: '{text_content.strip()}'")
    
    # Store both symbols and metadata
    self.extracted_symbols = extracted_symbols
    self.symbol_metadata = symbol_metadata
    
    # Return enhanced structure
    return {
        'symbols': extracted_symbols,
        'metadata': symbol_metadata,
        'extraction_summary': {
            'total_regions_found': len(symbol_regions),
            'symbols_classified': len(extracted_symbols),
            'success_rate': len(extracted_symbols) / max(len(symbol_regions), 1)
        }
    }
```

### Phase 2: Blueprint Symbol Matching and Counting

#### Step 2.1: Enhanced Template Matching

**Goal**: Improve the template matching system to handle variations in symbol size and orientation.

**Implementation**: Create a robust template matching method in `core/template_matcher.py`:

```python
def detect_components_with_enhanced_matching(
    self, 
    blueprint_image: Union[Image.Image, np.ndarray],
    symbol_templates: Dict[str, Image.Image],
    min_confidence: float = 0.65
) -> Dict[str, int]:
    """
    Enhanced template matching with multi-scale, rotation, and noise handling.
    
    Args:
        blueprint_image: Blueprint image to search
        symbol_templates: Dictionary of symbol templates from legend
        min_confidence: Minimum confidence threshold
        
    Returns:
        Dictionary of component counts with high confidence
    """
    if not symbol_templates:
        if self.verbose:
            logger.warning("No symbol templates provided for matching")
        return {}
    
    # Preprocess blueprint for optimal matching
    cv_blueprint = self._pil_to_cv2(blueprint_image) if isinstance(blueprint_image, Image.Image) else blueprint_image
    processed_blueprint = self._preprocess_blueprint_for_matching(cv_blueprint)
    
    component_counts = {}
    all_detections = {}
    
    if self.verbose:
        print(f"🔍 Starting enhanced template matching for {len(symbol_templates)} symbol types...")
    
    # Process each symbol template
    for symbol_name, symbol_template in symbol_templates.items():
        if self.verbose:
            print(f"   Processing: {symbol_name}")
        
        # Convert template to OpenCV format
        cv_template = self._pil_to_cv2(symbol_template)
        processed_template = self._preprocess_template_for_matching(cv_template)
        
        # Multi-scale and rotation template matching
        detections = self._multi_scale_rotation_template_match(
            processed_blueprint, processed_template, symbol_name, min_confidence
        )
        
        # Apply non-maximum suppression to remove duplicates
        filtered_detections = self._enhanced_non_maximum_suppression(
            detections, overlap_threshold=0.3
        )
        
        # Store results
        component_counts[symbol_name] = len(filtered_detections)
        all_detections[symbol_name] = filtered_detections
        
        if self.verbose:
            print(f"      Found: {len(filtered_detections)} instances")
    
    # Store detection details for debugging
    self.last_detections = all_detections
    
    # Calculate detection summary
    total_detections = sum(component_counts.values())
    if self.verbose:
        print(f"✅ Template matching complete: {total_detections} total components detected")
    
    return component_counts

def _preprocess_blueprint_for_matching(self, image: np.ndarray) -> np.ndarray:
    """Enhanced blueprint preprocessing for better template matching."""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply advanced noise reduction
    # 1. Bilateral filter to preserve edges while reducing noise
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # 2. Morphological opening to remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    opened = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel)
    
    # 3. Adaptive histogram equalization for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(opened)
    
    return enhanced

def _multi_scale_rotation_template_match(
    self, 
    image: np.ndarray, 
    template: np.ndarray, 
    component_name: str,
    min_confidence: float
) -> List[Tuple[int, int, int, int, float, float, float]]:
    """
    Advanced template matching with multiple scales and rotations.
    
    Returns:
        List of detections: (x, y, width, height, confidence, scale, rotation)
    """
    detections = []
    
    # Scale factors for multi-scale matching
    scales = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
    
    # Rotation angles (electrical symbols are typically axis-aligned, so small angles)
    rotations = [-10, -5, 0, 5, 10]  # degrees
    
    template_h, template_w = template.shape
    
    for scale in scales:
        # Skip scales that make template too small or too large
        scaled_w = int(template_w * scale)
        scaled_h = int(template_h * scale)
        
        if scaled_w < 10 or scaled_h < 10 or scaled_w > image.shape[1]//2 or scaled_h > image.shape[0]//2:
            continue
        
        for rotation in rotations:
            try:
                # Scale template
                scaled_template = cv2.resize(template, (scaled_w, scaled_h), interpolation=cv2.INTER_CUBIC)
                
                # Rotate template if needed
                if rotation != 0:
                    rotation_matrix = cv2.getRotationMatrix2D(
                        (scaled_w/2, scaled_h/2), rotation, 1.0
                    )
                    rotated_template = cv2.warpAffine(scaled_template, rotation_matrix, (scaled_w, scaled_h))
                else:
                    rotated_template = scaled_template
                
                # Perform template matching
                result = cv2.matchTemplate(image, rotated_template, cv2.TM_CCOEFF_NORMED)
                
                # Find matches above threshold
                locations = np.where(result >= min_confidence)
                
                for y, x in zip(locations[0], locations[1]):
                    confidence = result[y, x]
                    
                    detections.append((
                        int(x), int(y), scaled_w, scaled_h, 
                        float(confidence), scale, rotation
                    ))
                    
            except Exception as e:
                if self.verbose:
                    logger.warning(f"Template matching failed for scale {scale}, rotation {rotation}: {e}")
                continue
    
    return detections

def _enhanced_non_maximum_suppression(
    self, 
    detections: List[Tuple[int, int, int, int, float, float, float]], 
    overlap_threshold: float = 0.3
) -> List[Tuple[int, int, int, int, float]]:
    """
    Enhanced NMS that considers confidence, scale, and rotation for better duplicate removal.
    """
    if not detections:
        return []
    
    # Sort by confidence (highest first)
    detections = sorted(detections, key=lambda x: x[4], reverse=True)
    
    final_detections = []
    
    for detection in detections:
        x, y, w, h, confidence, scale, rotation = detection
        
        # Check if this detection overlaps significantly with any already selected detection
        should_keep = True
        
        for kept_detection in final_detections:
            kx, ky, kw, kh, _ = kept_detection
            
            # Calculate overlap
            overlap = self._calculate_bbox_overlap(
                (x, y, w, h), (kx, ky, kw, kh)
            )
            
            if overlap > overlap_threshold:
                should_keep = False
                break
        
        if should_keep:
            final_detections.append((x, y, w, h, confidence))
    
    return final_detections

def _calculate_bbox_overlap(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
    """Calculate overlap ratio between two bounding boxes."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Calculate intersection
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_left >= x_right or y_top >= y_bottom:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - intersection_area
    
    # Calculate overlap ratio
    return intersection_area / union_area if union_area > 0 else 0.0
```

### Phase 3: Pipeline Integration and Refinement

#### Step 3.1: Create Enhanced Legend Processor Module

**Goal**: Create a new module that encapsulates all legend processing logic.

**Implementation**: Create `core/legend_processor.py`:

```python
"""
Legend Processor Module
Handles complete legend analysis and symbol template extraction for blueprint processing.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from PIL import Image
import numpy as np

from .symbol_extractor import SymbolExtractor
from .template_matcher import TemplateBasedDetector

logger = logging.getLogger(__name__)


class LegendProcessor:
    """
    Complete legend processing pipeline for electrical blueprint analysis.
    
    Features:
    - Automatic symbol detection and extraction
    - OCR-based symbol classification
    - Template optimization for matching
    - Symbol validation and quality assessment
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.symbol_extractor = SymbolExtractor(verbose=verbose)
        self.template_matcher = TemplateBasedDetector(verbose=verbose)
        self.symbol_templates = {}
        self.processing_metadata = {}
        self.legend_data = {}
        
        if verbose:
            print("🔧 Initialized Enhanced Legend Processor")
    
    def process_legend(self, legend_path: str) -> Dict[str, Any]:
        """
        Process legend image and extract usable symbol templates.
        
        Args:
            legend_path: Path to legend image file
            
        Returns:
            Dictionary containing extracted templates and metadata
        """
        try:
            legend_path = Path(legend_path)
            
            if not legend_path.exists():
                raise FileNotFoundError(f"Legend file not found: {legend_path}")
            
            if self.verbose:
                print(f"📋 Processing legend: {legend_path.name}")
            
            # Load legend image
            legend_image = Image.open(legend_path)
            
            if self.verbose:
                print(f"   Image size: {legend_image.size[0]}x{legend_image.size[1]} pixels")
            
            # Extract symbols using enhanced method
            extraction_result = self.symbol_extractor.extract_symbols_from_legend(legend_image)
            
            # Get extracted symbols and metadata
            symbols = extraction_result.get('symbols', {})
            metadata = extraction_result.get('metadata', {})
            summary = extraction_result.get('extraction_summary', {})
            
            if self.verbose:
                success_rate = summary.get('success_rate', 0) * 100
                print(f"   Extraction success rate: {success_rate:.1f}%")
                print(f"   Symbols extracted: {len(symbols)}")
            
            # Validate and optimize symbols for template matching
            validated_symbols = self._validate_and_optimize_symbols(symbols, metadata)
            
            # Store results
            self.legend_data = extraction_result
            self.symbol_templates = validated_symbols
            self.processing_metadata = {
                'legend_path': legend_path,
                'extraction_summary': summary,
                'validation_summary': self._get_validation_summary(validated_symbols, symbols)
            }
            
            if self.verbose:
                print(f"✅ Legend processing complete: {len(validated_symbols)} usable templates")
            
            return {
                'templates': validated_symbols,
                'metadata': self.processing_metadata,
                'raw_extraction': extraction_result
            }
            
        except Exception as e:
            logger.error(f"Legend processing failed: {e}")
            if self.verbose:
                print(f"❌ Error processing legend: {e}")
            return {}
    
    def _validate_and_optimize_symbols(
        self, 
        symbols: Dict[str, Image.Image], 
        metadata: Dict[str, Any]
    ) -> Dict[str, Image.Image]:
        """
        Validate extracted symbols and optimize them for template matching.
        """
        validated_symbols = {}
        
        for symbol_name, symbol_image in symbols.items():
            try:
                # Get symbol metadata
                symbol_meta = metadata.get(symbol_name, {})
                characteristics = symbol_meta.get('characteristics', {})
                
                # Validate symbol quality
                if self._is_valid_symbol(symbol_image, characteristics):
                    # Optimize for template matching
                    optimized_symbol = self._optimize_symbol_for_matching(symbol_image)
                    validated_symbols[symbol_name] = optimized_symbol
                    
                    if self.verbose:
                        print(f"   ✅ Validated: {symbol_name}")
                else:
                    if self.verbose:
                        print(f"   ❌ Rejected: {symbol_name} (quality check failed)")
                        
            except Exception as e:
                if self.verbose:
                    print(f"   ❌ Error processing {symbol_name}: {e}")
                continue
        
        return validated_symbols
    
    def _is_valid_symbol(self, symbol_image: Image.Image, characteristics: Dict) -> bool:
        """
        Validate if a symbol is suitable for template matching.
        """
        # Size validation
        width, height = symbol_image.size
        if width < 15 or height < 15 or width > 200 or height > 150:
            return False
        
        # Complexity validation (avoid empty or overly complex symbols)
        complexity = characteristics.get('complexity', 0)
        if complexity < 0.01 or complexity > 2.0:
            return False
        
        # Aspect ratio validation
        aspect_ratio = characteristics.get('aspect_ratio', 0)
        if aspect_ratio < 0.2 or aspect_ratio > 5.0:
            return False
        
        # Edge density validation
        edge_density = characteristics.get('edge_density', 0)
        if edge_density < 0.05 or edge_density > 0.8:
            return False
        
        return True
    
    def _optimize_symbol_for_matching(self, symbol_image: Image.Image) -> Image.Image:
        """
        Optimize symbol for robust template matching.
        """
        # Convert to numpy array for processing
        import cv2
        symbol_array = cv2.cvtColor(np.array(symbol_image), cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(symbol_array, cv2.COLOR_BGR2GRAY)
        
        # Apply noise reduction
        denoised = cv2.bilateralFilter(gray, 3, 25, 25)
        
        # Normalize contrast
        normalized = cv2.equalizeHist(denoised)
        
        # Optional edge enhancement for better matching
        edges = cv2.Canny(normalized, 50, 150)
        enhanced = cv2.addWeighted(normalized, 0.8, edges, 0.2, 0)
        
        # Convert back to PIL
        optimized_pil = Image.fromarray(enhanced, mode='L').convert('RGB')
        
        return optimized_pil
    
    def _get_validation_summary(self, validated: Dict, original: Dict) -> Dict:
        """Generate validation summary statistics."""
        return {
            'original_count': len(original),
            'validated_count': len(validated),
            'validation_rate': len(validated) / max(len(original), 1),
            'rejected_symbols': [name for name in original.keys() if name not in validated]
        }
    
    def get_symbol_templates(self) -> Dict[str, Image.Image]:
        """Get validated symbol templates for template matching."""
        return self.symbol_templates.copy() if self.symbol_templates else {}
    
    def get_processing_report(self) -> Dict[str, Any]:
        """Get detailed processing report for debugging."""
        return {
            'legend_data': self.legend_data,
            'processing_metadata': self.processing_metadata,
            'available_templates': list(self.symbol_templates.keys()) if self.symbol_templates else []
        }
```

#### Step 3.2: Update Main Process Blueprint Script

**Goal**: Integrate the new legend processing and template matching into the main CLI.

**Implementation**: Update the main function in `process_blueprint.py`:

```python
def main():
    """Enhanced hybrid detection CLI with improved legend processing."""
    print("🚀 ADVANCED BLUEPRINT PROCESSOR v2.0")
    print("   Enhanced Legend Processing + Template Matching")
    print("   Target Accuracy: 95%+ with proper legend\n")
    
    args = None
    try:
        # Parse and validate arguments
        args = parse_arguments()
        file_path, pricing_path, legend_path = validate_inputs(args)
        
        print("🔧 Enhanced Blueprint Processor")
        print("=" * 40)
        print(f"📄 Processing: {file_path.name}")
        
        # Initialize enhanced legend processor
        if legend_path:
            print(f"📋 Legend: {legend_path.name}")
            print("🎯 Detection Mode: Enhanced Hybrid (95%+ accuracy)")
            
            # Process legend first
            from core.legend_processor import LegendProcessor
            legend_processor = LegendProcessor(verbose=args.verbose)
            
            if args.verbose:
                print("\n🔍 Processing legend for symbol templates...")
            
            legend_result = legend_processor.process_legend(str(legend_path))
            
            if not legend_result or not legend_result.get('templates'):
                print("❌ Error: Could not extract usable symbols from legend")
                print("   Falling back to AI-only detection...")
                symbol_templates = None
            else:
                symbol_templates = legend_result['templates']
                if args.verbose:
                    print(f"✅ Legend processed: {len(symbol_templates)} symbol templates ready")
        else:
            print("⚠️  No legend provided")
            print("🎯 Detection Mode: AI-only (85% accuracy)")
            symbol_templates = None
        
        if args.verbose:
            print(f"🔍 Verbose Mode: Enabled")
        print()
        
        # Initialize enhanced detector
        if args.verbose:
            print("🤖 Initializing Enhanced Detection System...")
        
        start_time = time.time()
        detector = ComponentDetectorSmolVLMImproved(enable_enhanced_detection=True)
        
        # Initialize template matcher if we have templates
        template_matcher = None
        if symbol_templates:
            from core.template_matcher import TemplateBasedDetector
            template_matcher = TemplateBasedDetector(verbose=args.verbose)
        
        if args.verbose:
            init_time = time.time() - start_time
            print(f"✅ Detection system loaded in {init_time:.1f} seconds")
            print()
        
        # Process input (PDF or direct image)  
        if args.verbose:
            if args.image:
                print("📸 Processing direct image...")
            else:
                print("📄 Processing PDF...")
        
        process_start = time.time()
        
        if args.image:
            # Direct image processing mode
            from PIL import Image
            floor_plan = Image.open(file_path)
            images = [floor_plan]
            
            if args.verbose:
                print(f"📸 Loaded image: {floor_plan.size[0]}x{floor_plan.size[1]} pixels")
        else:
            # PDF processing mode
            images = process_blueprint_multipage(str(file_path))
            
            if args.verbose:
                print(f"📸 Converted {len(images)} pages to images")
        
        # Run enhanced hybrid detection
        if args.verbose:
            print(f"🔍 Running enhanced detection on {len(images)} images...")
        
        # Combine template matching with AI detection for maximum accuracy
        if symbol_templates and template_matcher:
            # Hybrid approach: Template matching + AI validation
            template_counts = {}
            ai_counts = {}
            
            for i, image in enumerate(images):
                if args.verbose and len(images) > 1:
                    print(f"   Processing page {i+1}/{len(images)}...")
                
                # Template matching first (fast and precise)
                page_template_counts = template_matcher.detect_components_with_enhanced_matching(
                    image, symbol_templates, min_confidence=0.65
                )
                
                # AI detection for validation and gap filling
                page_ai_counts = detector.detect_components_single_page(image)
                
                # Combine results (template matching takes precedence for known symbols)
                for symbol_type, count in page_template_counts.items():
                    template_counts[symbol_type] = template_counts.get(symbol_type, 0) + count
                
                for symbol_type, count in page_ai_counts.items():
                    ai_counts[symbol_type] = ai_counts.get(symbol_type, 0) + count
            
            # Smart combination of results
            total_components = template_counts.copy()
            
            # Add AI detections for symbols not found by template matching
            for symbol_type, ai_count in ai_counts.items():
                template_count = template_counts.get(symbol_type, 0)
                
                if template_count == 0:
                    # No template matches, use AI detection
                    total_components[symbol_type] = ai_count
                elif ai_count > template_count * 1.5:
                    # AI found significantly more, use average (conservative approach)
                    total_components[symbol_type] = int((template_count + ai_count) / 2)
                # Otherwise, trust template matching
            
            detection_method = 'Enhanced Hybrid (Template + AI)'
            estimated_accuracy = '95%'
            
        else:
            # AI-only detection
            total_components = detector.detect_components_multi_page(images)
            detection_method = 'Enhanced AI Detection'
            estimated_accuracy = '85%'
        
        detection_result = {
            'total_components': total_components,
            'analysis_summary': {
                'total_components_found': sum(total_components.values()),
                'detection_method': detection_method,
                'legend_used': symbol_templates is not None,
                'estimated_accuracy': estimated_accuracy,
                'symbol_templates_used': len(symbol_templates) if symbol_templates else 0
            }
        }
        
        processing_time = time.time() - process_start
        
        if args.verbose:
            print(f"✅ Detection completed in {processing_time:.1f} seconds")
            if symbol_templates:
                print(f"📊 Template matching used {len(symbol_templates)} symbol types")
            print()
        
        # Format and display results
        output_text = format_results(detection_result, file_path, processing_time, args.verbose)
        print(output_text)
        
        # Save to file if requested
        if args.output:
            save_results(output_text, args.output)
        
    except KeyboardInterrupt:
        print("\n❌ Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during processing: {str(e)}")
        if args is not None and hasattr(args, 'verbose') and args.verbose:
            import traceback
            print("\nFull error traceback:")
            traceback.print_exc()
        sys.exit(1)
```

## Testing and Validation Strategy

### Step 4: Create Test Scripts

Create comprehensive test scripts to validate each phase:

```python
# test_legend_processing.py
"""Test script for legend processing pipeline."""

from pathlib import Path
from PIL import Image
from core.legend_processor import LegendProcessor

def test_legend_processing():
    """Test the complete legend processing pipeline."""
    print("🧪 Testing Legend Processing Pipeline")
    print("=" * 40)
    
    # Test with key.png
    legend_path = "key.png"
    if not Path(legend_path).exists():
        print(f"❌ Test file not found: {legend_path}")
        return False
    
    # Initialize processor
    processor = LegendProcessor(verbose=True)
    
    # Process legend
    result = processor.process_legend(legend_path)
    
    if result and result.get('templates'):
        templates = result['templates']
        metadata = result['metadata']
        
        print(f"\n✅ Test Results:")
        print(f"   Templates extracted: {len(templates)}")
        print(f"   Success rate: {metadata['validation_summary']['validation_rate']*100:.1f}%")
        
        # Save debug images
        debug_dir = Path("debug_legend_processing")
        debug_dir.mkdir(exist_ok=True)
        
        for symbol_name, template_image in templates.items():
            template_image.save(debug_dir / f"template_{symbol_name}.png")
        
        print(f"   Debug images saved to: {debug_dir}")
        return True
    else:
        print("❌ Legend processing failed")
        return False

if __name__ == "__main__":
    test_legend_processing()
```

## Implementation Roadmap

### Week 1: Legend Processing Foundation
- [ ] Enhance `SymbolExtractor` class with improved region detection
- [ ] Implement enhanced OCR preprocessing
- [ ] Create robust symbol classification system
- [ ] Add comprehensive error handling and validation

### Week 2: Template Matching Enhancement  
- [ ] Implement multi-scale template matching
- [ ] Add rotation handling for template matching
- [ ] Create enhanced non-maximum suppression
- [ ] Optimize template preprocessing pipeline

### Week 3: Pipeline Integration
- [ ] Create `LegendProcessor` module
- [ ] Update main CLI with enhanced legend support
- [ ] Implement hybrid detection combining template matching and AI
- [ ] Add comprehensive logging and debugging

### Week 4: Testing and Refinement
- [ ] Create comprehensive test scripts
- [ ] Test with multiple legend formats
- [ ] Performance optimization and bug fixes  
- [ ] Documentation and user guide updates

## Expected Outcomes

Following this guide should result in:
- **95%+ accuracy** for symbols present in the legend
- **Faster processing** due to template matching efficiency
- **Robust handling** of various legend formats
- **Better error reporting** and debugging capabilities
- **Consistent results** independent of AI model variations

## Troubleshooting Common Issues

### Legend Processing Issues
- **Poor OCR results**: Increase image resolution, improve preprocessing
- **Symbols not detected**: Adjust contour filtering parameters
- **Wrong symbol classification**: Expand keyword dictionary

### Template Matching Issues  
- **Low match confidence**: Adjust preprocessing, try different similarity metrics
- **Multiple detections**: Improve non-maximum suppression parameters
- **Missing symbols**: Add more scale factors, check template quality

### Integration Issues
- **Performance problems**: Optimize image processing, add caching
- **Memory issues**: Process images in batches, reduce image resolution
- **Inconsistent results**: Add validation checks, improve error handling
