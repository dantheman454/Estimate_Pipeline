"""
Symbol Extraction Module for Blueprint Pipeline
Analyzes electrical legend images to extract individual symbols and their classifications
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
from typing import Dict, List, Tuple, Optional
import re

# Import cv2 with path workaround
try:
    import cv2
except ImportError:
    venv_path = Path(__file__).parent.parent / "venv_cli" / "lib" / "python3.13" / "site-packages"
    if venv_path.exists():
        sys.path.insert(0, str(venv_path))
        import cv2
    else:
        raise ImportError("OpenCV not found")

# Optional OCR import
try:
    import pytesseract
    HAS_OCR = True
except ImportError:
    HAS_OCR = False


class SymbolExtractor:
    """
    Advanced symbol extractor for electrical legend analysis.
    
    Features:
    - Individual symbol detection using contour analysis
    - Symbol classification based on adjacent text
    - Template matching for symbol recognition
    - Symbol type categorization for electrical components
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.extracted_symbols = {}
        
        # Common electrical symbol classifications
        self.symbol_classifications = {
            'outlet': ['outlet', 'receptacle', 'duplex', 'gfci', 'gfi'],
            'switch': ['switch', 'toggle', 'dimmer', '3-way', '4-way'],
            'light': ['light', 'fixture', 'ceiling', 'pendant', 'chandelier', 'recessed'],
            'fan': ['fan', 'ceiling fan', 'exhaust'],
            'smoke_detector': ['smoke', 'detector', 'alarm', 'fire'],
            'panel': ['panel', 'breaker', 'electrical panel', 'distribution'],
            'junction': ['junction', 'box', 'j-box'],
            'transformer': ['transformer', 'xfmr', 'step-down', 'step-up']
        }
        
        if self.verbose:
            print(f"🔧 SymbolExtractor initialized")
            print(f"   OpenCV version: {cv2.__version__}")
            print(f"   OCR available: {HAS_OCR}")
    
    def extract_symbols_from_legend(self, legend_image: Image.Image) -> Dict[str, Image.Image]:
        """
        Extract individual symbols from electrical legend image.
        
        Args:
            legend_image: PIL Image of the electrical legend
            
        Returns:
            Dict mapping symbol types to extracted symbol images
        """
        if self.verbose:
            print("🔍 Analyzing legend for symbol extraction...")
        
        # Convert to OpenCV format
        legend_cv = cv2.cvtColor(np.array(legend_image), cv2.COLOR_RGB2BGR)
        
        # Find symbol regions
        symbol_regions = self._detect_symbol_regions(legend_cv)
        
        if self.verbose:
            print(f"   Found {len(symbol_regions)} symbol regions")
        
        # Extract and classify symbols
        extracted_symbols = {}
        for i, (symbol_bbox, text_bbox, symbol_img, text_content) in enumerate(symbol_regions):
            symbol_type = self._classify_symbol_text(text_content)
            
            if symbol_type:
                # Convert back to PIL
                symbol_pil = Image.fromarray(cv2.cvtColor(symbol_img, cv2.COLOR_BGR2RGB))
                extracted_symbols[symbol_type] = symbol_pil
                
                if self.verbose:
                    print(f"   Symbol {i+1}: {symbol_type} (text: '{text_content.strip()}')")
            else:
                if self.verbose:
                    print(f"   Symbol {i+1}: Unknown (text: '{text_content.strip()}')")
        
        self.extracted_symbols = extracted_symbols
        return extracted_symbols
    
    def _detect_symbol_regions(self, legend_cv: np.ndarray) -> List[Tuple]:
        """
        Detect individual symbol and text pairs in the legend.
        
        Returns:
            List of tuples: (symbol_bbox, text_bbox, symbol_image, text_content)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(legend_cv, cv2.COLOR_BGR2GRAY)
        
        # Apply binary threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size and aspect ratio
        symbol_candidates = []
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            # Filter by size (symbols should be reasonably sized)
            if (10 < w < 200 and 10 < h < 100 and area > 50):
                symbol_candidates.append((x, y, w, h, area))
        
        # Sort candidates by y-coordinate (top to bottom) then x-coordinate (left to right)
        symbol_candidates.sort(key=lambda c: (c[1], c[0]))
        
        # Extract symbol and adjacent text
        symbol_regions = []
        
        for x, y, w, h, area in symbol_candidates:
            # Extract symbol region
            symbol_img = legend_cv[y:y+h, x:x+w]
            
            # Find adjacent text region
            text_region, text_content = self._find_adjacent_text(legend_cv, x, y, w, h)
            
            symbol_bbox = (x, y, w, h)
            text_bbox = text_region
            
            symbol_regions.append((symbol_bbox, text_bbox, symbol_img, text_content))
        
        return symbol_regions
    
    def _find_adjacent_text(self, legend_cv: np.ndarray, x: int, y: int, w: int, h: int) -> Tuple[Optional[Tuple], str]:
        """
        Find text adjacent to a symbol region.
        
        Args:
            legend_cv: OpenCV image
            x, y, w, h: Symbol bounding box coordinates
            
        Returns:
            Tuple of (text_region_bbox, text_content)
        """
        img_height, img_width = legend_cv.shape[:2]
        
        # Define search regions around the symbol
        # Check to the right of the symbol (most common)
        right_x = x + w + 5
        right_w = min(200, img_width - right_x)
        right_region = (right_x, y, right_w, h)
        
        # Check below the symbol
        below_y = y + h + 5
        below_h = min(50, img_height - below_y)
        below_region = (x, below_y, w * 2, below_h)
        
        # Check above the symbol
        above_y = max(0, y - 50)
        above_h = y - above_y
        above_region = (x, above_y, w * 2, above_h)
        
        search_regions = [right_region, below_region, above_region]
        
        best_text = ""
        best_region = None
        
        for region in search_regions:
            rx, ry, rw, rh = region
            
            # Ensure region is within image bounds
            if (rx >= 0 and ry >= 0 and 
                rx + rw <= img_width and ry + rh <= img_height and
                rw > 0 and rh > 0):
                
                text_content = self._extract_text_from_region(legend_cv, rx, ry, rw, rh)
                
                if text_content and len(text_content.strip()) > len(best_text.strip()):
                    best_text = text_content
                    best_region = region
        
        return best_region, best_text
    
    def _extract_text_from_region(self, legend_cv: np.ndarray, x: int, y: int, w: int, h: int) -> str:
        """Extract text from a specific region using OCR."""
        if not HAS_OCR:
            return ""
        
        try:
            # Extract region
            region = legend_cv[y:y+h, x:x+w]
            
            # Convert to PIL format for pytesseract
            region_pil = Image.fromarray(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
            
            # Apply preprocessing for better OCR
            region_pil = region_pil.convert('L')  # Convert to grayscale
            region_pil = region_pil.resize((region_pil.width * 2, region_pil.height * 2), Image.Resampling.LANCZOS)  # Upscale
            
            # Extract text
            text = pytesseract.image_to_string(region_pil, config='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-/() ')
            
            return text.strip()
            
        except Exception as e:
            if self.verbose:
                print(f"   Text extraction failed: {e}")
            return ""
    
    def _classify_symbol_text(self, text: str) -> Optional[str]:
        """
        Classify symbol based on adjacent text content.
        
        Args:
            text: Text content found near the symbol
            
        Returns:
            Symbol type string or None if classification fails
        """
        if not text or len(text.strip()) < 2:
            return None
        
        text_lower = text.lower().strip()
        
        # Check each classification category
        for symbol_type, keywords in self.symbol_classifications.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return symbol_type
        
        # Additional pattern matching for common electrical terminology
        if re.search(r'\d+a|amp|circuit|breaker', text_lower):
            return 'panel'
        
        if re.search(r'110v|120v|220v|240v|volt', text_lower):
            return 'outlet'
        
        if re.search(r'led|fluorescent|incandescent', text_lower):
            return 'light'
        
        if re.search(r'cfm|exhaust', text_lower):
            return 'fan'
        
        return None
    
    def get_symbol_templates(self) -> Dict[str, Image.Image]:
        """
        Get extracted symbol templates for template matching.
        
        Returns:
            Dictionary mapping symbol types to template images
        """
        return self.extracted_symbols.copy()
    
    def save_extracted_symbols(self, output_dir: str = "extracted_symbols"):
        """
        Save all extracted symbols as individual image files.
        
        Args:
            output_dir: Directory to save symbol images
        """
        if not self.extracted_symbols:
            print("⚠️  No symbols to save. Run extract_symbols_from_legend first.")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        for symbol_type, symbol_image in self.extracted_symbols.items():
            symbol_path = output_path / f"symbol_{symbol_type}.png"
            symbol_image.save(symbol_path)
            
            if self.verbose:
                print(f"💾 Saved symbol: {symbol_path}")
    
    def analyze_symbol_characteristics(self, symbol_image: Image.Image) -> Dict:
        """
        Analyze characteristics of a symbol for template matching.
        
        Args:
            symbol_image: PIL Image of the symbol
            
        Returns:
            Dictionary of symbol characteristics
        """
        # Convert to OpenCV
        symbol_cv = cv2.cvtColor(np.array(symbol_image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(symbol_cv, cv2.COLOR_BGR2GRAY)
        
        # Calculate basic characteristics
        height, width = gray.shape
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (width * height)
        
        # Contour analysis
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_count = len(contours)
        
        # Aspect ratio
        aspect_ratio = width / max(height, 1)
        
        # Complexity score (based on contour count and edge density)
        complexity = contour_count * edge_density
        
        return {
            'width': width,
            'height': height,
            'aspect_ratio': aspect_ratio,
            'edge_density': edge_density,
            'contour_count': contour_count,
            'complexity': complexity
        }


def main():
    """Demo and testing of the symbol extraction module."""
    print("⚙️  Symbol Extractor - Testing with extracted legend")
    print("=" * 50)
    
    extractor = SymbolExtractor(verbose=True)
    
    # Test with key.png (standalone legend)
    key_path = "key.png"
    if Path(key_path).exists():
        print(f"\n📄 Processing: {key_path}")
        try:
            legend_image = Image.open(key_path)
            symbols = extractor.extract_symbols_from_legend(legend_image)
            
            print(f"✅ Extracted {len(symbols)} symbol types:")
            for symbol_type in symbols.keys():
                print(f"   • {symbol_type}")
            
            # Save extracted symbols
            extractor.save_extracted_symbols("debug_symbols")
            
            # Analyze symbol characteristics
            for symbol_type, symbol_img in symbols.items():
                characteristics = extractor.analyze_symbol_characteristics(symbol_img)
                print(f"📊 {symbol_type}: complexity={characteristics['complexity']:.2f}, aspect={characteristics['aspect_ratio']:.2f}")
            
        except Exception as e:
            print(f"❌ Error processing {key_path}: {e}")
    
    # Test with debug output from image processor
    debug_legend_path = "debug_output/extracted_legend.png"
    if Path(debug_legend_path).exists():
        print(f"\n📄 Processing extracted legend: {debug_legend_path}")
        try:
            legend_image = Image.open(debug_legend_path)
            symbols = extractor.extract_symbols_from_legend(legend_image)
            
            print(f"✅ Extracted {len(symbols)} symbol types from extracted legend")
            
        except Exception as e:
            print(f"❌ Error processing extracted legend: {e}")
    
    print("\n🎯 Symbol Extraction Module is ready!")


if __name__ == "__main__":
    main()
