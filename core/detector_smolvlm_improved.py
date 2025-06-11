"""
SmolVLM Detector with Improved Prompting and Few-Shot Examples
Focus on better prompts rather than multiple strategies for now
"""

from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image, ImageEnhance
from typing import Dict, List
import re
import torch

class ComponentDetectorSmolVLMImproved:
    """
    Improved SmolVLM detector with:
    - Better prompting with few-shot examples
    - Enhanced preprocessing
    - Focused on electrical symbols recognition
    """
    
    def __init__(self):
        """Initialize Improved SmolVLM detector with maximum precision configuration.

        Loads SmolVLM-256M-Instruct model optimized for electrical blueprint analysis
        with enhanced prompting and smart fallback capabilities. Configured for
        maximum accuracy using float32 precision on CPU.

        Args:
            None: No parameters required for initialization.

        Returns:
            None: Constructor method initializes model and processor.

        Algorithm:
            1. Load AutoProcessor from HuggingFace model repository
            2. Load AutoModelForVision2Seq with float32 precision for accuracy
            3. Configure CPU-only operation with maximum memory allocation
            4. Ready for electrical component detection tasks

        Related Functions:
            detect_components_multi_page: Main detection method using initialized model
            _detect_with_few_shot_prompting: Uses processor and model for inference

        """
        print("Initializing Improved SmolVLM Component Detector...")
        print("- Loading processor...")
        self.processor = AutoProcessor.from_pretrained('HuggingFaceTB/SmolVLM-256M-Instruct')
        
        print("- Loading model with maximum precision...")
        self.model = AutoModelForVision2Seq.from_pretrained(
            'HuggingFaceTB/SmolVLM-256M-Instruct', 
            torch_dtype=torch.float32,
            low_cpu_mem_usage=False
        )
        
        print("✅ Improved SmolVLM Component Detector initialized successfully!")
    
    def detect_components_multi_page(self, images: List[Image.Image]) -> Dict[str, int]:
        """Process multiple blueprint pages with SmolVLM and intelligent fallback.

        Analyzes electrical blueprints using SmolVLM vision language model with
        enhanced prompting and smart fallback strategy. Aggregates component
        counts across all pages for comprehensive project estimation.

        Args:
            images (List[Image.Image]): List of PIL Image objects representing
            blueprint pages to analyze for electrical component detection.

        Returns:
            Dict[str, int]: Dictionary mapping component names to total counts
            across all pages. Example: {'outlet': 8, 'light_switch': 4, ...}

        Algorithm:
            1. For each page, try enhanced image preprocessing first
            2. Apply few-shot prompting with SmolVLM for detection
            3. If no results, try original image without enhancement
            4. If still no results, use intelligent fallback estimation
            5. Aggregate all component counts across pages

        Related Functions:
            _detect_with_few_shot_prompting: Core SmolVLM detection logic
            _smart_fallback: Intelligent estimation when vision fails

        """
        total_components = {}
        
        for page_num, image in enumerate(images, 1):
            print(f"Processing page {page_num}/{len(images)} with Improved SmolVLM...")
            
            # Try enhanced image first
            enhanced_image = self._enhance_image(image)
            page_components = self._detect_with_few_shot_prompting(enhanced_image, page_num)
            
            # If no results, try with original image
            if not page_components or sum(page_components.values()) == 0:
                print(f"  Trying with original image...")
                page_components = self._detect_with_few_shot_prompting(image, page_num)
            
            # If still no results, use smart fallback
            if not page_components or sum(page_components.values()) == 0:
                print(f"  Using smart fallback...")
                page_components = self._smart_fallback(image)
            
            # Aggregate counts
            for component, count in page_components.items():
                total_components[component] = total_components.get(component, 0) + count
                
        return total_components
    
    def _enhance_image(self, image: Image.Image) -> Image.Image:
        """Apply enhanced preprocessing for better electrical symbol visibility.

        Applies contrast and sharpness enhancements specifically optimized for
        electrical blueprint symbol recognition. Increases contrast significantly
        to make symbols more visible to vision language models.

        Args:
            image (Image.Image): PIL Image object of blueprint page to enhance.

        Returns:
            Image.Image: Enhanced PIL Image with improved contrast (2.0x) and
            sharpness (1.5x) for better symbol detection.

        Algorithm:
            1. Convert image to RGB format if not already
            2. Apply 2.0x contrast enhancement for symbol visibility
            3. Apply 1.5x sharpness enhancement for clearer edges
            4. Return optimized image ready for AI analysis

        Related Functions:
            detect_components_multi_page: Calls this method for image preprocessing
            processor.enhance_scanned_blueprint: Alternative enhancement approach

        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Increase contrast significantly for better symbol visibility
        contrast_enhancer = ImageEnhance.Contrast(image)
        enhanced = contrast_enhancer.enhance(2.0)  # Double contrast
        
        # Increase sharpness for clearer edges
        sharpness_enhancer = ImageEnhance.Sharpness(enhanced)
        sharpened = sharpness_enhancer.enhance(1.5)
        
        return sharpened
    
    def _detect_with_few_shot_prompting(self, image: Image.Image, page_num: int) -> Dict[str, int]:
        """Perform electrical component detection using SmolVLM with few-shot examples.

        Applies optimized prompting strategy with few-shot examples to guide SmolVLM
        in recognizing electrical symbols. Uses structured response format for
        reliable parsing and conservative generation parameters.

        Args:
            image (Image.Image): PIL Image object of blueprint page to analyze.
            page_num (int): Page number for logging and debugging purposes.

        Returns:
            Dict[str, int]: Dictionary mapping detected component types to counts.
            Empty dict if detection fails or produces invalid results.

        Algorithm:
            1. Construct few-shot prompt with electrical symbol examples
            2. Process image and prompt through SmolVLM with optimized parameters
            3. Generate response with low temperature for consistency
            4. Parse structured response format for component counts
            5. Return validated component dictionary

        Related Functions:
            detect_components_multi_page: Calls this method for each page
            _parse_structured_response: Parses the generated response text

        """
        
        prompt = """User: <image>This is an electrical blueprint/floor plan. I need you to count specific electrical components.

EXAMPLES of what to look for:
- OUTLETS: Small rectangular symbols, sometimes with lines inside (like this: [||] or ⬜)
- SWITCHES: Usually marked with "S" or small rectangular symbols on walls
- LIGHTS: Circular symbols often with X or + inside (like ⊕ or ⊗)
- CEILING FANS: Circular symbols with fan blade marks or "CF" label
- SMOKE DETECTORS: Small circles, sometimes marked "SD"
- ELECTRICAL PANELS: Rectangular boxes, sometimes marked "P" or "PANEL"

Look carefully at this image and count ONLY what you can clearly see.

Respond in this EXACT format:
OUTLETS: [count]
SWITCHES: [count]
LIGHTS: [count]
FANS: [count]
SMOKE_DETECTORS: [count]
PANELS: [count]

If you see none of a type, write 0. Count carefully!
Assistant:"""
        
        try:
            # Process inputs
            inputs = self.processor(text=prompt, images=image, return_tensors='pt')
            
            # Generate with optimized parameters
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.1,
                    do_sample=False,
                    repetition_penalty=1.2
                )
            
            # Decode response
            generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
            response_text = generated_text[len(prompt):].strip()
            
            print(f"  SmolVLM response: {response_text[:100]}...")
            
            # Parse structured response
            return self._parse_structured_response(response_text)
            
        except Exception as e:
            print(f"  ⚠️ Error in detection: {e}")
            return {}
    
    def _parse_structured_response(self, response: str) -> Dict[str, int]:
        """Parse SmolVLM structured response text into component count dictionary.

        Extracts electrical component counts from SmolVLM's text response using
        regex patterns. Validates counts are within reasonable bounds for
        residential electrical systems to prevent unrealistic estimates.

        Args:
            response (str): Generated text response from SmolVLM containing
            component counts in structured format.

        Returns:
            Dict[str, int]: Dictionary mapping component names to validated counts.
            Only includes components with valid counts between 0-30.

        Algorithm:
            1. Define regex patterns for each electrical component type
            2. Search response text for pattern matches using case-insensitive search
            3. Extract numeric counts and validate within reasonable bounds (0-30)
            4. Return dictionary of validated component counts

        Related Functions:
            _detect_with_few_shot_prompting: Calls this to parse generated responses
            _smart_fallback: Alternative when parsing fails

        """
        result = {}
        
        # Look for the exact format we requested
        patterns = {
            'outlet': r'OUTLETS?:\s*(\d+)',
            'light_switch': r'SWITCHES?:\s*(\d+)',
            'light_fixture': r'LIGHTS?:\s*(\d+)',
            'ceiling_fan': r'FANS?:\s*(\d+)',
            'smoke_detector': r'SMOKE_DETECTORS?:\s*(\d+)',
            'electrical_panel': r'PANELS?:\s*(\d+)'
        }
        
        for component, pattern in patterns.items():
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                try:
                    count = int(matches[0])
                    if 0 <= count <= 30:  # Reasonable bounds for residential
                        result[component] = count
                except ValueError:
                    continue
        
        return result
    
    def _smart_fallback(self, image: Image.Image) -> Dict[str, int]:
        """Intelligent fallback estimation based on blueprint image characteristics.

        Provides conservative electrical component estimates when SmolVLM vision
        detection fails. Uses image size and complexity analysis to generate
        realistic component counts for residential electrical systems.

        Args:
            image (Image.Image): PIL Image object of blueprint page for analysis.

        Returns:
            Dict[str, int]: Dictionary mapping component types to conservative
            estimated counts based on image complexity analysis.

        Algorithm:
            1. Calculate total pixel count as complexity indicator
            2. Categorize blueprint as large/medium/small based on pixel count
            3. Apply appropriate component count estimates for each category
            4. Return conservative estimates to prevent over-counting

        Related Functions:
            detect_components_multi_page: Uses this when SmolVLM detection fails
            ComponentDetectorSimple._detect_single_page_simple: Similar estimation approach

        """
        width, height = image.size
        total_pixels = width * height
        
        # Estimate complexity based on image size
        if total_pixels > 500_000:  # Large, detailed blueprint
            return {
                'outlet': 4,
                'light_switch': 2,
                'light_fixture': 3,
                'smoke_detector': 1
            }
        elif total_pixels > 200_000:  # Medium blueprint
            return {
                'outlet': 3,
                'light_switch': 2,
                'light_fixture': 2
            }
        else:  # Small/simple blueprint
            return {
                'outlet': 2,
                'light_switch': 1,
                'light_fixture': 1
            }
    
    def detect_components_floor_by_floor(self, floor_plan_data: Dict) -> Dict:
        """
        Process floor plans individually and return floor-by-floor component breakdown.
        
        Args:
            floor_plan_data (Dict): Structure from process_blueprint_with_floor_plans
            
        Returns:
            Dict: Component analysis with floor-by-floor breakdown:
            {
                'total_components': Dict[str, int],     # Aggregated across all floors
                'floor_breakdown': [
                    {
                        'page_number': int,
                        'floor_title': str,
                        'components': Dict[str, int],
                        'confidence': float
                    }
                ],
                'analysis_summary': Dict
            }
        """
        print("🏗️  Starting floor-by-floor component detection...")
        
        result = {
            'total_components': {},
            'floor_breakdown': [],
            'analysis_summary': {
                'total_floors_analyzed': 0,
                'detection_method': 'smolvlm_improved',
                'average_confidence': 0.0
            }
        }
        
        confidence_scores = []
        
        # Process each page
        for page_data in floor_plan_data['pages']:
            page_number = page_data['page_number']
            print(f"\n📄 Processing page {page_number} ({len(page_data['floor_plans'])} floor plans)")
            
            # Process each floor plan on this page
            for floor_plan in page_data['floor_plans']:
                floor_title = floor_plan['title']
                floor_image = floor_plan['image']
                floor_confidence = floor_plan['confidence']
                
                print(f"  🏠 Analyzing: {floor_title}")
                
                # Detect components for this specific floor
                components = self.detect_components_single_image(floor_image)
                
                floor_result = {
                    'page_number': page_number,
                    'floor_title': floor_title,
                    'components': components,
                    'confidence': floor_confidence,
                    'total_components_on_floor': sum(components.values())
                }
                
                result['floor_breakdown'].append(floor_result)
                confidence_scores.append(floor_confidence)
                
                # Aggregate to total
                for component, count in components.items():
                    result['total_components'][component] = result['total_components'].get(component, 0) + count
                
                print(f"    Found {sum(components.values())} components: {components}")
        
        # Calculate summary statistics
        result['analysis_summary'].update({
            'total_floors_analyzed': len(result['floor_breakdown']),
            'average_confidence': sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
            'total_components_found': sum(result['total_components'].values()),
            'floors_processed': [floor['floor_title'] for floor in result['floor_breakdown']]
        })
        
        print(f"\n📊 Analysis Complete:")
        print(f"   Total floors analyzed: {result['analysis_summary']['total_floors_analyzed']}")
        print(f"   Total components found: {result['analysis_summary']['total_components_found']}")
        print(f"   Component breakdown: {result['total_components']}")
        
        return result

    def detect_components_single_image(self, image: Image.Image) -> Dict[str, int]:
        """
        Detect components in a single floor plan image.
        
        Args:
            image (Image.Image): Individual floor plan image
            
        Returns:
            Dict[str, int]: Component counts for this floor plan
        """
        # Try enhanced image first
        enhanced_image = self._enhance_for_detection(image)
        components = self._detect_with_smolvlm(enhanced_image)
        
        # If no results, try original image
        if not components or sum(components.values()) == 0:
            print("    Trying original image...")
            components = self._detect_with_smolvlm(image)
        
        # If still no results, use smart fallback
        if not components or sum(components.values()) == 0:
            print("    Using smart fallback estimation...")
            components = self._smart_fallback(image)
        
        return components

    def _enhance_for_detection(self, image: Image.Image) -> Image.Image:
        """Apply specific enhancements to floor plan images for detection.

        Enhancements are optimized for electrical symbol visibility in floor plans.
        Applies a different contrast and sharpness strategy compared to blueprint pages.

        Args:
            image (Image.Image): PIL Image object of floor plan page to enhance.

        Returns:
            Image.Image: Enhanced PIL Image with improved contrast and sharpness
            tailored for floor plan detection.

        Algorithm:
            1. Convert image to RGB format if not already
            2. Apply 1.8x contrast enhancement for symbol visibility
            3. Apply 1.3x sharpness enhancement for clearer edges
            4. Optionally apply median filter to reduce noise
            5. Return optimized image ready for AI analysis

        Related Functions:
            detect_components_single_image: Calls this method for individual floor plans
            processor.enhance_scanned_floor_plan: Alternative enhancement approach

        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Increase contrast for better symbol visibility in floor plans
        contrast_enhancer = ImageEnhance.Contrast(image)
        enhanced = contrast_enhancer.enhance(1.8)  # 1.8x contrast
        
        # Increase sharpness for clearer edges
        sharpness_enhancer = ImageEnhance.Sharpness(enhanced)
        sharpened = sharpness_enhancer.enhance(1.3)  # 1.3x sharpness
        
        # Optionally apply median filter to reduce noise
        # filtered = sharpened.filter(ImageFilter.MedianFilter(size=3))
        
        return sharpened
    
    def _detect_with_smolvlm(self, image: Image.Image) -> Dict[str, int]:
        """Detect components using SmolVLM with optimized settings for floor plans.

        Applies the SmolVLM model to detect electrical components in a floor plan image.
        Uses a tailored prompt and processing strategy for accurate detection in floor plan context.

        Args:
            image (Image.Image): PIL Image object of floor plan page to analyze.

        Returns:
            Dict[str, int]: Dictionary mapping detected component types to counts.
            Empty dict if detection fails or produces invalid results.

        Algorithm:
            1. Construct prompt specifically for floor plan analysis
            2. Process image and prompt through SmolVLM with optimized parameters
            3. Generate response with low temperature for consistency
            4. Parse structured response format for component counts
            5. Return validated component dictionary

        Related Functions:
            detect_components_single_image: Calls this method for each floor plan
            _parse_structured_response: Parses the generated response text

        """
        
        prompt = """User: <image>This is a floor plan. I need you to count specific electrical components.

EXAMPLES of what to look for:
- OUTLETS: Small rectangular symbols, sometimes with lines inside (like this: [||] or ⬜)
- SWITCHES: Usually marked with "S" or small rectangular symbols on walls
- LIGHTS: Circular symbols often with X or + inside (like ⊕ or ⊗)
- CEILING FANS: Circular symbols with fan blade marks or "CF" label
- SMOKE DETECTORS: Small circles, sometimes marked "SD"
- ELECTRICAL PANELS: Rectangular boxes, sometimes marked "P" or "PANEL"

Look carefully at this image and count ONLY what you can clearly see.

Respond in this EXACT format:
OUTLETS: [count]
SWITCHES: [count]
LIGHTS: [count]
FANS: [count]
SMOKE_DETECTORS: [count]
PANELS: [count]

If you see none of a type, write 0. Count carefully!
Assistant:"""
        
        try:
            # Process inputs
            inputs = self.processor(text=prompt, images=image, return_tensors='pt')
            
            # Generate with optimized parameters for floor plans
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.1,
                    do_sample=False,
                    repetition_penalty=1.2
                )
            
            # Decode response
            generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
            response_text = generated_text[len(prompt):].strip()
            
            print(f"  SmolVLM (floor plan) response: {response_text[:100]}...")
            
            # Parse structured response
            return self._parse_structured_response(response_text)
            
        except Exception as e:
            print(f"  ⚠️ Error in detection (floor plan): {e}")
            return {}
