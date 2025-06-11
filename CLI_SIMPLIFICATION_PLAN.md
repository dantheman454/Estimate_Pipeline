# Blueprint Processor CLI Simplification Plan

## Project Overview
Transform the existing web-based blueprint processor into a simple command-line interface that processes PDF blueprints locally without any web hosting requirements. Focus on getting a working pipeline first, then building the website later.

## Current System Analysis

### Current Architecture
- **FastAPI Web Application** with multiple endpoints
- **SmolVLM-256M-Instruct** AI model (76.9% accuracy)
- **Advanced preprocessing pipeline** (600 DPI, contrast enhancement, sharpening)
- **Floor plan detection** with OCR-based title recognition
- **Professional PDF bill generation** with detailed cost breakdowns
- **SQLite pricing database** for component costs

### Key Components to Preserve
1. **SmolVLM Improved Detector** (`core/detector_smolvlm_improved.py`)
2. **Advanced Image Enhancement** (`core/advanced_preprocessor.py`)
3. **Floor Plan Detection** (`core/floor_plan_detector.py`)
4. **PDF Processing** (`core/processor.py`)

### Components to Remove/Simplify
1. **FastAPI web framework** and all web-related dependencies
2. **Web templates and static files**
3. **Multiple detector variants** (keep only SmolVLM Improved)
4. **Complex billing system** (simplify to console output)

---

## Phase 1: Create Simple CLI Script

### 1.1 Main CLI Script (`process_blueprint.py`)

**Purpose**: Single entry point for command-line blueprint processing

**Features**:
- Accept PDF file path as command-line argument
- Process using SmolVLM Improved detector
- Maintain floor plan detection capability
- Output structured text results to console
- Handle errors gracefully

**Structure**:
```python
#!/usr/bin/env python3
"""
Blueprint Processor - Command Line Interface
Simple local processing of electrical blueprints without web hosting.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

# Core imports (simplified)
from core.processor import process_blueprint_with_floor_plans
from core.detector_smolvlm_improved import ComponentDetectorSmolVLMImproved
from core.pricing import PricingEngine

def main():
    # Parse command line arguments
    # Process blueprint PDF
    # Run SmolVLM detection
    # Display structured results
    pass

if __name__ == "__main__":
    main()
```

### 1.2 Console Output Format

**Structured Text Output**:
```
=== BLUEPRINT PROCESSING RESULTS ===

📄 File: /path/to/blueprint.pdf
🕐 Processed: 2025-06-11 14:30:25
🏗️  Pages Analyzed: 2
🔍 Detection Method: SmolVLM Improved (76.9% accuracy)

📊 FLOOR-BY-FLOOR BREAKDOWN:

🏠 BASEMENT ELECTRICAL PLAN
   • Outlets: 6
   • Light Switches: 4
   • Light Fixtures: 3
   • Smoke Detectors: 2
   • Ceiling Fans: 1
   Total Components: 16

🏠 FIRST FLOOR ELECTRICAL PLAN  
   • Outlets: 8
   • Light Switches: 6
   • Light Fixtures: 4
   • Smoke Detectors: 1
   • Ceiling Fans: 2
   Total Components: 21

💰 COST ESTIMATE SUMMARY:
   • Total Components: 37
   • Material Cost: $1,247.50
   • Estimated Labor: $925.00
   • Project Total: $2,172.50

✅ Processing Complete!
```

---

## Phase 2: Streamline Dependencies

### 2.1 Remove Web Dependencies

**Remove from requirements.txt**:
- `fastapi>=0.104.1`
- `uvicorn[standard]>=0.24.0` 
- `python-multipart>=0.0.6`

**Keep Essential Dependencies**:
- `pdf2image>=1.16.3` (PDF processing)
- `pillow>=10.0.0` (image processing)
- `transformers>=4.40.0` (SmolVLM model)
- `torch>=2.0.0` (AI inference)
- `torchvision>=0.15.0` (vision processing)
- `accelerate>=0.21.0` (model optimization)
- `opencv-python>=4.8.0` (advanced image processing)
- `numpy>=1.21.0` (numerical operations)
- `pytesseract` (OCR for floor plan detection)

### 2.2 Simplified Requirements File

**New `requirements_cli.txt`**:
```txt
# Blueprint Processor CLI - Simplified Dependencies
# Core PDF and image processing
pdf2image>=1.16.3
pillow>=10.0.0

# SmolVLM AI model for electrical component detection
transformers>=4.40.0
torch>=2.0.0
torchvision>=0.15.0
accelerate>=0.21.0

# Advanced image processing for blueprint enhancement
opencv-python>=4.8.0
numpy>=1.21.0

# OCR for floor plan title detection
pytesseract>=0.3.10

# Basic utilities
sqlite3  # Built into Python
argparse  # Built into Python
pathlib  # Built into Python
```

---

## Phase 3: Simplify File Structure

### 3.1 Keep Core Modules

**Essential Files to Preserve**:
```
blueprint-processor/
├── process_blueprint.py              # NEW: Main CLI script
├── requirements_cli.txt               # NEW: Simplified dependencies
├── setup_cli.py                      # NEW: CLI setup script
├── core/
│   ├── __init__.py
│   ├── processor.py                   # PDF processing & floor plan extraction
│   ├── detector_smolvlm_improved.py   # Main AI detector (76.9% accuracy)
│   ├── advanced_preprocessor.py       # Image enhancement pipeline
│   ├── floor_plan_detector.py         # Floor plan detection with OCR
│   ├── pricing.py                     # Component pricing engine
│   └── __pycache__/
├── pricing.db                         # SQLite pricing database
└── logs/                             # Processing logs
```

### 3.2 Remove Web-Related Files

**Files to Remove/Archive**:
- `main.py` (FastAPI application)
- `main_clean.py`
- `start_dev.py`
- All detector variants except SmolVLM Improved
- `generated_bills/` directory (no PDF generation)
- Web templates and static files
- Complex billing system files

---

## Phase 4: One-time Setup Script

### 4.1 Automated Setup (`setup_cli.py`)

**Purpose**: Handle SmolVLM model download and dependency installation

**Features**:
- Install Python dependencies from requirements_cli.txt
- Download SmolVLM-256M-Instruct model (~5GB, one-time)
- Verify model access and CPU compatibility
- Create necessary directories
- Test basic functionality

**Structure**:
```python
#!/usr/bin/env python3
"""
Blueprint Processor CLI - Setup Script
Downloads SmolVLM model and installs dependencies for local processing.
"""

import subprocess
import sys
from pathlib import Path

def install_dependencies():
    """Install required Python packages."""
    pass

def download_smolvlm_model():
    """Download and cache SmolVLM model."""
    pass

def verify_setup():
    """Test that everything works."""
    pass

def main():
    print("🔧 Setting up Blueprint Processor CLI...")
    install_dependencies()
    download_smolvlm_model()
    verify_setup()
    print("✅ Setup complete! Run: python process_blueprint.py your_file.pdf")

if __name__ == "__main__":
    main()
```

---

## Phase 5: Implementation Details

### 5.1 Core Processing Pipeline

**Simplified Workflow**:
1. **PDF Input** → Parse command line arguments, validate file
2. **PDF Processing** → Convert to images using existing processor
3. **Floor Plan Detection** → Use existing OCR-based detection
4. **Image Enhancement** → Apply advanced preprocessing pipeline
5. **AI Detection** → Run SmolVLM Improved detector on each floor
6. **Results Aggregation** → Combine floor-by-floor results
7. **Console Output** → Display structured text results
8. **Cost Estimation** → Basic pricing using existing database

### 5.2 Error Handling Strategy

**Graceful Degradation**:
- PDF parsing errors → Clear error message with file format check
- SmolVLM model issues → Fallback to smart estimation (existing feature)
- Floor plan detection failures → Process as single plan
- Missing pricing data → Use default estimates
- Image processing errors → Retry with original image

### 5.3 Performance Considerations

**Optimizations**:
- **Lazy Loading**: Load SmolVLM model only when needed
- **Memory Management**: Process pages sequentially, not in parallel
- **Model Caching**: Reuse loaded model for multiple files
- **Progress Indicators**: Show processing status for long operations

---

## Phase 6: Testing & Validation

### 6.1 Test Cases

**Basic Functionality**:
1. **Single Floor Plan**: Simple PDF with one electrical plan
2. **Multi-Floor Plan**: Complex PDF with multiple floors
3. **Error Cases**: Invalid PDFs, missing files, corrupted images
4. **Performance**: Large files, multiple pages, memory usage

### 6.2 Success Criteria

**Must Work**:
- ✅ Process any electrical blueprint PDF
- ✅ Detect electrical components with reasonable accuracy
- ✅ Maintain floor plan detection capability
- ✅ Provide cost estimates
- ✅ Handle errors gracefully
- ✅ Complete processing in <30 seconds for typical blueprints

---

## Expected Usage Examples

### Basic Usage
```bash
# Install and setup (one-time)
python setup_cli.py

# Process a blueprint
python process_blueprint.py /path/to/blueprint.pdf

# Process with verbose output
python process_blueprint.py /path/to/blueprint.pdf --verbose

# Process and save results to file
python process_blueprint.py /path/to/blueprint.pdf --output results.txt
```

### Advanced Usage
```bash
# Process multiple files
python process_blueprint.py file1.pdf file2.pdf file3.pdf

# Skip floor plan detection (process as single plan)
python process_blueprint.py blueprint.pdf --single-plan

# Use custom pricing database
python process_blueprint.py blueprint.pdf --pricing custom_prices.db
```

---

## Implementation Priority

### High Priority (Must Have)
1. ✅ **Main CLI script** with argument parsing
2. ✅ **SmolVLM Improved integration** (core AI functionality)
3. ✅ **Floor plan detection preservation** (key differentiator)
4. ✅ **Console output formatting** (user experience)
5. ✅ **Error handling** (reliability)

### Medium Priority (Nice to Have)
1. ⭕ **Setup automation script** (ease of installation)
2. ⭕ **Progress indicators** (user feedback)
3. ⭕ **Multiple file processing** (batch operations)
4. ⭕ **Output format options** (flexibility)

### Low Priority (Future Enhancement)
1. ⚪ **Configuration file support** (customization)
2. ⚪ **Logging improvements** (debugging)
3. ⚪ **Performance optimizations** (speed)
4. ⚪ **Additional output formats** (JSON, CSV)

---

## Technical Implementation Notes

### Model Integration
- **Preserve** existing SmolVLM Improved detector unchanged
- **Maintain** 76.9% accuracy with few-shot prompting
- **Keep** smart fallback system for reliability
- **Use** existing image enhancement pipeline

### Floor Plan Detection
- **Preserve** OCR-based title detection
- **Maintain** automatic boundary detection
- **Keep** confidence scoring system
- **Simplify** confirmation workflow (auto-accept)

### Pricing Integration
- **Use** existing SQLite pricing database
- **Simplify** cost calculation (remove labor/markup complexity)
- **Focus** on component counts and basic material costs
- **Add** basic totaling for project estimates

### Performance Targets
- **Model Loading**: <10 seconds (one-time per session)
- **PDF Processing**: <5 seconds per page
- **AI Detection**: <3 seconds per floor plan
- **Total Processing**: <30 seconds for typical 2-page blueprint

---

## Success Metrics

### Functionality
- ✅ Processes 100% of valid electrical blueprint PDFs
- ✅ Maintains 70%+ component detection accuracy  
- ✅ Preserves floor plan detection capability
- ✅ Provides reasonable cost estimates

### Usability
- ✅ Single command operation: `python process_blueprint.py file.pdf`
- ✅ Clear, structured console output
- ✅ Helpful error messages
- ✅ Processing time under 30 seconds

### Reliability
- ✅ Graceful handling of all error conditions
- ✅ Consistent results across runs
- ✅ No crashes or hangs
- ✅ Memory usage under 8GB

---

## Next Steps After Implementation

### Phase 7: User Testing
1. Test with real electrical blueprints
2. Gather feedback on output format
3. Identify common error scenarios
4. Refine accuracy and performance

### Phase 8: Documentation
1. Create user guide with examples
2. Document troubleshooting steps
3. Add installation instructions
4. Provide sample blueprints for testing

### Phase 9: Enhancement Planning
1. Identify most requested features
2. Plan web interface development
3. Consider additional output formats
4. Evaluate model improvements

---

This plan maintains all the sophisticated AI detection capabilities while dramatically simplifying the user experience to a single command-line operation. The focus is on getting a reliable, accurate pipeline working locally before adding web interface complexity.
