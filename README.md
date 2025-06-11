# Estimate_Pipeline

> **Simple command-line electrical blueprint analysis using AI**

Process electrical blueprints locally without any web hosting requirements. Uses SmolVLM AI (76.9% accuracy) to detect electrical components and generate cost estimates.

## 🚀 Quick Start 

### 1. Setup (One-time)
```bash
# Clone or navigate to the blueprint-processor directory
cd blueprint-processor

# Run the setup script (downloads ~5GB SmolVLM model)
python setup_cli.py
```

### 2. Process a Blueprint
```bash
# Basic usage
python process_blueprint.py your_blueprint.pdf

# Verbose output with detailed processing info
python process_blueprint.py your_blueprint.pdf --verbose

# Save results to file
python process_blueprint.py your_blueprint.pdf --output results.txt
```

## 📋 Features

### 🤖 AI-Powered Detection
- **SmolVLM Model**: 76.9% component detection accuracy
- **Floor Plan Detection**: Automatically detects and processes multiple floor plans
- **Smart Fallback**: Provides reasonable estimates when AI detection fails
- **Component Types**: Outlets, switches, lights, fans, smoke detectors, panels

### 🔧 Processing Pipeline
- **PDF Processing**: Converts multi-page PDFs to optimized images
- **Image Enhancement**: Advanced preprocessing for better AI analysis
- **Floor-by-Floor Analysis**: Separate component detection for each floor
- **Cost Estimation**: Material and labor cost calculations

### 💻 Local Processing
- **No Internet Required**: Runs completely offline after setup
- **CPU Optimized**: Works without GPU requirements
- **Privacy Focused**: All processing happens on your machine
- **No Authentication**: No API keys or accounts needed

## 🎯 Example Output

```
==================================================
🔧 BLUEPRINT PROCESSING RESULTS
==================================================

📄 File: residential_blueprint.pdf
🕐 Processed: 2025-06-11 14:30:25
⏱️  Processing Time: 12.3 seconds
🏗️  Floors Analyzed: 2
🔍 Detection Method: SmolVLM Improved

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
   • Estimated Labor: $925.00 (2.8 hours)
   • Project Total: $2,172.50

✅ Processing Complete!
```

## 📖 Usage Options

### Basic Commands
```bash
# Process single blueprint
python process_blueprint.py blueprint.pdf

# Skip floor plan detection (treat as single plan)
python process_blueprint.py blueprint.pdf --single-plan

# Use custom pricing database
python process_blueprint.py blueprint.pdf --pricing-db custom_prices.db
```

### Output Options
```bash
# Save detailed results to file
python process_blueprint.py blueprint.pdf --output detailed_results.txt

# Verbose processing information
python process_blueprint.py blueprint.pdf --verbose
```

### Help
```bash
# Show all available options
python process_blueprint.py --help
```

## 🛠️ Requirements

### System Requirements
- **Python**: 3.8+ (tested with Python 3.13)
- **RAM**: 8GB+ recommended for SmolVLM model
- **Disk Space**: 6GB+ for model cache
- **OS**: macOS, Linux, Windows

### Python Dependencies
- `pdf2image` - PDF to image conversion
- `pillow` - Image processing
- `transformers` - SmolVLM model
- `torch` - AI inference
- `opencv-python` - Advanced image processing
- `pytesseract` - OCR for floor plan detection

## 🔧 Installation Details

### Automatic Setup
The `setup_cli.py` script handles everything:
1. Installs Python dependencies
2. Downloads SmolVLM model (~5GB)
3. Verifies installation
4. Tests basic functionality

### Manual Installation
If you prefer manual setup:
```bash
# Install dependencies
pip install -r requirements_cli.txt

# Test the processor
python process_blueprint.py --help
```

## 📁 File Structure

```
blueprint-processor/
├── process_blueprint.py          # Main CLI script
├── setup_cli.py                  # Setup and installation
├── requirements_cli.txt          # Python dependencies
├── README_CLI.md                 # This file
├── core/                         # Core processing modules
│   ├── processor.py              # PDF and floor plan processing
│   ├── detector_smolvlm_improved.py  # AI component detection
│   ├── floor_plan_detector.py    # Floor plan extraction
│   ├── advanced_preprocessor.py  # Image enhancement
│   └── pricing.py                # Cost calculation
├── pricing.db                    # Component pricing database
└── logs/                         # Processing logs
```

## 🎯 Supported File Types

- **Input**: PDF blueprints (electrical floor plans)
- **Output**: Structured text to console or file
- **Future**: JSON, CSV export options planned

## 🔍 Troubleshooting

### Common Issues

**"PDF file not found"**
- Check file path and ensure the file exists
- Make sure the file has a `.pdf` extension

**"SmolVLM model failed to load"**
- Ensure you have 8GB+ available RAM
- Re-run `setup_cli.py` to re-download the model
- Check internet connection during setup

**"No components detected"**
- Try `--verbose` flag to see detailed processing
- Ensure blueprint contains electrical symbols
- Check if PDF is readable (not a scanned image)

**"Pricing database not found"**
- Use `--pricing-db` to specify correct database path
- Default costs will be used if database is missing

### Performance Tips

**Speed up processing:**
- Use `--single-plan` to skip floor detection for simple blueprints
- Close other applications to free up RAM
- Use solid-state drive for model cache

**Improve accuracy:**
- Ensure blueprints are high-quality PDFs
- Use electrical blueprints with standard symbols
- Try processing individual pages if multi-page detection fails

## 🚀 Next Steps

### After Basic Setup
1. Test with sample blueprints
2. Customize pricing database for your region
3. Process your electrical blueprints
4. Save results for project documentation

### Future Enhancements
- Web interface for easier use
- Batch processing for multiple files
- Export to Excel/CSV formats
- Custom component libraries
- Integration with estimating software

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Run with `--verbose` for detailed error information
3. Ensure all requirements are met
4. Try re-running `setup_cli.py`

---

**Note**: This CLI version focuses on core functionality without web hosting complexity. A web interface version is planned for future development.
