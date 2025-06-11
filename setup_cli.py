#!/usr/bin/env python3
"""
Blueprint Processor CLI - Setup Script
Downloads SmolVLM model and installs dependencies for local processing.

This script:
1. Installs required Python packages
2. Downloads and caches SmolVLM model (~5GB, one-time)
3. Verifies model access and CPU compatibility
4. Tests basic functionality

Usage:
    python setup_cli.py
"""

import subprocess
import sys
import time
from pathlib import Path


def print_banner():
    """Print setup banner."""
    print("=" * 60)
    print("🔧 Blueprint Processor CLI - Setup")
    print("=" * 60)
    print("Setting up local blueprint processing with SmolVLM AI...")
    print()


def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8+ required")
        print(f"   Current version: {sys.version}")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    print()


def install_dependencies():
    """Install required Python packages."""
    print("📦 Installing Python dependencies...")
    
    requirements_file = Path("requirements_cli.txt")
    if not requirements_file.exists():
        print(f"❌ Error: {requirements_file} not found")
        sys.exit(1)
    
    try:
        # Install packages
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print("❌ Error installing dependencies:")
            print(result.stderr)
            sys.exit(1)
        
        print("✅ Dependencies installed successfully")
        print()
        
    except Exception as e:
        print(f"❌ Error during installation: {e}")
        sys.exit(1)


def download_smolvlm_model():
    """Download and cache SmolVLM model."""
    print("🤖 Downloading SmolVLM model (~5GB, one-time download)...")
    print("   This may take several minutes depending on your internet speed...")
    print()
    
    try:
        # Import and initialize the model to trigger download
        from transformers import AutoProcessor, AutoModelForVision2Seq
        import torch
        
        model_name = 'HuggingFaceTB/SmolVLM-256M-Instruct'
        
        print(f"📥 Downloading processor for {model_name}...")
        processor = AutoProcessor.from_pretrained(model_name)
        
        print(f"📥 Downloading model for {model_name}...")
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=False
        )
        
        print("✅ SmolVLM model downloaded and cached successfully")
        print(f"   Model location: {model.config._name_or_path}")
        print()
        
        return processor, model
        
    except Exception as e:
        print(f"❌ Error downloading SmolVLM model: {e}")
        print("   This might be due to:")
        print("   - Slow internet connection")
        print("   - Insufficient disk space (~5GB required)")
        print("   - Hugging Face server issues")
        sys.exit(1)


def verify_core_imports():
    """Verify that core modules can be imported."""
    print("🔍 Verifying core module imports...")
    
    try:
        # Test core imports (simplified version)
        from core.processor_simple import process_blueprint_multipage
        from core.detector_smolvlm_improved import ComponentDetectorSmolVLMImproved
        
        print("✅ Core modules imported successfully")
        
    except ImportError as e:
        print(f"❌ Error importing core modules: {e}")
        print("   Make sure you're running this script from the blueprint-processor directory")
        sys.exit(1)


def test_basic_functionality(processor, model):
    """Test basic functionality with the model."""
    print("🧪 Testing basic functionality...")
    
    try:
        # Create a simple test image
        from PIL import Image
        import torch
        
        # Create a small test image
        test_image = Image.new('RGB', (200, 200), color='white')
        
        # Test the model with a simple prompt
        prompt = "User: <image>What do you see?"
        
        inputs = processor(text=prompt, images=test_image, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,
                do_sample=False
            )
        
        response = processor.decode(outputs[0], skip_special_tokens=True)
        
        print("✅ SmolVLM model test successful")
        print(f"   Model response length: {len(response)} characters")
        print()
        
    except Exception as e:
        print(f"⚠️  Warning: Model test failed: {e}")
        print("   The model downloaded but may have issues.")
        print("   You can still try running the blueprint processor.")
        print()


def create_test_command():
    """Show user how to test the installation."""
    print("🚀 Setup Complete!")
    print()
    print("Next steps:")
    print("1. Place an electrical blueprint PDF in this directory")
    print("2. Run the processor:")
    print()
    print("   python process_blueprint.py your_blueprint.pdf")
    print()
    print("For verbose output:")
    print("   python process_blueprint.py your_blueprint.pdf --verbose")
    print()
    print("For help:")
    print("   python process_blueprint.py --help")
    print()


def main():
    """Main setup function."""
    try:
        print_banner()
        check_python_version()
        install_dependencies()
        verify_core_imports()
        processor, model = download_smolvlm_model()
        test_basic_functionality(processor, model)
        create_test_command()
        
    except KeyboardInterrupt:
        print("\n❌ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during setup: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
