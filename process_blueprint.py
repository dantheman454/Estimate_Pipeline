#!/usr/bin/env python3
"""
Blueprint Processor - Command Line Interface
Simple local processing of electrical blueprints without web hosting.

Usage:
    python process_blueprint.py blueprint.pdf
    python process_blueprint.py blueprint.pdf --verbose
    python process_blueprint.py blueprint.pdf --output results.txt
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Core imports
from core.processor_simple import process_blueprint_with_floor_plans, process_blueprint_multipage
from core.detector_smolvlm_improved import ComponentDetectorSmolVLMImproved
from core.pricing import PricingEngine


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process electrical blueprints locally using AI detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process_blueprint.py blueprint.pdf
  python process_blueprint.py blueprint.pdf --verbose
  python process_blueprint.py blueprint.pdf --output results.txt
  python process_blueprint.py blueprint.pdf --single-plan
        """
    )
    
    parser.add_argument(
        'pdf_file',
        type=str,
        help='Path to the electrical blueprint PDF file'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output with detailed processing information'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Save results to specified text file (default: console only)'
    )
    
    parser.add_argument(
        '--single-plan',
        action='store_true',
        help='Skip floor plan detection, process as single plan'
    )
    
    parser.add_argument(
        '--pricing-db',
        type=str,
        default='pricing.db',
        help='Path to custom pricing database (default: pricing.db)'
    )
    
    return parser.parse_args()


def validate_inputs(args):
    """Validate command line arguments."""
    # Check if PDF file exists
    pdf_path = Path(args.pdf_file)
    if not pdf_path.exists():
        print(f"❌ Error: PDF file not found: {pdf_path}")
        sys.exit(1)
    
    if not pdf_path.suffix.lower() == '.pdf':
        print(f"❌ Error: File must be a PDF: {pdf_path}")
        sys.exit(1)
    
    # Check if pricing database exists
    pricing_path = Path(args.pricing_db)
    if not pricing_path.exists():
        print(f"⚠️  Warning: Pricing database not found: {pricing_path}")
        print("   Will use default component prices.")
    
    return pdf_path.absolute(), pricing_path


def format_results(detection_result: Dict, pdf_path: Path, processing_time: float, verbose: bool = False) -> str:
    """Format detection results for console output."""
    
    # Header
    output = []
    output.append("=" * 50)
    output.append("🔧 BLUEPRINT PROCESSING RESULTS")
    output.append("=" * 50)
    output.append("")
    
    # File information
    output.append(f"📄 File: {pdf_path.name}")
    output.append(f"🕐 Processed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output.append(f"⏱️  Processing Time: {processing_time:.1f} seconds")
    
    # Analysis summary
    analysis_summary = detection_result.get('analysis_summary', {})
    total_floors = analysis_summary.get('total_floors_analyzed', 0)
    total_components = analysis_summary.get('total_components_found', 0)
    detection_method = analysis_summary.get('detection_method', 'SmolVLM Improved')
    
    output.append(f"🏗️  Floors Analyzed: {total_floors}")
    output.append(f"🔍 Detection Method: {detection_method}")
    output.append("")
    
    # Floor-by-floor breakdown
    output.append("📊 FLOOR-BY-FLOOR BREAKDOWN:")
    output.append("")
    
    floor_breakdown = detection_result.get('floor_breakdown', [])
    
    if not floor_breakdown:
        # Fallback to simple component display
        total_components_dict = detection_result.get('total_components', {})
        output.append("🏠 ELECTRICAL PLAN")
        for component, count in total_components_dict.items():
            if count > 0:
                display_name = component.replace('_', ' ').title()
                output.append(f"   • {display_name}: {count}")
        output.append(f"   Total Components: {sum(total_components_dict.values())}")
        output.append("")
    else:
        # Floor-by-floor breakdown
        for floor_data in floor_breakdown:
            floor_title = floor_data.get('floor_title', 'Unknown Floor')
            floor_components = floor_data.get('components', {})
            floor_total = sum(floor_components.values())
            
            output.append(f"🏠 {floor_title}")
            
            if floor_components:
                for component, count in floor_components.items():
                    if count > 0:
                        display_name = component.replace('_', ' ').title()
                        output.append(f"   • {display_name}: {count}")
                output.append(f"   Total Components: {floor_total}")
            else:
                output.append("   • No components detected")
            
            output.append("")
    
    # Cost estimate (simplified)
    output.append("💰 COST ESTIMATE SUMMARY:")
    
    try:
        from core.pricing import PricingEngine
        pricing = PricingEngine()
        
        # Calculate basic material costs
        total_cost = 0
        total_components_dict = detection_result.get('total_components', {})
        
        for component, count in total_components_dict.items():
            if count > 0:
                unit_price = pricing.get_price(component)
                component_cost = count * unit_price
                total_cost += component_cost
                
                if verbose:
                    display_name = component.replace('_', ' ').title()
                    output.append(f"   • {display_name}: {count} × ${unit_price:.2f} = ${component_cost:.2f}")
        
        if not verbose:
            output.append(f"   • Total Components: {sum(total_components_dict.values())}")
        
        output.append(f"   • Material Cost: ${total_cost:.2f}")
        
        # Simple labor estimate (1 hour per 10 components)
        labor_hours = max(1, sum(total_components_dict.values()) / 10)
        labor_cost = labor_hours * 75  # $75/hour
        output.append(f"   • Estimated Labor: ${labor_cost:.2f} ({labor_hours:.1f} hours)")
        
        project_total = total_cost + labor_cost
        output.append(f"   • Project Total: ${project_total:.2f}")
        
    except Exception as e:
        output.append(f"   • Cost calculation error: {str(e)}")
        output.append(f"   • Total Components: {sum(total_components_dict.values())}")
    
    output.append("")
    output.append("✅ Processing Complete!")
    output.append("")
    
    return "\n".join(output)


def save_results(output_text: str, output_file: str):
    """Save results to file."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output_text)
        print(f"💾 Results saved to: {output_file}")
    except Exception as e:
        print(f"❌ Error saving results: {e}")


def main():
    """Main CLI function."""
    try:
        # Parse and validate arguments
        args = parse_arguments()
        pdf_path, pricing_path = validate_inputs(args)
        
        print("🔧 Blueprint Processor CLI")
        print("=" * 30)
        print(f"📄 Processing: {pdf_path.name}")
        if args.verbose:
            print(f"🔍 Mode: Verbose")
        if args.single_plan:
            print(f"📋 Mode: Single Plan (skip floor detection)")
        print()
        
        # Initialize detector
        if args.verbose:
            print("🤖 Initializing SmolVLM detector...")
        
        start_time = time.time()
        detector = ComponentDetectorSmolVLMImproved()
        
        if args.verbose:
            init_time = time.time() - start_time
            print(f"✅ SmolVLM loaded in {init_time:.1f} seconds")
            print()
        
        # Process blueprint
        if args.verbose:
            print("📄 Processing PDF and detecting floor plans...")
        
        process_start = time.time()
        
        if args.single_plan:
            # Simple processing without floor plan detection
            images = process_blueprint_multipage(str(pdf_path))
            
            if args.verbose:
                print(f"📸 Converted {len(images)} pages to images")
                print("🔍 Running component detection...")
            
            components = detector.detect_components_multi_page(images)
            
            # Format as simple result
            detection_result = {
                'total_components': components,
                'floor_breakdown': [],
                'analysis_summary': {
                    'total_floors_analyzed': len(images),
                    'total_components_found': sum(components.values()),
                    'detection_method': 'SmolVLM Improved (Single Plan Mode)'
                }
            }
        else:
            # Full processing with floor plan detection
            floor_plan_data = process_blueprint_with_floor_plans(str(pdf_path))
            
            if args.verbose:
                total_floors = sum(len(page['floor_plans']) for page in floor_plan_data['pages'])
                print(f"🏠 Detected {total_floors} floor plans across {floor_plan_data['total_pages']} pages")
                print("🔍 Running AI component detection...")
            
            detection_result = detector.detect_components_floor_by_floor(floor_plan_data)
        
        processing_time = time.time() - process_start
        
        if args.verbose:
            print(f"✅ Detection completed in {processing_time:.1f} seconds")
            print()
        
        # Format and display results
        output_text = format_results(detection_result, pdf_path, processing_time, args.verbose)
        print(output_text)
        
        # Save to file if requested
        if args.output:
            save_results(output_text, args.output)
        
    except KeyboardInterrupt:
        print("\n❌ Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during processing: {str(e)}")
        if args.verbose:
            import traceback
            print("\nFull error traceback:")
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
