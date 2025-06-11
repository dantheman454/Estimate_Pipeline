#!/bin/bash
# Blueprint Processor CLI - Environment Activation
echo "🔧 Activating Blueprint Processor CLI environment..."
source venv_cli/bin/activate
echo "✅ Environment activated!"
echo ""
echo "Usage:"
echo "  python process_blueprint.py your_blueprint.pdf"
echo "  python process_blueprint.py your_blueprint.pdf --verbose"
echo "  python process_blueprint.py your_blueprint.pdf --single-plan"
echo "  python process_blueprint.py --help"
echo ""
