#!/bin/bash

# GA Technologies - Real Technical Demo
# No fake numbers, just working ML models

set -e

echo "🏢 GA Technologies MLOps Platform - Technical Demo"
echo "================================================================"
echo "Real ML models | Real predictions | Real A/B testing"
echo "No fake business metrics - just working technology"
echo "================================================================"
echo

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8+"
    exit 1
fi

# Install requirements
echo "📦 Installing required packages..."
pip3 install numpy pandas scikit-learn opencv-python pillow torch torchvision --quiet

echo "✅ Dependencies installed"
echo

# Run the demo
echo "🚀 Running Technical Demo..."
echo

python3 real_demo.py

echo
echo "✅ Demo completed successfully!"
echo
echo "What you just saw:"
echo "• Real ML model training with actual accuracy metrics"
echo "• Computer vision processing of floorplan images"
echo "• A/B testing framework with statistical analysis"  
echo "• Database operations and health monitoring"
echo
echo "This is production-ready MLOps technology."