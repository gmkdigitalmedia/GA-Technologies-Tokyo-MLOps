#!/bin/bash

# GP MLOps - Real Technical Demo
# No fake numbers, just working ML models

set -e

echo "🏢 GP MLOps MLOps Platform - Technical Demo"
echo "================================================================"
echo "Real ML models | Real predictions | Real A/B testing"
echo "No fake business metrics - just working technology"
echo "================================================================"
echo

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "FAIL Python 3 not found. Please install Python 3.8+"
    exit 1
fi

# Install requirements
echo "PKG Installing required packages..."
pip3 install numpy pandas scikit-learn opencv-python pillow torch torchvision --quiet

echo "PASS Dependencies installed"
echo

# Run the demo
echo "LAUNCH Running Technical Demo..."
echo

python3 real_demo.py

echo
echo "PASS Demo completed successfully!"
echo
echo "What you just saw:"
echo "• Real ML model training with actual accuracy metrics"
echo "• Computer vision processing of floorplan images"
echo "• A/B testing framework with statistical analysis"  
echo "• Database operations and health monitoring"
echo
echo "This is production-ready MLOps technology."