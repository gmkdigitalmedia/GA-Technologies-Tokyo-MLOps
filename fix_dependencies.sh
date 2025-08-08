#!/bin/bash

# Fix dependency conflicts in GP MLOps MLOps Platform

echo "🔧 Fixing dependency conflicts..."
echo "================================"
echo

# Backup original requirements
cp requirements.txt requirements.txt.backup 2>/dev/null || true

# Use the fixed requirements file (all conflicts resolved)
echo "✅ Using fixed requirements file to avoid all conflicts..."
cp requirements-fixed.txt requirements.txt

# Update Dockerfile to use fixed version
echo "✅ Using fixed Dockerfile..."
cp Dockerfile.fixed Dockerfile

# Clean up Docker cache
echo "🧹 Cleaning Docker cache..."
docker system prune -f 2>/dev/null || true

echo
echo "✅ Dependencies fixed!"
echo
echo "You can now run:"
echo "  1. ./start_services.sh     - Start database services"
echo "  2. cd dashboard && ./start_dashboard.sh - Start the main demo"
echo
echo "Or if you want to try Docker build again:"
echo "  docker-compose build api"
echo