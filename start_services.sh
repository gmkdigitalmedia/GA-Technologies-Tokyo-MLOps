#!/bin/bash

# GP MLOps - Quick Services Startup (No Docker Build Required)
# This script starts only the pre-built services that don't require building

set -e

echo "🚀 GP MLOps Services Startup"
echo "===================================="
echo "Starting database and cache services only..."
echo

# Start only the services that use pre-built images
echo "📦 Starting PostgreSQL..."
docker-compose up -d postgres

echo "📦 Starting Redis..."
docker-compose up -d redis

echo "📦 Starting MLflow..."
docker-compose up -d mlflow 2>/dev/null || echo "MLflow service not configured"

echo "📦 Starting Prometheus..."
docker-compose up -d prometheus 2>/dev/null || echo "Prometheus service not configured"

echo "📦 Starting Grafana..."
docker-compose up -d grafana 2>/dev/null || echo "Grafana service not configured"

echo
echo "✅ Database services started!"
echo
echo "🌐 Available Services:"
echo "   • PostgreSQL: localhost:2224"
echo "   • Redis: localhost:2225"
echo
echo "📊 To start the main dashboard (recommended):"
echo "   cd dashboard && ./start_dashboard.sh"
echo
echo "🔧 The dashboard includes:"
echo "   • Working ML models"
echo "   • Tokyo real estate A/B testing"
echo "   • Interactive visualizations"
echo "   • Port 2222 (frontend) & 2233 (backend)"
echo
echo "Ready to demo! 🎯"