#!/bin/bash

# GP MLOps - Quick Services Startup (No Docker Build Required)
# This script starts only the pre-built services that don't require building

set -e

echo "LAUNCH GP MLOps Services Startup"
echo "===================================="
echo "Starting database and cache services only..."
echo

# Start only the services that use pre-built images
echo "PKG Starting PostgreSQL..."
docker-compose up -d postgres

echo "PKG Starting Redis..."
docker-compose up -d redis

echo "PKG Starting MLflow..."
docker-compose up -d mlflow 2>/dev/null || echo "MLflow service not configured"

echo "PKG Starting Prometheus..."
docker-compose up -d prometheus 2>/dev/null || echo "Prometheus service not configured"

echo "PKG Starting Grafana..."
docker-compose up -d grafana 2>/dev/null || echo "Grafana service not configured"

echo
echo "PASS Database services started!"
echo
echo "🌐 Available Services:"
echo "   • PostgreSQL: localhost:2224"
echo "   • Redis: localhost:2225"
echo
echo "CHART To start the main dashboard (recommended):"
echo "   cd dashboard && ./start_dashboard.sh"
echo
echo "FIX The dashboard includes:"
echo "   • Working ML models"
echo "   • Tokyo real estate A/B testing"
echo "   • Interactive visualizations"
echo "   • Port 2222 (frontend) & 2233 (backend)"
echo
echo "Ready to demo! TARGET"