#!/bin/bash

# GP MLOps - Step by Step Service Startup
# Starts services one by one to identify any build issues

set -e

echo "🚀 GP MLOps Step-by-Step Startup"
echo "========================================"
echo "Starting services individually..."
echo

# Fix dependencies first
./fix_dependencies.sh

echo "📦 Step 1: Starting PostgreSQL..."
docker-compose up -d postgres
sleep 5

echo "📦 Step 2: Starting Redis..."
docker-compose up -d redis
sleep 3

echo "📦 Step 3: Building and starting API..."
if docker-compose build api; then
    echo "✅ API build successful!"
    docker-compose up -d api
    sleep 10
else
    echo "❌ API build failed. Continuing with other services..."
fi

echo "📦 Step 4: Starting MLflow (if available)..."
docker-compose up -d mlflow 2>/dev/null && echo "✅ MLflow started" || echo "⚠️ MLflow not available"

echo "📦 Step 5: Starting monitoring services..."
docker-compose up -d prometheus 2>/dev/null && echo "✅ Prometheus started" || echo "⚠️ Prometheus not available"
docker-compose up -d grafana 2>/dev/null && echo "✅ Grafana started" || echo "⚠️ Grafana not available"

echo
echo "✅ Step-by-step startup complete!"
echo
echo "🔍 Service Status:"
docker-compose ps

echo
echo "🌐 Available Services:"
echo "   • PostgreSQL: localhost:2224"
echo "   • Redis: localhost:2225"
echo "   • API (if built): localhost:2223"
echo "   • MLflow: localhost:2226"
echo "   • Prometheus: localhost:2227"
echo "   • Grafana: localhost:2228"
echo
echo "📱 For working demo, run:"
echo "   cd dashboard && ./start_dashboard.sh"
echo