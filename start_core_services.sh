#!/bin/bash

# GP MLOps - Core Services Only (Skip Airbyte)
# Starts the essential MLOps services without Airbyte to avoid health check issues

set -e

echo "LAUNCH GP MLOps Core MLOps Services"
echo "======================================"
echo "Starting essential services (excluding Airbyte)..."
echo

# Fix dependencies first
echo "FIX Applying dependency fixes..."
./fix_dependencies.sh

echo "PKG Starting core services..."

# Core database services
echo "   - Starting PostgreSQL..."
docker-compose up -d postgres
sleep 5

echo "   - Starting Redis..."
docker-compose up -d redis
sleep 3

# MLOps services  
echo "   - Starting MLflow..."
docker-compose up -d mlflow
sleep 5

# Monitoring services
echo "   - Starting Prometheus..."
docker-compose up -d prometheus
sleep 3

echo "   - Starting Grafana..."
docker-compose up -d grafana
sleep 3

# Dify services (without dependencies on Airbyte)
echo "   - Starting Dify database..."
docker-compose up -d dify-db
sleep 5

echo "   - Starting Dify Redis..."
docker-compose up -d dify-redis
sleep 3

echo "   - Starting Dify API..."
docker-compose up -d dify-api
sleep 5

echo "   - Starting Dify Web..."
docker-compose up -d dify-web

# Try to start the main API (may fail due to build issues, but that's OK)
echo "   - Attempting to start main API..."
if docker-compose up -d api; then
    echo "   PASS Main API started successfully"
else
    echo "   ⚠️ Main API build failed (expected - use dashboard instead)"
fi

echo
echo "PASS Core MLOps services started!"
echo
echo "SEARCH Service Status:"
docker-compose ps

echo
echo "🌐 Available Services:"
echo "   • PostgreSQL: localhost:2224"
echo "   • Redis: localhost:2225" 
echo "   • MLflow: localhost:2226"
echo "   • Prometheus: localhost:2227"
echo "   • Grafana: localhost:2228 (admin/admin)"
echo "   • Dify API: localhost:2229"
echo "   • Dify Web: localhost:2230"
echo
echo "⚠️ Skipped Services (due to health check issues):"
echo "   • Airbyte (can be started separately)"
echo "   • Main API (if build failed)"
echo
echo "TARGET For working demo, run:"
echo "   cd dashboard && ./start_dashboard.sh"
echo
echo "INFO To check service logs:"
echo "   docker-compose logs -f [service_name]"