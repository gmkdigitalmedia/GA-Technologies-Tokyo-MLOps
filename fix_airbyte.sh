#!/bin/bash

# Fix Airbyte Database Health Check Issues

echo "FIX Fixing Airbyte database health check..."
echo "========================================="

# Stop any existing Airbyte services
echo "🛑 Stopping existing Airbyte services..."
docker-compose stop airbyte-db airbyte-bootloader airbyte-worker airbyte-server airbyte-webapp airbyte-temporal 2>/dev/null || true

# Remove any unhealthy containers
echo "🧹 Cleaning up unhealthy containers..."
docker-compose rm -f airbyte-db airbyte-bootloader airbyte-worker airbyte-server airbyte-webapp airbyte-temporal 2>/dev/null || true

# Clear any volumes that might be corrupted
echo "🗑️ Clearing Airbyte volumes..."
docker volume rm ga-airbyte_db ga-airbyte_workspace ga-airbyte_local 2>/dev/null || true

# Start Airbyte database with extended timeout
echo "PKG Starting Airbyte database (with extended timeout)..."
docker-compose up -d airbyte-db

# Wait for health check
echo "⏳ Waiting for Airbyte database to become healthy..."
for i in {1..20}; do
    if docker-compose ps airbyte-db | grep -q "healthy"; then
        echo "PASS Airbyte database is healthy!"
        break
    elif [ $i -eq 20 ]; then
        echo "FAIL Airbyte database failed to become healthy after 10 minutes"
        echo "SEARCH Checking logs..."
        docker-compose logs airbyte-db | tail -20
        exit 1
    else
        echo "   Attempt $i/20: Still waiting... (${i}0s elapsed)"
        sleep 30
    fi
done

# Start the rest of Airbyte services
echo "LAUNCH Starting Airbyte bootloader..."
docker-compose up -d airbyte-bootloader

echo "⏳ Waiting for bootloader to complete..."
sleep 30

echo "LAUNCH Starting remaining Airbyte services..."
docker-compose up -d airbyte-worker airbyte-server airbyte-webapp airbyte-temporal

echo
echo "PASS Airbyte services started!"
echo
echo "🌐 Airbyte Access:"
echo "   • Web UI: http://localhost:2237"
echo "   • API: http://localhost:2236"
echo "   • Database: localhost:2235"
echo
echo "SEARCH Service Status:"
docker-compose ps | grep airbyte
echo