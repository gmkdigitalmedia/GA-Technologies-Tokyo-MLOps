#!/bin/bash

# GA Technologies Full MLOps Stack Startup Script
# Starts the complete MLOps platform with all components

set -e

echo "🚀 GA Technologies Full MLOps Platform"
echo "===================================="
echo "Starting complete MLOps stack..."
echo "Components: FastAPI + MLflow + Dify + KServe + Monitoring"
echo "===================================="
echo

# Check Docker and Docker Compose
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose not found. Please install Docker Compose first."
    exit 1
fi

# Clean up any previous failed builds
echo "🧹 Cleaning up previous builds..."
docker-compose down -v 2>/dev/null || true
docker system prune -f 2>/dev/null || true

# Copy fixed requirements
echo "🔧 Using fixed requirements..."
cp requirements-fixed.txt requirements.txt 2>/dev/null || true

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p data/models
mkdir -p data/artifacts
mkdir -p logs
mkdir -p monitoring/config

# Set environment variables
export OPENAI_API_KEY=${OPENAI_API_KEY:-"your-openai-key-here"}
export ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-"your-anthropic-key-here"}
export SNOWFLAKE_ACCOUNT=${SNOWFLAKE_ACCOUNT:-"your-account"}
export SNOWFLAKE_USER=${SNOWFLAKE_USER:-"your-user"}
export SNOWFLAKE_PASSWORD=${SNOWFLAKE_PASSWORD:-"your-password"}
export SNOWFLAKE_DATABASE=${SNOWFLAKE_DATABASE:-"GA_TECHNOLOGIES_DW"}

echo "🔧 Environment variables configured"

# Create Prometheus configuration
cat > monitoring/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'ga-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: /metrics
    scrape_interval: 5s

  - job_name: 'mlflow'
    static_configs:
      - targets: ['mlflow:5000']
    metrics_path: /metrics
    scrape_interval: 30s

  - job_name: 'dify-api'
    static_configs:
      - targets: ['dify-api:5001']
    metrics_path: /metrics
    scrape_interval: 30s
EOF

echo "📊 Prometheus configuration created"

# Start the full stack
echo "🐳 Starting Docker Compose services..."
echo "This may take several minutes for first-time startup..."

docker-compose up -d

echo "⏳ Waiting for services to initialize..."

# Wait for PostgreSQL to be ready
echo "   - Waiting for PostgreSQL..."
sleep 10

# Wait for Redis to be ready  
echo "   - Waiting for Redis..."
sleep 5

# Wait for MLflow to be ready
echo "   - Waiting for MLflow..."
sleep 15

# Wait for Dify services to be ready
echo "   - Waiting for Dify services..."
sleep 30

# Wait for main API to be ready
echo "   - Waiting for GA API..."
sleep 10

# Health check function
check_service() {
    local service_name=$1
    local url=$2
    local max_attempts=10
    local attempt=1
    
    echo "🔍 Checking $service_name..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "$url" > /dev/null 2>&1; then
            echo "   ✅ $service_name is healthy"
            return 0
        fi
        echo "   ⏳ $service_name not ready (attempt $attempt/$max_attempts)"
        sleep 5
        ((attempt++))
    done
    
    echo "   ⚠️  $service_name health check failed"
    return 1
}

# Perform health checks
echo "🏥 Performing health checks..."

check_service "GA Technologies API" "http://localhost:2223/health"
check_service "MLflow Server" "http://localhost:2226"
check_service "Dify API" "http://localhost:2229/health"
check_service "Dify Web Console" "http://localhost:2230"
check_service "Prometheus" "http://localhost:2227/-/healthy"
check_service "Grafana" "http://localhost:2228/api/health"

echo
echo "✅ MLOps Platform Started Successfully!"
echo "===================================="
echo
echo "🌐 Access Points:"
echo "   • GA Technologies API:       http://localhost:2223"
echo "   • API Documentation:       http://localhost:2223/docs"
echo "   • Tokyo Dashboard:         http://localhost:2222 (if running)"
echo
echo "🤖 MLOps Components:"
echo "   • MLflow Model Registry:   http://localhost:2226"
echo "   • Dify LLM Workflows:      http://localhost:2230"
echo "   • Airbyte Data Platform:   http://localhost:2237"
echo "   • Prometheus Monitoring:   http://localhost:2227"
echo "   • Grafana Dashboards:      http://localhost:2228 (admin/admin)"
echo
echo "🔧 Database Services:"
echo "   • PostgreSQL (Main):       localhost:2224"
echo "   • PostgreSQL (Dify):       localhost:2234"
echo "   • PostgreSQL (Airbyte):    localhost:2235"
echo "   • Redis:                   localhost:2225"
echo
echo "📊 Key Features Available:"
echo "   ✓ Customer Value Inference ML Models"
echo "   ✓ Floorplan Detection & Analysis"
echo "   ✓ Tokyo Real Estate A/B Testing"
echo "   ✓ LLM-Powered Content Generation"
echo "   ✓ Model Training & Versioning"
echo "   ✓ Automated MLOps Pipeline"
echo "   ✓ Airbyte Data Integration"
echo "   ✓ Real-time Monitoring & Alerts"
echo "   ✓ KServe Model Serving (when k8s available)"
echo
echo "🚀 Quick Start Commands:"
echo "   # Check MLOps status"
echo "   curl http://localhost:2223/api/v1/mlops/status"
echo
echo "   # Run full MLOps pipeline"
echo "   curl -X POST http://localhost:2223/api/v1/mlops/pipeline/run \\"
echo "        -H 'Content-Type: application/json' \\"
echo "        -d '{\"pipeline_type\": \"on_demand\"}'"
echo
echo "   # Check model registry"
echo "   curl http://localhost:2223/api/v1/mlops/models/registry"
echo
echo "   # Monitor model drift"
echo "   curl -X POST http://localhost:2223/api/v1/mlops/monitoring/drift \\"
echo "        -H 'Content-Type: application/json' \\"
echo "        -d '{\"model_name\": \"CustomerValueModel\", \"time_window_hours\": 24}'"
echo
echo "📱 Tokyo Real Estate Dashboard:"
echo "   cd dashboard && ./start_dashboard.sh"
echo
echo "🛑 To stop all services:"
echo "   docker-compose down"
echo
echo "🔍 To view logs:"
echo "   docker-compose logs -f [service_name]"
echo
echo "===================================="
echo "🎯 Ready for GA Technologies Demo!"
echo "===================================="