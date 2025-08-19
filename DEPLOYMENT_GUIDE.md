# GP MLOps Platform - Comprehensive Deployment Guide

## Overview
This is a production-ready MLOps platform for real estate customer value inference and floorplan analysis, featuring FastAPI backend, machine learning pipelines, and comprehensive monitoring.

## Architecture
- **FastAPI**: Async REST API with auto-generated documentation
- **PostgreSQL**: Primary database for application data
- **Redis**: Caching and session management
- **MLflow**: ML model versioning and experiment tracking
- **Airbyte**: Data integration and ETL pipelines
- **Dify**: LLM workflow management
- **Prometheus + Grafana**: Monitoring and observability
- **KServe**: ML model serving (Kubernetes-based)

## Prerequisites
- Docker Desktop or Docker Engine 20.10+
- Docker Compose v2.0+
- 8GB+ RAM recommended
- 10GB+ free disk space

## Quick Start (5 minutes)

### 1. Clone and Navigate
```bash
git clone <repository-url>
cd gp-mlops-platform
```

### 2. Environment Setup
```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your credentials (optional for demo)
# SNOWFLAKE_ACCOUNT=your_account
# SNOWFLAKE_USER=your_user
# SNOWFLAKE_PASSWORD=your_password
# OPENAI_API_KEY=your_openai_key
```

### 3. Start Core Services
```bash
# Start essential services (API, database, cache)
docker-compose up -d api postgres redis

# Wait for services to initialize (30-60 seconds)
docker-compose logs -f api
```

### 4. Verify Deployment
```bash
# Check service health
curl http://localhost:2223/health/

# Access API documentation
open http://localhost:2223/docs
```

## Full Platform Deployment

### Start All Services
```bash
# Launch complete MLOps platform
docker-compose up -d

# Monitor startup progress
docker-compose logs -f
```

### Service Endpoints
| Service | URL | Purpose |
|---------|-----|---------|
| **Main API** | http://localhost:2223 | FastAPI application |
| **API Docs** | http://localhost:2223/docs | Interactive API documentation |
| **Health Check** | http://localhost:2223/health/ | Service health monitoring |
| **MLflow** | http://localhost:2226 | ML experiment tracking |
| **Prometheus** | http://localhost:2227 | Metrics collection |
| **Grafana** | http://localhost:2228 | Monitoring dashboards |
| **Dify** | http://localhost:2230 | LLM workflow management |
| **Airbyte** | http://localhost:2237 | Data integration |

## Testing the Platform

### 1. Basic Health Check
```bash
# Test API responsiveness
curl -X GET "http://localhost:2223/health/" \
  -H "accept: application/json"
```

### 2. Customer Prediction Example
```bash
# Make a prediction request
curl -X POST "http://localhost:2223/api/v1/customer/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "annual_income": 8000000,
    "profession": "engineer",
    "location_preference": "Shibuya",
    "family_size": 2
  }'
```

### 3. Floorplan Analysis
```bash
# Upload and analyze floorplan
curl -X POST "http://localhost:2223/api/v1/floorplan/detect" \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/floorplan.jpg"
  }'
```

### 4. Run Async Functionality Test
```bash
# Comprehensive async testing
python3 test_async.py
```

## Configuration Options

### Environment Variables (.env)
```bash
# Database Configuration
DATABASE_URL=postgresql://postgres:password@postgres:5432/ga_platform
REDIS_URL=redis://redis:6379

# External Service Integration
SNOWFLAKE_ACCOUNT=your_account
SNOWFLAKE_USER=your_user
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_DATABASE=your_database

# LLM Integration
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ADMIN_PASSWORD=admin
```

### Service Scaling
```bash
# Scale API instances
docker-compose up -d --scale api=3

# Scale worker processes
docker-compose up -d --scale celery-worker=2
```

## Production Deployment

### 1. Infrastructure Requirements
- **CPU**: 4+ cores recommended
- **Memory**: 16GB+ for full platform
- **Storage**: 50GB+ SSD recommended
- **Network**: Load balancer for multi-instance deployment

### 2. Security Configuration
```bash
# Generate secure secrets
openssl rand -hex 32  # For SECRET_KEY

# Update docker-compose.yml with production values
# - Remove default passwords
# - Use secrets management
# - Enable SSL/TLS
```

### 3. Database Setup
```bash
# Initialize database schema
docker-compose exec api alembic upgrade head

# Create initial admin user (if applicable)
docker-compose exec api python -m app.scripts.create_admin
```

### 4. Monitoring Setup
```bash
# Access Grafana
open http://localhost:2228
# Login: admin / admin (change on first login)

# Import pre-built dashboards
# - MLOps Pipeline Metrics
# - API Performance
# - System Resources
```

## Development Workflow

### 1. Local Development
```bash
# Start development services
docker-compose up -d postgres redis

# Run API locally for development
cd app
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Testing
```bash
# Run async functionality tests
python3 test_async.py

# Run API tests (if test suite exists)
pytest tests/

# Load testing
curl -X GET "http://localhost:2223/health/" &
curl -X GET "http://localhost:2223/health/" &
curl -X GET "http://localhost:2223/health/" &
```

### 3. Code Quality
```bash
# Format code (if tools are configured)
black app/
isort app/

# Type checking
mypy app/

# Security scanning
bandit -r app/
```

## Troubleshooting

### Common Issues

#### Service Won't Start
```bash
# Check logs
docker-compose logs api

# Restart specific service
docker-compose restart api

# Rebuild if needed
docker-compose build api
docker-compose up -d api
```

#### Database Connection Issues
```bash
# Verify PostgreSQL is running
docker-compose ps postgres

# Check database connectivity
docker-compose exec postgres psql -U postgres -d ga_platform -c "SELECT 1;"

# Reset database (CAUTION: Data loss)
docker-compose down
docker volume rm gp-mlops-platform_postgres_data
docker-compose up -d postgres
```

#### Port Conflicts
```bash
# Check port usage
netstat -tulpn | grep :2223

# Modify docker-compose.yml ports if needed
# Change "2223:8000" to "8080:8000" for example
```

#### Memory Issues
```bash
# Check Docker resource usage
docker stats

# Increase Docker Desktop memory allocation
# Docker Desktop > Settings > Resources > Memory
```

### Performance Optimization

#### API Performance
```bash
# Monitor API response times
curl -w "%{time_total}\n" -o /dev/null -s http://localhost:2223/health/

# Check concurrent request handling
ab -n 100 -c 10 http://localhost:2223/health/
```

#### Database Performance
```bash
# Monitor database connections
docker-compose exec postgres psql -U postgres -c "SELECT count(*) FROM pg_stat_activity;"

# Optimize queries (check slow query log)
docker-compose exec postgres psql -U postgres -c "SELECT query, calls, mean_time FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;"
```

## MLOps Pipeline Operations

### 1. Trigger Full Pipeline
```bash
curl -X POST "http://localhost:2223/api/v1/mlops/pipeline/run" \
  -H "Content-Type: application/json" \
  -d '{"pipeline_type": "daily"}'
```

### 2. Model Management
```bash
# Check model registry
curl "http://localhost:2223/api/v1/mlops/models"

# Deploy model to production
curl -X POST "http://localhost:2223/api/v1/mlops/models/CustomerValueModel/deploy"
```

### 3. Monitor Model Performance
```bash
# Get model metrics
curl "http://localhost:2223/api/v1/mlops/models/CustomerValueModel/metrics"

# Check for model drift
curl "http://localhost:2223/api/v1/mlops/models/CustomerValueModel/drift"
```

## Data Integration (Airbyte)

### 1. Access Airbyte UI
```bash
open http://localhost:2237
```

### 2. Setup Data Sources
- Configure Snowflake connection
- Setup data transformation pipelines
- Schedule regular data syncs

## LLM Workflows (Dify)

### 1. Access Dify Platform
```bash
open http://localhost:2230
```

### 2. Configure Workflows
- Customer interaction automation
- Property description generation
- Market analysis reports

## Backup and Recovery

### 1. Database Backup
```bash
# Create backup
docker-compose exec postgres pg_dump -U postgres ga_platform > backup.sql

# Restore backup
docker-compose exec -T postgres psql -U postgres ga_platform < backup.sql
```

### 2. Model Artifacts
```bash
# Backup MLflow artifacts
docker-compose exec mlflow tar -czf /backup/mlflow-artifacts.tar.gz /mlflow/artifacts

# Volume backup
docker run --rm -v gp-mlops-platform_mlflow_artifacts:/data -v $(pwd):/backup alpine tar czf /backup/mlflow-backup.tar.gz /data
```

## Monitoring and Alerting

### 1. Key Metrics to Monitor
- API response times and error rates
- Database connection pool usage
- ML model prediction accuracy
- System resource utilization

### 2. Alert Configuration
- Setup Prometheus alerting rules
- Configure Grafana notifications
- Monitor service health endpoints

## Security Best Practices

### 1. Network Security
```bash
# Use Docker networks for service isolation
# Configure firewall rules for production
# Enable HTTPS with proper certificates
```

### 2. Data Protection
```bash
# Encrypt sensitive environment variables
# Use Docker secrets for production passwords
# Regular security updates for base images
```

## Support and Maintenance

### 1. Regular Maintenance
- Weekly dependency updates
- Monthly security patches
- Quarterly capacity planning

### 2. Performance Monitoring
- Daily health checks
- Weekly performance reviews
- Monthly capacity assessments

---

## Quick Reference Commands

```bash
# Start platform
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f api

# Stop platform
docker-compose down

# Reset everything
docker-compose down -v
docker system prune -a

# Test async functionality
python3 test_async.py
```

For technical support or questions about this deployment, refer to the API documentation at http://localhost:2223/docs when the platform is running.