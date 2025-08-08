# GP MLOps MLOps Platform

A comprehensive MLOps/LLMOps platform for GP MLOps' real estate business, featuring customer value inference, floorplan layout detection, and conversion optimization.

## Architecture Overview

This platform integrates:
- **AWS Infrastructure**: ECS, SageMaker, S3, RDS, ElastiCache
- **Snowflake Data Warehouse**: Central data repository
- **MLOps Pipeline**: Model training, deployment, and monitoring
- **Real Estate ML Models**: Customer value inference and floorplan analysis
- **API Services**: RESTful APIs for all platform functionality

## Features

### Core ML Capabilities
- **Customer Value Inference**: Predict conversion probability and lifetime value
- **Floorplan Layout Detection**: AI-powered room detection and layout analysis
- **Property Recommendations**: Personalized property matching
- **Conversion API (CAPI)**: Optimize ad targeting and customer acquisition

### Platform Features
- **Batch Processing**: Handle large-scale predictions
- **Model Management**: Train, deploy, and monitor ML models
- **Data Pipeline**: Seamless Snowflake to AWS integration
- **Monitoring & Observability**: Comprehensive metrics and logging
- **Auto-scaling**: Cloud-native scalability

## Prerequisites

- AWS CLI configured with appropriate permissions
- Docker and Docker Compose
- Python 3.9+
- Snowflake account and credentials
- Git

## Quick Start

### 1. Clone and Setup
```bash
git clone <repository-url>
cd GA
cp .env.example .env
# Edit .env with your configurations
```

### 2. Local Development
```bash
# Start services
docker-compose up -d

# Install Python dependencies
pip install -r requirements.txt

# Run the API
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. AWS Deployment
```bash
# Configure AWS credentials
aws configure

# Set Snowflake credentials
export SNOWFLAKE_ACCOUNT="your_account"
export SNOWFLAKE_USER="your_user"  
export SNOWFLAKE_PASSWORD="your_password"

# Deploy infrastructure and application
./deploy.sh dev
```

## API Endpoints

### Health & Status
- `GET /health` - Basic health check
- `GET /health/detailed` - Detailed service status
- `GET /metrics` - Prometheus metrics

### Customer Analytics
- `POST /api/v1/customer/` - Create customer profile
- `GET /api/v1/customer/{customer_id}` - Get customer details
- `GET /api/v1/customer/{customer_id}/predict` - Generate predictions
- `POST /api/v1/customer/{customer_id}/interactions` - Log interactions

### Floorplan Analysis
- `POST /api/v1/floorplan/upload` - Upload floorplan image
- `POST /api/v1/floorplan/{property_id}/analyze` - Analyze floorplan
- `GET /api/v1/floorplan/{property_id}/analysis` - Get analysis results

### Batch Operations
- `POST /api/v1/predictions/batch` - Batch predictions
- `GET /api/v1/predictions/batch/{job_id}` - Check batch status
- `POST /api/v1/predictions/train-model` - Train/retrain models

## Data Architecture

### Snowflake Integration
- **Customer Data**: Demographics, preferences, interaction history  
- **Property Data**: Listings, features, pricing, availability
- **Analytics**: Conversion events, model predictions, performance metrics

### AWS Services
- **S3**: Data lake for raw/processed data and model artifacts
- **SageMaker**: Model training, deployment, and inference
- **RDS PostgreSQL**: Application database
- **ElastiCache Redis**: Caching and session management
- **ECS Fargate**: Containerized application hosting

## ML Models

### Customer Value Inference
- **Conversion Model**: Gradient Boosting Classifier
- **Lifetime Value Model**: Random Forest Regressor
- **Features**: Demographics, behavior, interaction patterns
- **Training**: Automated retraining on schedule or performance degradation

### Floorplan Detection
- **Architecture**: Custom CNN for room detection and layout analysis
- **Capabilities**: Room type identification, accessibility scoring, family-friendliness
- **Output**: Structured layout data, flow analysis, recommendations

## Monitoring & Operations

### Metrics & Observability
- **Application Metrics**: Request rates, response times, error rates
- **Model Metrics**: Accuracy, precision, recall, feature importance  
- **Infrastructure Metrics**: CPU, memory, disk usage
- **Business Metrics**: Conversion rates, prediction accuracy

### Alerting
- Model performance degradation
- Infrastructure issues
- Data quality problems
- SLA violations

## Configuration

### Environment Variables
Key configurations in `.env`:
```bash
# AWS
AWS_REGION=us-east-1
S3_BUCKET_NAME=gp-mlops-mlops-dev
SAGEMAKER_EXECUTION_ROLE=arn:aws:iam::account:role/SageMakerRole

# Snowflake
SNOWFLAKE_ACCOUNT=account.region
SNOWFLAKE_DATABASE=GP_MLOPS
SNOWFLAKE_WAREHOUSE=COMPUTE_WH

# Application
DATABASE_URL=postgresql://user:pass@localhost/ga_platform
REDIS_URL=redis://localhost:6379
```

## Testing

```bash
# Run unit tests
pytest tests/

# Run integration tests
pytest tests/integration/

# Test API endpoints
pytest tests/api/

# Load testing
locust -f tests/load_test.py --host=http://localhost:8000
```

## Development

### Adding New Features
1. Create feature branch
2. Implement changes with tests
3. Update documentation  
4. Submit pull request

### Model Development
1. Develop in Jupyter notebooks
2. Export to production code
3. Add to ML pipeline
4. Deploy via SageMaker

### API Development
1. Add endpoint to appropriate router
2. Add request/response models
3. Implement business logic
4. Add monitoring and logging

## 🔒 Security

- **Authentication**: JWT-based API authentication
- **Authorization**: Role-based access control
- **Data Encryption**: At rest and in transit
- **Secret Management**: AWS Secrets Manager
- **Network Security**: VPC, security groups, private subnets

## 📖 Documentation

- **API Docs**: Available at `/docs` (Swagger UI)
- **Model Documentation**: In `docs/models/`
- **Infrastructure**: CloudFormation templates in `infrastructure/`
- **Runbooks**: Operational procedures in `docs/operations/`

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is proprietary software owned by GP MLOps.

## 🆘 Support

For support and questions:
- Create issue in repository
- Contact DevOps team
- Check monitoring dashboards
- Review application logs

---

## Architecture Diagram

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client Apps   │    │   Load Balancer │    │      API        │
│                 │────│                 │────│   (ECS Fargate) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                               ┌─────────────────┐
                                               │   Redis Cache   │
                                               └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Snowflake DW  │    │      AWS S3     │    │  PostgreSQL DB  │
│                 │    │   Data Lake     │    │   App Database  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                       ┌─────────────────┐    ┌─────────────────┐
                       │   SageMaker     │    │   Monitoring    │
                       │   ML Models     │    │  (Prometheus)   │
                       └─────────────────┘    └─────────────────┘
```




  Architecture Overview:
  ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
  │   Tokyo Dashboard│    │   FastAPI Backend │    │   KServe Cluster│
  │   (Port 2222)   │◄──►│   (Port 8888)    │◄──►│   ML Models     │
  └─────────────────┘    └──────────────────┘    └─────────────────┘
           │                       │                       │
           │                       ▼                       ▼
           │              ┌──────────────────┐    ┌─────────────────┐
           │              │   Snowflake DW   │    │   AWS Services  │
           └──────────────┤   Data Storage   │    │   S3, ECS, etc  │
                          └──────────────────┘    └─────────────────┘
Built with ❤️ for GP MLOps' real estate platform.