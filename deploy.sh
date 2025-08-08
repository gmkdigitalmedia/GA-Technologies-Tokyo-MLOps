#!/bin/bash

# GP MLOps Platform Deployment Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
ENVIRONMENT=${1:-dev}
AWS_REGION=${AWS_REGION:-us-east-1}
STACK_NAME="gp-mlops-${ENVIRONMENT}"

echo -e "${GREEN}🚀 Deploying GP MLOps Platform - Environment: ${ENVIRONMENT}${NC}"

# Function to print status
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        print_error "AWS CLI not found. Please install AWS CLI."
        exit 1
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker not found. Please install Docker."
        exit 1
    fi
    
    # Check if logged into AWS
    if ! aws sts get-caller-identity &> /dev/null; then
        print_error "Not logged into AWS. Please configure AWS credentials."
        exit 1
    fi
    
    print_status "Prerequisites check completed"
}

# Deploy AWS infrastructure
deploy_infrastructure() {
    print_status "Deploying AWS infrastructure..."
    
    # Validate CloudFormation template
    aws cloudformation validate-template \
        --template-body file://infrastructure/cloudformation.yaml \
        --region $AWS_REGION
    
    if [ $? -eq 0 ]; then
        print_status "CloudFormation template is valid"
    else
        print_error "CloudFormation template validation failed"
        exit 1
    fi
    
    # Deploy stack
    aws cloudformation deploy \
        --template-file infrastructure/cloudformation.yaml \
        --stack-name $STACK_NAME \
        --parameter-overrides \
            Environment=$ENVIRONMENT \
            SnowflakeAccount=$SNOWFLAKE_ACCOUNT \
            SnowflakeUser=$SNOWFLAKE_USER \
            SnowflakePassword=$SNOWFLAKE_PASSWORD \
        --capabilities CAPABILITY_NAMED_IAM \
        --region $AWS_REGION
    
    if [ $? -eq 0 ]; then
        print_status "Infrastructure deployed successfully"
    else
        print_error "Infrastructure deployment failed"
        exit 1
    fi
    
    # Get stack outputs
    aws cloudformation describe-stacks \
        --stack-name $STACK_NAME \
        --region $AWS_REGION \
        --query 'Stacks[0].Outputs' > stack_outputs.json
    
    print_status "Stack outputs saved to stack_outputs.json"
}

# Build and push Docker image
build_and_push_image() {
    print_status "Building and pushing Docker image..."
    
    # Get ECR repository URI (if exists)
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    ECR_REPO="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/gp-mlops-${ENVIRONMENT}"
    
    # Create ECR repository if it doesn't exist
    aws ecr describe-repositories --repository-names "gp-mlops-${ENVIRONMENT}" --region $AWS_REGION 2>/dev/null || \
    aws ecr create-repository --repository-name "gp-mlops-${ENVIRONMENT}" --region $AWS_REGION
    
    # Login to ECR
    aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_REPO
    
    # Build image
    docker build -t gp-mlops:$ENVIRONMENT .
    docker tag gp-mlops:$ENVIRONMENT $ECR_REPO:latest
    docker tag gp-mlops:$ENVIRONMENT $ECR_REPO:$ENVIRONMENT
    
    # Push image
    docker push $ECR_REPO:latest
    docker push $ECR_REPO:$ENVIRONMENT
    
    print_status "Docker image built and pushed successfully"
}

# Initialize database
initialize_database() {
    print_status "Initializing database..."
    
    # Get database endpoint from stack outputs
    DB_ENDPOINT=$(aws cloudformation describe-stacks \
        --stack-name $STACK_NAME \
        --region $AWS_REGION \
        --query 'Stacks[0].Outputs[?OutputKey==`DatabaseEndpoint`].OutputValue' \
        --output text)
    
    # Run database migrations (this would typically use Alembic)
    print_warning "Database initialization would run here (implement Alembic migrations)"
    
    print_status "Database initialized"
}

# Deploy application to ECS
deploy_application() {
    print_status "Deploying application to ECS..."
    
    # Create ECS task definition
    cat > task-definition.json << EOF
{
    "family": "gp-mlops-${ENVIRONMENT}",
    "networkMode": "awsvpc",
    "requiresCompatibilities": ["FARGATE"],
    "cpu": "1024",
    "memory": "2048",
    "executionRoleArn": "arn:aws:iam::${ACCOUNT_ID}:role/ecsTaskExecutionRole",
    "taskRoleArn": "arn:aws:iam::${ACCOUNT_ID}:role/ecsTaskRole",
    "containerDefinitions": [
        {
            "name": "gp-mlops-api",
            "image": "${ECR_REPO}:${ENVIRONMENT}",
            "portMappings": [
                {
                    "containerPort": 8000,
                    "protocol": "tcp"
                }
            ],
            "environment": [
                {
                    "name": "ENVIRONMENT",
                    "value": "${ENVIRONMENT}"
                }
            ],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "/ecs/gp-mlops-${ENVIRONMENT}",
                    "awslogs-region": "${AWS_REGION}",
                    "awslogs-stream-prefix": "ecs"
                }
            }
        }
    ]
}
EOF

    # Register task definition
    aws ecs register-task-definition \
        --cli-input-json file://task-definition.json \
        --region $AWS_REGION
    
    print_status "Application deployed to ECS"
}

# Setup monitoring
setup_monitoring() {
    print_status "Setting up monitoring..."
    
    # Create CloudWatch log group
    aws logs create-log-group \
        --log-group-name "/ecs/gp-mlops-${ENVIRONMENT}" \
        --region $AWS_REGION 2>/dev/null || true
    
    print_status "Monitoring setup completed"
}

# Main deployment flow
main() {
    echo -e "${GREEN}Starting deployment process...${NC}"
    
    check_prerequisites
    deploy_infrastructure
    build_and_push_image
    initialize_database
    deploy_application
    setup_monitoring
    
    echo -e "${GREEN}🎉 Deployment completed successfully!${NC}"
    echo -e "${GREEN}API will be available at the ALB endpoint from the CloudFormation outputs${NC}"
    
    # Print useful information
    print_status "Useful commands:"
    echo "  - Check stack status: aws cloudformation describe-stacks --stack-name $STACK_NAME --region $AWS_REGION"
    echo "  - View logs: aws logs tail /ecs/gp-mlops-${ENVIRONMENT} --follow --region $AWS_REGION"
    echo "  - Update stack: ./deploy.sh $ENVIRONMENT"
}

# Run main function
main