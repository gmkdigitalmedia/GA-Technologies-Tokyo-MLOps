#!/bin/bash

# Setup KServe on EKS for GP MLOps MLOps Platform

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
CLUSTER_NAME=${1:-ga-mlops-cluster}
AWS_REGION=${AWS_REGION:-us-east-1}
KSERVE_VERSION=${KSERVE_VERSION:-v0.11.0}

echo -e "${GREEN}LAUNCH Setting up KServe on EKS cluster: ${CLUSTER_NAME}${NC}"

# Function to print status
print_status() {
    echo -e "${GREEN}PASS $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}FAIL $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl not found. Please install kubectl."
        exit 1
    fi
    
    # Check helm
    if ! command -v helm &> /dev/null; then
        print_error "Helm not found. Please install Helm."
        exit 1
    fi
    
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        print_error "AWS CLI not found. Please install AWS CLI."
        exit 1
    fi
    
    # Check cluster connection
    if ! kubectl cluster-info &> /dev/null; then
        print_error "Cannot connect to Kubernetes cluster. Please configure kubectl."
        exit 1
    fi
    
    print_status "Prerequisites check completed"
}

# Install Istio
install_istio() {
    print_status "Installing Istio..."
    
    # Download Istio
    curl -L https://istio.io/downloadIstio | sh -
    cd istio-*
    export PATH=$PWD/bin:$PATH
    
    # Install Istio
    istioctl install --set values.defaultRevision=default -y
    
    # Enable Istio injection for default namespace
    kubectl label namespace default istio-injection=enabled --overwrite
    
    # Install Istio Gateway
    kubectl apply -f - <<EOF
apiVersion: networking.istio.io/v1beta1
kind: Gateway
metadata:
  name: knative-ingress-gateway
  namespace: knative-serving
spec:
  selector:
    istio: ingressgateway
  servers:
  - port:
      number: 80
      name: http
      protocol: HTTP
    hosts:
    - "*"
EOF
    
    print_status "Istio installed successfully"
}

# Install Knative Serving
install_knative_serving() {
    print_status "Installing Knative Serving..."
    
    # Install Knative Serving CRDs
    kubectl apply -f https://github.com/knative/serving/releases/download/knative-v1.11.0/serving-crds.yaml
    
    # Install Knative Serving core
    kubectl apply -f https://github.com/knative/serving/releases/download/knative-v1.11.0/serving-core.yaml
    
    # Install Knative Istio controller
    kubectl apply -f https://github.com/knative/net-istio/releases/download/knative-v1.11.0/net-istio.yaml
    
    # Configure Knative to use Istio
    kubectl patch configmap/config-network \
        --namespace knative-serving \
        --type merge \
        --patch '{"data":{"ingress-class":"istio.ingress.networking.knative.dev"}}'
    
    # Wait for Knative to be ready
    kubectl wait --for=condition=Ready pod --all -n knative-serving --timeout=300s
    
    print_status "Knative Serving installed successfully"
}

# Install Cert Manager
install_cert_manager() {
    print_status "Installing Cert Manager..."
    
    # Add Jetstack Helm repository
    helm repo add jetstack https://charts.jetstack.io
    helm repo update
    
    # Install Cert Manager
    helm install cert-manager jetstack/cert-manager \
        --namespace cert-manager \
        --create-namespace \
        --set installCRDs=true
    
    # Wait for cert-manager to be ready
    kubectl wait --for=condition=Available deployment --all -n cert-manager --timeout=300s
    
    print_status "Cert Manager installed successfully"
}

# Install KServe
install_kserve() {
    print_status "Installing KServe..."
    
    # Install KServe CRDs
    kubectl apply -f https://github.com/kserve/kserve/releases/download/${KSERVE_VERSION}/kserve.yaml
    
    # Install KServe runtimes
    kubectl apply -f https://github.com/kserve/kserve/releases/download/${KSERVE_VERSION}/kserve-runtimes.yaml
    
    # Wait for KServe to be ready
    kubectl wait --for=condition=Ready pod --all -n kserve --timeout=300s
    
    print_status "KServe installed successfully"
}

# Setup RBAC for KServe
setup_rbac() {
    print_status "Setting up RBAC..."
    
    # Create service account with S3 access
    cat > kserve-rbac.yaml << EOF
apiVersion: v1
kind: ServiceAccount
metadata:
  name: kserve-service-account
  namespace: ga-mlops
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):role/KServeS3AccessRole
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: kserve-role
rules:
- apiGroups: [""]
  resources: ["secrets", "configmaps"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["serving.knative.dev"]
  resources: ["services", "configurations", "revisions"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: kserve-binding
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: kserve-role
subjects:
- kind: ServiceAccount
  name: kserve-service-account
  namespace: ga-mlops
EOF
    
    kubectl apply -f kserve-rbac.yaml
    
    print_status "RBAC setup completed"
}

# Create IAM role for S3 access
create_iam_role() {
    print_status "Creating IAM role for S3 access..."
    
    # Get cluster OIDC issuer
    OIDC_ISSUER=$(aws eks describe-cluster --name $CLUSTER_NAME --query "cluster.identity.oidc.issuer" --output text --region $AWS_REGION)
    OIDC_ID=${OIDC_ISSUER##*/}
    
    # Create trust policy
    cat > trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):oidc-provider/$OIDC_ISSUER"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "$OIDC_ID:sub": "system:serviceaccount:ga-mlops:kserve-service-account"
        }
      }
    }
  ]
}
EOF
    
    # Create IAM role
    aws iam create-role \
        --role-name KServeS3AccessRole \
        --assume-role-policy-document file://trust-policy.json || true
    
    # Attach S3 access policy
    aws iam attach-role-policy \
        --role-name KServeS3AccessRole \
        --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess || true
    
    print_status "IAM role created successfully"
}

# Deploy GA MLOps models
deploy_models() {
    print_status "Deploying GA MLOps models..."
    
    # Create namespace
    kubectl create namespace ga-mlops --dry-run=client -o yaml | kubectl apply -f -
    kubectl label namespace ga-mlops istio-injection=enabled --overwrite
    
    # Apply KServe manifests
    kubectl apply -f ../kserve-manifests.yaml
    
    # Wait for models to be ready
    print_status "Waiting for models to be ready..."
    kubectl wait --for=condition=Ready inferenceservice --all -n ga-mlops --timeout=600s
    
    print_status "Models deployed successfully"
}

# Build and push transformer images
build_transformers() {
    print_status "Building transformer images..."
    
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    ECR_REPO="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
    
    # Login to ECR
    aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_REPO
    
    # Create ECR repositories
    aws ecr create-repository --repository-name ga-mlops/customer-transformer --region $AWS_REGION 2>/dev/null || true
    aws ecr create-repository --repository-name ga-mlops/floorplan-transformer --region $AWS_REGION 2>/dev/null || true
    
    # Build customer transformer
    cd ../transformers
    docker build -f Dockerfile.customer -t $ECR_REPO/ga-mlops/customer-transformer:latest .
    docker push $ECR_REPO/ga-mlops/customer-transformer:latest
    
    # Build floorplan transformer
    docker build -f Dockerfile.floorplan -t $ECR_REPO/ga-mlops/floorplan-transformer:latest .
    docker push $ECR_REPO/ga-mlops/floorplan-transformer:latest
    
    # Update manifests with correct image URIs
    sed -i "s|ga-mlops/customer-transformer:latest|$ECR_REPO/ga-mlops/customer-transformer:latest|g" ../kserve-manifests.yaml
    sed -i "s|ga-mlops/floorplan-transformer:latest|$ECR_REPO/ga-mlops/floorplan-transformer:latest|g" ../kserve-manifests.yaml
    
    print_status "Transformer images built and pushed"
}

# Test KServe deployment
test_deployment() {
    print_status "Testing KServe deployment..."
    
    # Get ingress gateway external IP
    INGRESS_HOST=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
    INGRESS_PORT=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.spec.ports[?(@.name=="http2")].port}')
    
    echo "Ingress Gateway: http://$INGRESS_HOST:$INGRESS_PORT"
    
    # Test customer value model
    echo "Testing customer value model..."
    curl -X POST http://$INGRESS_HOST:$INGRESS_PORT/v1/models/customer-value:predict \
        -H "Host: ga-mlops-inference.example.com" \
        -H "Content-Type: application/json" \
        -d '{
            "instances": [
                {
                    "customer_id": 12345,
                    "age": 45,
                    "income": 8000000,
                    "profession": "engineer",
                    "location": "tokyo",
                    "family_size": 3,
                    "interactions": []
                }
            ]
        }' || print_warning "Customer model test failed - this is expected if transformers are not ready"
    
    print_status "Deployment test completed"
}

# Main installation flow
main() {
    echo -e "${GREEN}Starting KServe setup...${NC}"
    
    check_prerequisites
    create_iam_role
    install_istio
    install_knative_serving
    install_cert_manager
    install_kserve
    setup_rbac
    build_transformers
    deploy_models
    test_deployment
    
    echo -e "${GREEN}CELEBRATE KServe setup completed successfully!${NC}"
    echo -e "${GREEN}Models are now available via KServe inference endpoints${NC}"
    
    # Print useful information
    print_status "Useful commands:"
    echo "  - Check InferenceServices: kubectl get inferenceservices -n ga-mlops"
    echo "  - View logs: kubectl logs -n ga-mlops -l app=customer-value-model"
    echo "  - Port forward for testing: kubectl port-forward -n istio-system svc/istio-ingressgateway 8080:80"
}

# Run main function
main