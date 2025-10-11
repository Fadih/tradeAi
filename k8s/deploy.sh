#!/bin/bash

# Trading AI Kubernetes Deployment Script
# This script deploys the Trading AI application to Kubernetes

set -e  # Exit on any error

echo "🚀 Starting Trading AI Kubernetes Deployment"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    print_error "kubectl is not installed or not in PATH"
    exit 1
fi

# Check if minikube is running
if ! kubectl cluster-info &> /dev/null; then
    print_error "Kubernetes cluster is not running. Please start minikube first:"
    echo "  minikube start"
    exit 1
fi

print_status "Kubernetes cluster is running"

# Build the Docker image in minikube
print_status "Building Docker image in minikube..."
eval $(minikube docker-env)
docker build -t trading-ai:latest --target production ..

if [ $? -eq 0 ]; then
    print_success "Docker image built successfully"
else
    print_error "Failed to build Docker image"
    exit 1
fi

# Apply Kubernetes manifests in order
print_status "Applying Kubernetes manifests..."

# 1. ConfigMap
print_status "Creating ConfigMap..."
kubectl apply -f configmap.yaml
print_success "ConfigMap created"

# 2. Secrets
print_status "Creating Secrets..."
kubectl apply -f secrets.yaml
print_success "Secrets created"

# 3. Redis
print_status "Deploying Redis..."
kubectl apply -f redis.yaml
print_success "Redis deployment created"

# Wait for Redis to be ready
print_status "Waiting for Redis to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/redis
print_success "Redis is ready"

# 4. Trading AI Application
print_status "Deploying Trading AI application..."
kubectl apply -f trading-ai-deployment.yaml
print_success "Trading AI deployment created"

# Wait for Trading AI to be ready
print_status "Waiting for Trading AI application to be ready..."
kubectl wait --for=condition=available --timeout=600s deployment/trading-ai
print_success "Trading AI application is ready"

# Show deployment status
echo ""
print_status "Deployment Status:"
echo "==================="
kubectl get pods -l app=trading-ai
kubectl get pods -l app=redis

echo ""
print_status "Services:"
echo "=========="
kubectl get services

echo ""
print_status "Access Information:"
echo "======================"
echo "To access the application:"
echo "1. Port forward: kubectl port-forward service/trading-ai-service 8000:8000"
echo "2. Open browser: http://localhost:8000"
echo ""
echo "To access Redis:"
echo "1. Port forward: kubectl port-forward service/redis-service 6379:6379"
echo "2. Connect with: redis-cli -h localhost -p 6379"

print_success "Deployment completed successfully! 🎉"
