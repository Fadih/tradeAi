# Trading AI Application - Kubernetes Manifests

This directory contains Kubernetes manifests for the Trading AI application, optimized for ArgoCD deployment.

## Directory Structure

```
app/
├── kustomization.yaml          # Kustomize configuration
├── configmap.yaml              # Application configuration
├── secrets.yaml                # Sensitive credentials
├── redis.yaml                  # Redis deployment and service
├── trading-ai-deployment.yaml  # Main application deployment
└── README.md                   # This file
```

## Components

### 1. ConfigMap (`configmap.yaml`)
- Application settings (non-sensitive)
- Redis connection parameters
- Feature flags
- Logging configuration
- Server settings

### 2. Secrets (`secrets.yaml`)
- JWT secret key
- Redis password
- Admin credentials
- API keys (Alpaca, CCXT, Telegram, etc.)

### 3. Redis (`redis.yaml`)
- Redis 7 Alpine deployment
- Persistent storage (emptyDir for dev, PVC for prod)
- Health checks and resource limits
- ClusterIP service on port 6379

### 4. Trading AI Application (`trading-ai-deployment.yaml`)
- FastAPI web application
- Prometheus metrics at `/metrics`
- Health checks at `/api/health`
- NodePort service on port 32224
- Config mounted from ConfigMaps
- Secrets injected as environment variables

## Deployment Order (Sync Waves)

ArgoCD deploys resources in this order:

1. **Wave 1:** ConfigMap, Secrets
2. **Wave 2:** Redis deployment
3. **Wave 3:** Trading AI deployment

This ensures dependencies are ready before dependent services start.

## Deployment Methods

### Option 1: ArgoCD (Recommended)

```bash
# Apply the ArgoCD Application
kubectl apply -f k8s/argocd-apps/trading-ai-app.yaml

# Sync the application
argocd app sync trading-ai

# Watch deployment
argocd app get trading-ai --refresh

# Access ArgoCD UI
kubectl port-forward -n argocd svc/argocd-server 8080:443
# Open https://localhost:8080
```

### Option 2: Kustomize

```bash
# Preview what will be deployed
kubectl kustomize k8s/app/

# Apply
kubectl apply -k k8s/app/

# Verify
kubectl get all -l app.kubernetes.io/name=trading-ai
```

### Option 3: kubectl (Direct)

```bash
cd k8s/app
kubectl apply -f configmap.yaml
kubectl apply -f secrets.yaml
kubectl apply -f redis.yaml
kubectl apply -f trading-ai-deployment.yaml
```

### Option 4: Original deploy.sh Script

```bash
cd k8s
./deploy.sh
```

## Access the Application

### Minikube
```bash
# Get Minikube IP
MINIKUBE_IP=$(minikube ip)

# Access the app
open http://$MINIKUBE_IP:32224

# Or port-forward
kubectl port-forward svc/trading-ai-service 8000:8000
open http://localhost:8000
```

### Rancher/External Cluster
```bash
# Get node IP
kubectl get nodes -o wide

# Access at
# http://<node-ip>:32224
```

## Configuration Updates

### Update ConfigMap
```bash
# Edit configmap.yaml
vim k8s/app/configmap.yaml

# Commit and push (ArgoCD auto-syncs)
git add k8s/app/configmap.yaml
git commit -m "Update config"
git push

# Or apply manually
kubectl apply -f k8s/app/configmap.yaml
kubectl rollout restart deployment/trading-ai
```

### Update Secrets

**IMPORTANT:** Never commit unencrypted secrets to Git!

For production, use:
- Sealed Secrets (see `k8s/SECRET_MANAGEMENT_GUIDE.md`)
- External Secrets Operator
- HashiCorp Vault
- Cloud provider secret managers (AWS Secrets Manager, GCP Secret Manager, etc.)

```bash
# For development only (not for production!)
kubectl apply -f k8s/app/secrets.yaml
kubectl rollout restart deployment/trading-ai
```

## Monitoring

The application exposes Prometheus metrics at `/metrics`:

```bash
# Check metrics
kubectl port-forward deployment/trading-ai 8000:8000
curl http://localhost:8000/metrics
```

Available metrics:
- `http_requests_total` - HTTP request count
- `http_request_duration_seconds` - Request latency histogram
- `trading_signals_total` - Total signals generated
- `trading_signals_generated_total` - Signals by type and symbol
- `trading_active_monitors` - Active monitoring jobs
- `redis_connected_clients` - Redis connections
- `redis_memory_used_bytes` - Redis memory usage
- `redis_db_keys` - Keys in Redis
- `app_uptime_seconds` - Application uptime
- `app_info` - Application version info

## Health Checks

```bash
# Application health
kubectl port-forward svc/trading-ai-service 8000:8000
curl http://localhost:8000/api/health

# Redis health
kubectl exec -it deployment/redis -- redis-cli ping
```

## Troubleshooting

### Pods Not Starting
```bash
# Check pod status
kubectl get pods -l app=trading-ai

# View logs
kubectl logs deployment/trading-ai -f

# Describe pod
kubectl describe pod -l app=trading-ai
```

### Redis Connection Issues
```bash
# Check Redis service
kubectl get svc redis-service

# Test Redis from app pod
kubectl exec -it deployment/trading-ai -- sh -c "python -c 'import os; print(os.getenv(\"REDIS_HOST\"))'"

# Check Redis logs
kubectl logs deployment/redis -f
```

### Configuration Not Loading
```bash
# Check ConfigMap
kubectl get cm trading-ai-config -o yaml

# Check if mounted correctly
kubectl exec -it deployment/trading-ai -- ls -la /app/config

# Check environment variables
kubectl exec -it deployment/trading-ai -- env | grep REDIS
```

## Scaling

```bash
# Scale Trading AI (be careful with scheduler conflicts)
kubectl scale deployment/trading-ai --replicas=2

# Note: Keep replicas=1 to avoid:
# - Multiple auth token stores
# - Duplicate scheduler jobs
# - Race conditions in signal generation
```

## Cleanup

```bash
# With ArgoCD
argocd app delete trading-ai

# With kubectl
kubectl delete -k k8s/app/

# Or delete individually
kubectl delete -f k8s/app/
```

## Next Steps

1. Set up monitoring: See `k8s/monitoring/README.md`
2. Configure sealed secrets: See `k8s/SECRET_MANAGEMENT_GUIDE.md`
3. Set up ingress: See `k8s/ingress.yaml`
4. Review production checklist: See `PRODUCTION_DEPLOYMENT.md`

