# Deployment Guide - v${VERSION}

## Prerequisites

- Kubernetes cluster (1.24+)
- kubectl configured
- ArgoCD installed (optional but recommended)
- Docker Hub account

## Quick Deployment

### 1. Clone Repository

```bash
git clone https://github.com/Fadih/tradeAi.git
cd tradeAi
git checkout v${VERSION}
```

### 2. Configure Secrets

```bash
# Copy example secrets
cp k8s/app/secrets.example.yaml k8s/app/secrets.yaml

# Edit with your values
vim k8s/app/secrets.yaml
```

**Required Secrets:**
- `JWT_SECRET_KEY` - JWT token secret
- `REDIS_PASSWORD` - Redis password
- `DEFAULT_ADMIN_PASSWORD` - Admin password
- `CCXT_API_KEY` - Binance API key
- `CCXT_API_SECRET` - Binance API secret
- `ALPACA_KEY_ID` - Alpaca key
- `ALPACA_SECRET_KEY` - Alpaca secret
- `HF_TOKEN` - Hugging Face token
- `TELEGRAM_BOT_TOKEN` - Telegram bot token
- `TELEGRAM_CHAT_ID` - Telegram chat ID

### 3. Deploy with Kustomize

```bash
kubectl apply -k k8s/app/
```

### 4. Verify Deployment

```bash
# Check pods
kubectl get pods -l app=trading-ai

# Check services
kubectl get svc

# Check logs
kubectl logs -f deployment/trading-ai
```

## ArgoCD Deployment

### 1. Install ArgoCD

```bash
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
```

### 2. Access ArgoCD UI

```bash
kubectl port-forward svc/argocd-server -n argocd 8080:443
```

### 3. Login

```bash
# Get initial password
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d

# Login with argocd CLI
argocd login localhost:8080
```

### 4. Create Application

```bash
kubectl apply -f k8s/argocd-apps/trading-ai-app.yaml
```

### 5. Sync Application

```bash
argocd app sync trading-ai
```

## Monitoring Stack Deployment

### Deploy Prometheus & Grafana

```bash
kubectl apply -k k8s/monitoring/
```

### Access Grafana

```bash
# Get Grafana URL
kubectl get svc grafana -n monitoring

# Port forward
kubectl port-forward -n monitoring svc/grafana 3000:3000

# Open http://localhost:3000
# Username: admin
# Password: admin123
```

## Production Considerations

### 1. Use Sealed Secrets

```bash
# Install Sealed Secrets controller
kubectl apply -f https://github.com/bitnami-labs/sealed-secrets/releases/download/v0.24.0/controller.yaml

# Seal your secrets
kubeseal --format yaml < k8s/app/secrets.yaml > k8s/app/sealed-secrets.yaml
```

### 2. Configure Ingress

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: trading-ai-ingress
spec:
  rules:
  - host: trading.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: trading-ai-service
            port:
              number: 8000
```

### 3. Enable TLS

```bash
# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Create certificate issuer
kubectl apply -f k8s/cert-issuer.yaml
```

### 4. Configure Resource Limits

Adjust in `k8s/app/trading-ai-deployment.yaml`:

```yaml
resources:
  requests:
    cpu: "500m"
    memory: "512Mi"
  limits:
    cpu: "2000m"
    memory: "2Gi"
```

### 5. Enable Horizontal Pod Autoscaling

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: trading-ai-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: trading-ai
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## Troubleshooting

### Pods Not Starting

```bash
# Check pod events
kubectl describe pod <pod-name>

# Check logs
kubectl logs <pod-name>

# Check resources
kubectl top pods
```

### Redis Connection Issues

```bash
# Test Redis connection
kubectl exec -it deployment/redis -- redis-cli ping

# Check Redis password
kubectl get secret trading-ai-secrets -o jsonpath='{.data.REDIS_PASSWORD}' | base64 -d
```

### Image Pull Errors

```bash
# Check image exists
docker pull fadihussien/tradingtips:${VERSION}

# Verify image in deployment
kubectl get deployment trading-ai -o jsonpath='{.spec.template.spec.containers[0].image}'
```

## Backup & Recovery

### Backup Redis Data

```bash
kubectl exec deployment/redis -- redis-cli BGSAVE
kubectl cp <redis-pod>:/data/dump.rdb ./backup/dump.rdb
```

### Backup Kubernetes Resources

```bash
kubectl get all -o yaml > backup/resources.yaml
kubectl get configmap,secret -o yaml > backup/configs.yaml
```

## Support

- **GitHub Issues:** https://github.com/Fadih/tradeAi/issues
- **Documentation:** https://github.com/Fadih/tradeAi/tree/main/docs
- **Discussions:** https://github.com/Fadih/tradeAi/discussions

---

**Deployed successfully? Star the repo!** ⭐
