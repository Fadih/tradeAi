# Trading AI Monitoring Stack

This directory contains the monitoring stack (Prometheus + Grafana) for the Trading AI application, designed for deployment via ArgoCD.

## Directory Structure

```
monitoring/
├── namespace.yaml                      # Monitoring namespace
├── kustomization.yaml                  # Kustomize configuration
├── prometheus/
│   ├── configmap.yaml                 # Prometheus scrape config
│   └── deployment.yaml                # Prometheus deployment + service + RBAC
└── grafana/
    ├── deployment.yaml                # Grafana deployment + service + secrets
    └── dashboard-configmap.yaml       # Trading AI custom dashboard
```

## Deployment Options

### Option 1: ArgoCD (Recommended)

1. **Update the ArgoCD Application manifest:**
   ```bash
   # Edit k8s/argocd-apps/monitoring-app.yaml
   # Update spec.source.repoURL with your Git repository URL
   ```

2. **Apply the ArgoCD Application:**
   ```bash
   kubectl apply -f k8s/argocd-apps/monitoring-app.yaml
   ```

3. **Check ArgoCD UI or CLI:**
   ```bash
   argocd app get monitoring-stack
   argocd app sync monitoring-stack
   ```

### Option 2: Kustomize

```bash
kubectl apply -k k8s/monitoring/
```

### Option 3: Direct kubectl

```bash
kubectl apply -f k8s/monitoring/namespace.yaml
kubectl apply -f k8s/monitoring/prometheus/
kubectl apply -f k8s/monitoring/grafana/
```

## Access

### NodePort Access

- **Grafana:** `http://<node-ip>:32000`
- **Prometheus:** `http://<node-ip>:32090`

### Port-Forward Access

```bash
# Grafana
kubectl port-forward -n monitoring svc/grafana 3000:3000

# Prometheus
kubectl port-forward -n monitoring svc/prometheus 9090:9090
```

### Credentials

- **Username:** admin
- **Password:** admin123

## Features

- ✅ **Prometheus:** Metrics collection with 15-day retention
- ✅ **Grafana:** Pre-configured dashboards
- ✅ **Auto-discovery:** Kubernetes pods, nodes, services
- ✅ **Trading AI Dashboard:** Custom metrics for your application
- ✅ **Redis Monitoring:** Track Redis performance
- ✅ **RBAC:** Properly configured service accounts

## Monitoring Your Trading AI App

The stack automatically discovers and monitors:
- Application pods (CPU, memory, network, disk)
- HTTP endpoints (request rate, response time)
- Redis connections and performance
- Trading signals generation rate
- Active monitoring jobs

## Customization

### Update Prometheus Scrape Config
Edit `prometheus/configmap.yaml` to add custom scrape targets.

### Add New Dashboards
Add dashboard JSON to `grafana/dashboard-configmap.yaml`.

### Change Credentials
Edit `grafana/deployment.yaml` and update the Secret `grafana-credentials`.

## Troubleshooting

```bash
# Check pod status
kubectl get pods -n monitoring

# View Prometheus logs
kubectl logs -n monitoring deployment/prometheus -f

# View Grafana logs
kubectl logs -n monitoring deployment/grafana -f

# Check if metrics are being scraped
kubectl port-forward -n monitoring svc/prometheus 9090:9090
# Open http://localhost:9090/targets
```

## Cleanup

### With ArgoCD
```bash
argocd app delete monitoring-stack
```

### With kubectl
```bash
kubectl delete -k k8s/monitoring/
# Or
kubectl delete namespace monitoring
```

