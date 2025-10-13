# Release v${VERSION} - ${RELEASE_DATE}

## 🎉 What's New

- fix: Use patch-based approach for committing release changes (dcc85d1)

## 📦 Docker Image

**Docker Hub:** `fadihussien/tradingtips:${VERSION}`

```bash
# Pull the image
docker pull fadihussien/tradingtips:${VERSION}

# Run locally
docker run -p 8000:8000 -e REDIS_HOST=localhost fadihussien/tradingtips:${VERSION}
```

## 🚀 Deployment

### Using ArgoCD (GitOps)

The Kubernetes manifests have been automatically updated. ArgoCD will sync the changes.

```bash
# Check ArgoCD sync status
argocd app get trading-ai

# Force sync if needed
argocd app sync trading-ai

# Watch deployment
kubectl rollout status deployment/trading-ai
```

### Using kubectl (Manual)

```bash
# Update deployment image
kubectl set image deployment/trading-ai trading-ai=fadihussien/tradingtips:${VERSION}

# Or apply the updated manifest
kubectl apply -f k8s/app/trading-ai-deployment.yaml

# Watch rollout
kubectl rollout status deployment/trading-ai
```

### Using Kustomize

```bash
# Apply all manifests
kubectl apply -k k8s/app/

# Verify deployment
kubectl get pods -l app=trading-ai
```

## 🔍 Verification

After deployment, verify the application is running correctly:

```bash
# Check pod status
kubectl get pods -l app=trading-ai

# Check application version
kubectl exec -it deployment/trading-ai -- python -c "print('Version: ${VERSION}')"

# Check health endpoint
kubectl port-forward svc/trading-ai-service 8000:8000 &
curl http://localhost:8000/api/health

# Check metrics
curl http://localhost:8000/metrics
```

## 📊 Monitoring

Access monitoring dashboards:

- **Grafana:** `http://<node-ip>:32000`
- **Prometheus:** `http://<node-ip>:32090`
- **Trading AI:** `http://<node-ip>:32224`

### Key Metrics to Watch

- `app_info` - Application version and status
- `http_requests_total` - Request rate
- `trading_signals_generated_total` - Signal generation
- `redis_connected_clients` - Redis connections
- `kube_pod_container_status_restarts_total` - Pod restarts

## 🐛 Rollback (If Needed)

If issues occur, rollback to previous version:

```bash
# Using kubectl
kubectl rollout undo deployment/trading-ai

# Or specify revision
kubectl rollout history deployment/trading-ai
kubectl rollout undo deployment/trading-ai --to-revision=<revision>

# Using ArgoCD
argocd app rollback trading-ai <revision>
```

## 📝 Configuration Changes

Review configuration updates in this release:

- **ConfigMaps:** Check `k8s/app/configmap.yaml`
- **Secrets:** Update secrets if needed (use Sealed Secrets)
- **Resources:** Review CPU/Memory limits in deployment

## 🔗 Useful Links

- **GitHub Release:** https://github.com/Fadih/tradeAi/releases/tag/v${VERSION}
- **Docker Hub:** https://hub.docker.com/r/fadihussien/tradingtips
- **Documentation:** https://github.com/Fadih/tradeAi/tree/main/docs
- **Issues:** https://github.com/Fadih/tradeAi/issues

## 👥 Contributors

Thank you to everyone who contributed to this release!

---

**Questions or Issues?** Please open an issue on GitHub.
