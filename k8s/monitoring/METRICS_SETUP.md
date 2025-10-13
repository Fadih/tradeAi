# Trading AI Prometheus Metrics Setup

This document explains how to enable metrics collection for the Trading AI application.

## Overview

The application now exposes Prometheus metrics at the `/metrics` endpoint, which includes:

- **HTTP Metrics**: Request count, duration, status codes
- **Trading Metrics**: Signals generated, active monitors
- **Redis Metrics**: Connections, commands, memory usage
- **Application Metrics**: Uptime, version info

## Quick Start

### 1. Build and Deploy with Metrics Support

```bash
# Build the Docker image with the new metrics code
cd k8s
eval $(minikube docker-env)  # For minikube
docker build -t trading-ai:latest --target production ..

# Deploy or restart
kubectl rollout restart deployment/trading-ai
```

### 2. Verify Metrics Endpoint

```bash
# Port-forward to the app
kubectl port-forward deployment/trading-ai 8000:8000

# Check metrics
curl http://localhost:8000/metrics
```

You should see output like:
```
# HELP http_requests_total Total HTTP requests
# TYPE http_requests_total counter
http_requests_total{method="GET",path="/api/health",status="200"} 42.0
...
```

### 3. Check Prometheus is Scraping

```bash
# Port-forward to Prometheus
kubectl port-forward -n monitoring svc/prometheus 9090:9090

# Open http://localhost:9090/targets
# Look for the "trading-ai" job - it should show as "UP"
```

### 4. View Metrics in Grafana

```bash
# Access Grafana
kubectl port-forward -n monitoring svc/grafana 3000:3000

# Login: admin / admin123
# Navigate to the "Trading AI - Complete Dashboard"
```

## Available Metrics

### HTTP Metrics

```promql
# Total requests
http_requests_total

# Request duration (histogram)
http_request_duration_seconds

# Example queries:
rate(http_requests_total[5m])
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))
```

### Trading Metrics

```promql
# Signals generated
trading_signals_generated_total{signal_type="BUY"}
trading_signals_generated_total{signal_type="SELL"}

# Total signals
trading_signals_total

# Active monitors
trading_active_monitors
```

### Redis Metrics

```promql
# Connected clients
redis_connected_clients

# Commands processed
rate(redis_commands_processed_total[5m])

# Memory usage
redis_memory_used_bytes
redis_memory_max_bytes

# Database keys
redis_db_keys
```

### Application Metrics

```promql
# App uptime
app_uptime_seconds

# App info
app_info{version="1.0.0",name="Trading AI"}
```

## Troubleshooting

### No Data in Dashboards

1. **Check if metrics endpoint is accessible:**
   ```bash
   kubectl exec -it deployment/trading-ai -- curl localhost:8000/metrics
   ```

2. **Check if Prometheus is scraping:**
   - Open Prometheus UI: http://localhost:9090/targets
   - Find the `trading-ai` target
   - Check Status column (should be "UP")
   - Check "Last Scrape" time

3. **Check Prometheus configuration:**
   ```bash
   kubectl get cm prometheus-config -n monitoring -o yaml
   ```

4. **View Prometheus logs:**
   ```bash
   kubectl logs -n monitoring deployment/prometheus -f
   ```

5. **Check pod annotations:**
   ```bash
   kubectl get pods -l app=trading-ai -o yaml | grep -A 3 annotations
   ```

### Metrics Not Updating

1. **Restart the application:**
   ```bash
   kubectl rollout restart deployment/trading-ai
   ```

2. **Check application logs:**
   ```bash
   kubectl logs deployment/trading-ai -f | grep -i metric
   ```

3. **Verify prometheus-client is installed:**
   ```bash
   kubectl exec -it deployment/trading-ai -- pip list | grep prometheus
   ```

### Prometheus Can't Reach Application

1. **Check service endpoints:**
   ```bash
   kubectl get endpoints trading-ai-service
   ```

2. **Test connectivity from Prometheus pod:**
   ```bash
   kubectl exec -it -n monitoring deployment/prometheus -- wget -O- http://trading-ai-service.default.svc:8000/metrics
   ```

3. **Check network policies:**
   ```bash
   kubectl get networkpolicies -n default
   kubectl get networkpolicies -n monitoring
   ```

## Adding Custom Metrics

To add new metrics to your application:

1. **Define the metric in `web/prometheus_metrics.py`:**
   ```python
   from prometheus_client import Counter
   
   my_custom_metric = Counter(
       'my_custom_metric',
       'Description of my metric',
       ['label1', 'label2'],
       registry=registry
   )
   ```

2. **Use it in your code:**
   ```python
   from .prometheus_metrics import my_custom_metric
   
   # Increment counter
   my_custom_metric.labels(label1='value1', label2='value2').inc()
   ```

3. **Query it in Prometheus:**
   ```promql
   rate(my_custom_metric[5m])
   ```

## Next Steps

- Add alerting rules in Prometheus
- Create custom Grafana dashboards
- Export metrics to external monitoring systems
- Set up long-term metrics storage (e.g., Thanos, Cortex)


