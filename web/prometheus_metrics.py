#!/usr/bin/env python3
"""
Prometheus Metrics for Trading AI Application
Exposes application metrics for Prometheus scraping
"""

from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client import CollectorRegistry
from fastapi import Response
import time

# Create a custom registry
registry = CollectorRegistry()

# HTTP Metrics
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'path', 'status'],
    registry=registry
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'path'],
    registry=registry
)

# Trading Signal Metrics
trading_signals_generated_total = Counter(
    'trading_signals_generated_total',
    'Total number of trading signals generated',
    ['signal_type', 'symbol'],
    registry=registry
)

trading_signals_total = Gauge(
    'trading_signals_total',
    'Current total number of signals',
    registry=registry
)

trading_active_monitors = Gauge(
    'trading_active_monitors',
    'Number of active monitoring jobs',
    registry=registry
)

# Redis Metrics
redis_connected_clients = Gauge(
    'redis_connected_clients',
    'Number of Redis connected clients',
    registry=registry
)

redis_commands_processed_total = Counter(
    'redis_commands_processed_total',
    'Total number of Redis commands processed',
    registry=registry
)

redis_memory_used_bytes = Gauge(
    'redis_memory_used_bytes',
    'Redis memory usage in bytes',
    registry=registry
)

redis_memory_max_bytes = Gauge(
    'redis_memory_max_bytes',
    'Redis max memory in bytes',
    registry=registry
)

redis_db_keys = Gauge(
    'redis_db_keys',
    'Number of keys in Redis database',
    ['db'],
    registry=registry
)

# Application Metrics
app_uptime_seconds = Gauge(
    'app_uptime_seconds',
    'Application uptime in seconds',
    registry=registry
)

app_info = Gauge(
    'app_info',
    'Application information',
    ['version', 'name'],
    registry=registry
)


def metrics_response():
    """Generate Prometheus metrics response"""
    return Response(
        content=generate_latest(registry),
        media_type=CONTENT_TYPE_LATEST
    )


class PrometheusMiddleware:
    """Middleware to track HTTP requests"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        method = scope["method"]
        path = scope["path"]
        
        # Skip metrics endpoint to avoid recursion
        if path == "/metrics":
            await self.app(scope, receive, send)
            return
        
        start_time = time.time()
        
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status = message["status"]
                duration = time.time() - start_time
                
                # Record metrics
                http_requests_total.labels(
                    method=method,
                    path=path,
                    status=status
                ).inc()
                
                http_request_duration_seconds.labels(
                    method=method,
                    path=path
                ).observe(duration)
            
            await send(message)
        
        await self.app(scope, receive, send_wrapper)

