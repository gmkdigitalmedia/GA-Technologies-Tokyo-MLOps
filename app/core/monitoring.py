from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import FastAPI, Request, Response
from fastapi.responses import Response as FastAPIResponse
import time

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration', ['method', 'endpoint'])
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Active connections')
MODEL_PREDICTION_COUNT = Counter('model_predictions_total', 'Total model predictions', ['model_type', 'status'])
MODEL_PREDICTION_DURATION = Histogram('model_prediction_duration_seconds', 'Model prediction duration', ['model_type'])

def setup_prometheus_metrics(app: FastAPI):
    @app.middleware("http")
    async def prometheus_middleware(request: Request, call_next):
        start_time = time.time()
        
        # Increment active connections
        ACTIVE_CONNECTIONS.inc()
        
        try:
            response = await call_next(request)
            
            # Record metrics
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code
            ).inc()
            
            REQUEST_DURATION.labels(
                method=request.method,
                endpoint=request.url.path
            ).observe(time.time() - start_time)
            
            return response
            
        finally:
            # Decrement active connections
            ACTIVE_CONNECTIONS.dec()
    
    @app.get("/metrics")
    async def metrics():
        return FastAPIResponse(generate_latest(), media_type="text/plain")

def record_model_prediction(model_type: str, duration: float, success: bool = True):
    """Record model prediction metrics"""
    status = "success" if success else "error"
    MODEL_PREDICTION_COUNT.labels(model_type=model_type, status=status).inc()
    MODEL_PREDICTION_DURATION.labels(model_type=model_type).observe(duration)