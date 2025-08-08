from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
import uvicorn

from app.api.endpoints import customer, floorplan, predictions, health, mlops
from app.core.config import settings
from app.core.database import engine, Base
from app.core.monitoring import setup_prometheus_metrics

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="GA Technologies MLOps Platform",
    description="Real Estate MLOps Platform for Customer Value Inference and Floorplan Analysis",
    version="1.0.0"
)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup monitoring
setup_prometheus_metrics(app)

# Security
security = HTTPBearer()

# Include routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(customer.router, prefix="/api/v1/customer", tags=["customer"])
app.include_router(floorplan.router, prefix="/api/v1/floorplan", tags=["floorplan"])
app.include_router(predictions.router, prefix="/api/v1/predictions", tags=["predictions"])

# Import and include advertising router
from app.api.endpoints import advertising, conversion_api
app.include_router(advertising.router, prefix="/api/v1/ads", tags=["advertising"])
app.include_router(conversion_api.router, prefix="/api/v1/capi", tags=["conversion-api"])

# Include MLOps router
app.include_router(mlops.router, prefix="/api/v1", tags=["mlops"])

@app.get("/")
async def root():
    return {
        "message": "GA Technologies MLOps Platform", 
        "version": "1.0.0",
        "services": ["customer_inference", "floorplan_detection", "conversion_api", "mlops_pipeline", "dify_workflows"]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)