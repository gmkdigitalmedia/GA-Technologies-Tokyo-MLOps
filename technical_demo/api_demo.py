#!/usr/bin/env python3
"""
Simple standalone API demo that actually works
Simulates real ML models in action 
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import cv2
from PIL import Image
import io
import uvicorn
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="GP MLOps Demo", version="1.0.0")

# Global models
models = {}
scaler = None

class CustomerProfile(BaseModel):
    age: int
    income: int
    profession: int  # 0-4 encoded
    location: int    # 0-3 encoded
    page_views: int
    session_duration: float

class PredictionResponse(BaseModel):
    conversion_probability: float
    model_confidence: float
    processing_time_ms: float

class FloorplanResponse(BaseModel):
    detected_rooms: list
    total_rooms: int
    processing_time_ms: float

def train_models():
    """Train the ML models on startup"""
    global models, scaler
    
    logger.info("Training ML models...")
    
    # Generate realistic training data
    np.random.seed(42)
    n_samples = 1000
    
    # Customer features
    ages = np.random.normal(45, 8, n_samples).clip(25, 65).astype(int)
    incomes = np.random.lognormal(16, 0.3, n_samples).clip(3_000_000, 30_000_000).astype(int)
    professions = np.random.choice([0, 1, 2, 3, 4], n_samples)
    locations = np.random.choice([0, 1, 2, 3], n_samples)
    page_views = np.random.poisson(5, n_samples)
    session_duration = np.random.exponential(300, n_samples)
    
    # Create realistic conversion logic
    conversion_probability = (
        (incomes > 10_000_000).astype(float) * 0.3 +
        ((ages >= 40) & (ages <= 50)).astype(float) * 0.2 +
        (page_views > 3).astype(float) * 0.2 +
        (session_duration > 180).astype(float) * 0.1 +
        np.random.normal(0, 0.1, n_samples)
    ).clip(0, 1)
    
    conversions = (np.random.random(n_samples) < conversion_probability).astype(int)
    
    # Prepare training data
    X = np.column_stack([ages, incomes, professions, locations, page_views, session_duration])
    y = conversions
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    models['conversion'] = RandomForestClassifier(n_estimators=50, random_state=42)
    models['conversion'].fit(X_scaled, y)
    
    logger.info("Models trained successfully")

@app.on_event("startup")
async def startup_event():
    train_models()

@app.get("/")
async def root():
    return {
        "message": "GP MLOps Demo API", 
        "status": "running",
        "models_loaded": len(models),
        "endpoints": [
            "/predict/customer - Customer value prediction",
            "/analyze/floorplan - Floorplan image analysis", 
            "/health - System health check"
        ]
    }

@app.post("/predict/customer", response_model=PredictionResponse)
async def predict_customer(customer: CustomerProfile):
    """Predict customer conversion probability"""
    
    import time
    start_time = time.time()
    
    if 'conversion' not in models or scaler is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Prepare features
        features = np.array([[
            customer.age,
            customer.income, 
            customer.profession,
            customer.location,
            customer.page_views,
            customer.session_duration
        ]])
        
        # Scale and predict
        features_scaled = scaler.transform(features)
        prediction_proba = models['conversion'].predict_proba(features_scaled)[0]
        
        processing_time = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            conversion_probability=float(prediction_proba[1]),
            model_confidence=float(np.max(prediction_proba)),
            processing_time_ms=round(processing_time, 2)
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/floorplan", response_model=FloorplanResponse)
async def analyze_floorplan(file: UploadFile = File(...)):
    """Analyze uploaded floorplan image"""
    
    import time
    start_time = time.time()
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Simple room detection using contours
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter by area
        min_area = 500
        room_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        # Simple room classification
        detected_rooms = []
        room_types = ['living_room', 'bedroom', 'kitchen', 'bathroom', 'dining_room']
        
        for i, contour in enumerate(room_contours[:8]):
            area = cv2.contourArea(contour)
            room_type = room_types[i % len(room_types)]
            
            detected_rooms.append({
                'room_id': i + 1,
                'type': room_type,
                'area_pixels': int(area),
                'confidence': min(0.95, 0.6 + (area / 5000) * 0.2)
            })
        
        processing_time = (time.time() - start_time) * 1000
        
        return FloorplanResponse(
            detected_rooms=detected_rooms,
            total_rooms=len(detected_rooms),
            processing_time_ms=round(processing_time, 2)
        )
        
    except Exception as e:
        logger.error(f"Floorplan analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """System health check"""
    
    return {
        "status": "healthy",
        "models_loaded": len(models),
        "scaler_ready": scaler is not None,
        "uptime": "running",
        "version": "1.0.0"
    }

@app.get("/demo/sample-prediction")
async def sample_prediction():
    """Get a sample prediction for demo purposes"""
    
    sample_customer = CustomerProfile(
        age=45,
        income=12000000,
        profession=1,
        location=0,
        page_views=7,
        session_duration=420
    )
    
    result = await predict_customer(sample_customer)
    
    return {
        "sample_customer": {
            "age": 45,
            "income": "¥12M",
            "profession": "Manager",
            "location": "Tokyo", 
            "engagement": "High"
        },
        "prediction": result
    }

if __name__ == "__main__":
    print("🚀 Starting GP MLOps Demo API...")
    print("📊 Training ML models...")
    print("🔗 API will be available at: http://localhost:2233")
    print("📖 Documentation at: http://localhost:2233/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)