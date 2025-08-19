from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel
from app.core.database import get_db
from app.models.customer import Customer, CustomerInteraction, CustomerPrediction
from app.services.customer_inference import CustomerInferenceService
from app.core.monitoring import record_model_prediction
import time

router = APIRouter()

class CustomerCreate(BaseModel):
    external_id: str
    age: int
    income: float
    profession: str
    location: str
    family_size: int
    property_preferences: dict

class CustomerResponse(BaseModel):
    id: int
    external_id: str
    age: int
    income: float
    profession: str
    location: str
    family_size: int
    conversion_score: Optional[float]
    lifetime_value: Optional[float]
    
    class Config:
        from_attributes = True

class InteractionCreate(BaseModel):
    customer_id: int
    interaction_type: str
    property_id: Optional[str] = None
    interaction_data: dict

class PredictionResponse(BaseModel):
    customer_id: int
    conversion_probability: float
    estimated_lifetime_value: float
    confidence_score: float
    recommended_properties: List[int]

@router.post("/", response_model=CustomerResponse)
async def create_customer(customer: CustomerCreate, db: Session = Depends(get_db)):
    db_customer = Customer(**customer.dict())
    db.add(db_customer)
    db.commit()
    db.refresh(db_customer)
    return db_customer

@router.get("/{customer_id}", response_model=CustomerResponse)
async def get_customer(customer_id: int, db: Session = Depends(get_db)):
    customer = db.query(Customer).filter(Customer.id == customer_id).first()
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    return customer

@router.post("/{customer_id}/interactions")
async def log_interaction(
    customer_id: int, 
    interaction: InteractionCreate, 
    db: Session = Depends(get_db)
):
    db_interaction = CustomerInteraction(**interaction.dict())
    db.add(db_interaction)
    db.commit()
    return {"message": "Interaction logged successfully"}

@router.get("/{customer_id}/predict", response_model=PredictionResponse)
async def predict_customer_value(
    customer_id: int, 
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = None
):
    start_time = time.time()
    
    try:
        customer = db.query(Customer).filter(Customer.id == customer_id).first()
        if not customer:
            raise HTTPException(status_code=404, detail="Customer not found")
        
        # Get customer interactions for context
        interactions = db.query(CustomerInteraction).filter(
            CustomerInteraction.customer_id == customer_id
        ).limit(100).all()
        
        # Initialize inference service
        inference_service = CustomerInferenceService()
        
        # Generate predictions
        prediction = inference_service.predict_customer_value(customer, interactions)
        
        # Store prediction in database
        db_prediction = CustomerPrediction(
            customer_id=customer_id,
            prediction_type="comprehensive",
            prediction_value=prediction["estimated_lifetime_value"],
            confidence_score=prediction["confidence_score"],
            model_version="v1.0",
            features_used=prediction.get("features", {})
        )
        db.add(db_prediction)
        db.commit()
        
        # Update customer record with latest predictions
        customer.conversion_score = prediction["conversion_probability"]
        customer.lifetime_value = prediction["estimated_lifetime_value"]
        db.commit()
        
        # Record metrics
        duration = time.time() - start_time
        record_model_prediction("customer_value", duration, True)
        
        return PredictionResponse(**prediction)
        
    except Exception as e:
        duration = time.time() - start_time
        record_model_prediction("customer_value", duration, False)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.get("/segment/{segment_type}")
async def get_customer_segment(segment_type: str, db: Session = Depends(get_db)):
    """Get customers by segment (high_value, high_conversion, etc.)"""
    if segment_type == "high_value":
        customers = db.query(Customer).filter(Customer.lifetime_value > 500000).all()
    elif segment_type == "high_conversion":
        customers = db.query(Customer).filter(Customer.conversion_score > 0.7).all()
    else:
        raise HTTPException(status_code=400, detail="Invalid segment type")
    
    return {"segment": segment_type, "count": len(customers), "customers": customers}