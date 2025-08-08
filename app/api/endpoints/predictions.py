from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import time
import json
from app.core.database import get_db
from app.services.customer_inference import CustomerInferenceService
from app.services.aws_services import SageMakerService
from app.services.snowflake_service import SnowflakeService
from app.core.monitoring import record_model_prediction
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

class BatchPredictionRequest(BaseModel):
    customer_ids: List[int]
    prediction_types: List[str] = ["conversion", "value"]

class BatchPredictionResponse(BaseModel):
    job_id: str
    status: str
    customer_count: int
    estimated_completion_time: str

class ModelTrainingRequest(BaseModel):
    model_type: str  # "conversion" or "value"
    training_data_days: int = 365
    retrain: bool = False

class ModelMetrics(BaseModel):
    model_type: str
    version: str
    accuracy: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    f1_score: Optional[float]
    r2_score: Optional[float]
    mse: Optional[float]
    training_date: str
    feature_importance: Dict[str, float]

@router.post("/batch", response_model=BatchPredictionResponse)
async def batch_predictions(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Run batch predictions for multiple customers"""
    try:
        job_id = f"batch_{int(time.time())}"
        
        # Validate customer IDs exist
        if len(request.customer_ids) > 1000:
            raise HTTPException(
                status_code=400, 
                detail="Batch size cannot exceed 1000 customers"
            )
        
        # Queue batch job
        background_tasks.add_task(
            _run_batch_predictions,
            job_id,
            request.customer_ids,
            request.prediction_types,
            db
        )
        
        # Estimate completion time (roughly 1 second per customer)
        estimated_minutes = len(request.customer_ids) / 60
        
        return BatchPredictionResponse(
            job_id=job_id,
            status="queued",
            customer_count=len(request.customer_ids),
            estimated_completion_time=f"{estimated_minutes:.1f} minutes"
        )
        
    except Exception as e:
        logger.error(f"Batch prediction request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/batch/{job_id}")
async def get_batch_job_status(job_id: str):
    """Get status of batch prediction job"""
    try:
        # In a real implementation, you'd check job status from a queue/database
        # For now, return a mock response
        return {
            "job_id": job_id,
            "status": "processing",  # queued, processing, completed, failed
            "progress": "45%",
            "completed_count": 450,
            "total_count": 1000,
            "results_s3_path": f"s3://ga-technology-mlops/batch_results/{job_id}.parquet"
        }
        
    except Exception as e:
        logger.error(f"Failed to get batch job status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/train-model")
async def train_model(
    request: ModelTrainingRequest,
    background_tasks: BackgroundTasks
):
    """Train or retrain ML models"""
    try:
        # Validate model type
        if request.model_type not in ["conversion", "value", "both"]:
            raise HTTPException(
                status_code=400,
                detail="Model type must be 'conversion', 'value', or 'both'"
            )
        
        # Queue training job
        training_job_id = f"training_{request.model_type}_{int(time.time())}"
        
        background_tasks.add_task(
            _run_model_training,
            training_job_id,
            request.model_type,
            request.training_data_days,
            request.retrain
        )
        
        return {
            "training_job_id": training_job_id,
            "status": "queued",
            "model_type": request.model_type,
            "message": f"Model training queued for {request.model_type}"
        }
        
    except Exception as e:
        logger.error(f"Model training request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model-metrics", response_model=List[ModelMetrics])
async def get_model_metrics():
    """Get current model performance metrics"""
    try:
        # In a real implementation, this would fetch from MLflow or model registry
        metrics = [
            ModelMetrics(
                model_type="conversion",
                version="v1.2",
                accuracy=0.87,
                precision=0.84,
                recall=0.82,
                f1_score=0.83,
                r2_score=None,
                mse=None,
                training_date="2024-01-15T10:30:00Z",
                feature_importance={
                    "engagement_score": 0.25,
                    "income": 0.22,
                    "age": 0.18,
                    "total_interactions": 0.15,
                    "location_encoded": 0.12,
                    "family_size": 0.08
                }
            ),
            ModelMetrics(
                model_type="value",
                version="v1.1",
                accuracy=None,
                precision=None,
                recall=None,
                f1_score=None,
                r2_score=0.73,
                mse=125000.0,
                training_date="2024-01-15T11:00:00Z",
                feature_importance={
                    "income": 0.35,
                    "engagement_score": 0.20,
                    "age": 0.15,
                    "location_encoded": 0.12,
                    "profession_encoded": 0.10,
                    "family_size": 0.08
                }
            )
        ]
        
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get model metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/feature-importance/{model_type}")
async def get_feature_importance(model_type: str):
    """Get feature importance for a specific model"""
    try:
        if model_type not in ["conversion", "value"]:
            raise HTTPException(
                status_code=400,
                detail="Model type must be 'conversion' or 'value'"
            )
        
        # Load current model and return feature importance
        inference_service = CustomerInferenceService()
        
        # In a real implementation, this would load the actual model
        # For now, return mock data
        if model_type == "conversion":
            importance = {
                "engagement_score": 0.25,
                "income": 0.22,
                "age": 0.18,
                "total_interactions": 0.15,
                "location_encoded": 0.12,
                "family_size": 0.08
            }
        else:  # value
            importance = {
                "income": 0.35,
                "engagement_score": 0.20,
                "age": 0.15,
                "location_encoded": 0.12,
                "profession_encoded": 0.10,
                "family_size": 0.08
            }
        
        return {
            "model_type": model_type,
            "feature_importance": importance,
            "version": "v1.2" if model_type == "conversion" else "v1.1"
        }
        
    except Exception as e:
        logger.error(f"Failed to get feature importance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/deploy-to-sagemaker")
async def deploy_model_to_sagemaker(
    model_type: str,
    endpoint_name: Optional[str] = None
):
    """Deploy trained model to SageMaker endpoint"""
    try:
        if model_type not in ["conversion", "value"]:
            raise HTTPException(
                status_code=400,
                detail="Model type must be 'conversion' or 'value'"
            )
        
        sagemaker_service = SageMakerService()
        
        # Generate endpoint name if not provided
        if not endpoint_name:
            endpoint_name = f"ga-{model_type}-model-{int(time.time())}"
        
        # Get model S3 path
        model_s3_path = f"s3://ga-technology-mlops/models/{model_type}_model.pkl"
        
        # Deploy model
        deployed_endpoint = sagemaker_service.deploy_model(
            model_name=f"ga-{model_type}-model",
            model_s3_path=model_s3_path
        )
        
        return {
            "model_type": model_type,
            "endpoint_name": deployed_endpoint,
            "status": "deploying",
            "message": f"Model deployment initiated for {model_type}"
        }
        
    except Exception as e:
        logger.error(f"SageMaker deployment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Background task functions
async def _run_batch_predictions(
    job_id: str, 
    customer_ids: List[int], 
    prediction_types: List[str],
    db: Session
):
    """Background task to run batch predictions"""
    try:
        inference_service = CustomerInferenceService()
        snowflake_service = SnowflakeService()
        
        # Get customer data from Snowflake
        customer_data = snowflake_service.get_customer_data(customer_ids)
        
        results = []
        for _, customer_row in customer_data.iterrows():
            # Create mock customer object
            customer = type('Customer', (), {
                'id': customer_row['customer_id'],
                'age': customer_row.get('age'),
                'income': customer_row.get('income'),
                'profession': customer_row.get('profession'),
                'location': customer_row.get('location'),
                'family_size': customer_row.get('family_size')
            })()
            
            # Get predictions
            prediction = inference_service.predict_customer_value(customer, [])
            results.append(prediction)
        
        # Store results in S3 or database
        logger.info(f"Batch job {job_id} completed with {len(results)} predictions")
        
    except Exception as e:
        logger.error(f"Batch prediction job {job_id} failed: {e}")

async def _run_model_training(
    job_id: str,
    model_type: str,
    training_data_days: int,
    retrain: bool
):
    """Background task to train models"""
    try:
        inference_service = CustomerInferenceService()
        
        # Prepare training data
        training_data = inference_service.prepare_training_data()
        
        # Train models based on type
        if model_type in ["conversion", "both"]:
            conversion_metrics = inference_service.train_conversion_model(training_data)
            logger.info(f"Conversion model training completed: {conversion_metrics}")
        
        if model_type in ["value", "both"]:
            value_metrics = inference_service.train_value_model(training_data)
            logger.info(f"Value model training completed: {value_metrics}")
        
        # Save models to S3
        inference_service.save_models_to_s3()
        
        logger.info(f"Model training job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Model training job {job_id} failed: {e}")