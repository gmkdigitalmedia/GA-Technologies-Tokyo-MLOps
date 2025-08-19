from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import time
import uuid
from app.core.database import get_db
from app.models.property import FloorplanAnalysis
from app.services.floorplan_detection import FloorplanDetectionService
from app.services.aws_services import S3DataService
from app.core.monitoring import record_model_prediction
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

class FloorplanAnalysisResponse(BaseModel):
    analysis_id: int
    property_id: int
    detected_layout: Dict[str, Any]
    room_count: Dict[str, int]
    estimated_flow: Dict[str, Any]
    accessibility_score: float
    family_friendliness: float
    confidence_scores: Dict[str, float]
    
    class Config:
        from_attributes = True

class FloorplanUploadResponse(BaseModel):
    upload_id: str
    s3_path: str
    status: str
    message: str

@router.post("/upload", response_model=FloorplanUploadResponse)
async def upload_floorplan_image(
    file: UploadFile = File(...),
    property_id: Optional[int] = None
):
    """Upload floorplan image to S3"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Generate unique filename
        upload_id = str(uuid.uuid4())
        file_extension = file.filename.split('.')[-1] if file.filename else 'jpg'
        s3_key = f"floorplans/{upload_id}.{file_extension}"
        
        # Upload to S3
        s3_service = S3DataService()
        
        # Save file temporarily
        temp_path = f"/tmp/{upload_id}.{file_extension}"
        with open(temp_path, "wb") as temp_file:
            content = await file.read()
            temp_file.write(content)
        
        # Upload to S3
        s3_path = s3_service.upload_model_artifact(temp_path, s3_key)
        
        return FloorplanUploadResponse(
            upload_id=upload_id,
            s3_path=s3_path,
            status="uploaded",
            message="Floorplan image uploaded successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to upload floorplan image: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.post("/{property_id}/analyze", response_model=FloorplanAnalysisResponse)
async def analyze_floorplan(
    property_id: int,
    image_path: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Analyze floorplan layout and features"""
    start_time = time.time()
    
    try:
        # Initialize floorplan detection service
        detection_service = FloorplanDetectionService()
        
        # Perform analysis
        analysis_result = detection_service.analyze_floorplan(image_path)
        
        # Store analysis in database
        db_analysis = FloorplanAnalysis(
            property_id=property_id,
            image_path=image_path,
            detected_layout=analysis_result["detected_layout"],
            room_count=analysis_result["room_count"],
            estimated_flow=analysis_result["estimated_flow"],
            accessibility_score=analysis_result["accessibility_score"],
            family_friendliness=analysis_result["family_friendliness"],
            model_version="v1.0",
            confidence_scores=analysis_result["confidence_scores"]
        )
        
        db.add(db_analysis)
        db.commit()
        db.refresh(db_analysis)
        
        # Record metrics
        duration = time.time() - start_time
        record_model_prediction("floorplan_detection", duration, True)
        
        return FloorplanAnalysisResponse(**db_analysis.__dict__)
        
    except Exception as e:
        duration = time.time() - start_time
        record_model_prediction("floorplan_detection", duration, False)
        logger.error(f"Floorplan analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.get("/{property_id}/analysis", response_model=List[FloorplanAnalysisResponse])
async def get_floorplan_analyses(property_id: int, db: Session = Depends(get_db)):
    """Get all floorplan analyses for a property"""
    analyses = db.query(FloorplanAnalysis).filter(
        FloorplanAnalysis.property_id == property_id
    ).all()
    
    if not analyses:
        raise HTTPException(status_code=404, detail="No analyses found for this property")
    
    return analyses

@router.post("/batch-analyze")
async def batch_analyze_floorplans(
    property_ids: List[int],
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Batch analyze multiple floorplans"""
    try:
        detection_service = FloorplanDetectionService()
        
        # Queue batch analysis as background task
        background_tasks.add_task(
            detection_service.batch_analyze,
            property_ids,
            db
        )
        
        return {
            "message": f"Batch analysis queued for {len(property_ids)} properties",
            "property_ids": property_ids,
            "status": "queued"
        }
        
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

@router.get("/room-types")
async def get_supported_room_types():
    """Get list of supported room types for detection"""
    return {
        "supported_room_types": [
            "bedroom", "living_room", "kitchen", "bathroom", "dining_room",
            "hallway", "closet", "balcony", "study", "storage", "utility"
        ],
        "layout_features": [
            "traffic_flow", "natural_light", "room_connections",
            "accessibility_features", "storage_space"
        ]
    }

@router.get("/analysis/{analysis_id}", response_model=FloorplanAnalysisResponse)
async def get_analysis_details(analysis_id: int, db: Session = Depends(get_db)):
    """Get detailed analysis results"""
    analysis = db.query(FloorplanAnalysis).filter(
        FloorplanAnalysis.id == analysis_id
    ).first()
    
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return analysis