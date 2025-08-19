from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.database import get_db
import redis
from app.core.config import settings

router = APIRouter()

@router.get("/")
async def health_check():
    return {"status": "healthy", "service": "GP MLOps Platform"}

@router.get("/detailed")
async def detailed_health_check(db: Session = Depends(get_db)):
    health_status = {
        "status": "healthy",
        "services": {},
        "timestamp": None
    }
    
    # Check database
    try:
        db.execute("SELECT 1")
        health_status["services"]["database"] = {"status": "healthy"}
    except Exception as e:
        health_status["services"]["database"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "degraded"
    
    # Check Redis
    try:
        r = redis.from_url(settings.REDIS_URL)
        r.ping()
        health_status["services"]["redis"] = {"status": "healthy"}
    except Exception as e:
        health_status["services"]["redis"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "degraded"
    
    return health_status