from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import Response
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime, timedelta
import logging

from app.core.database import get_db
from app.models.advertising import (
    AdCampaign, AdCreative, AdServing, AdInteraction, AudienceProfile, ABTestResults,
    AdCampaignStatus, AdVariantType, AudienceSegment
)
from app.models.customer import Customer
from app.services.ad_serving import AdServingEngine, ABTestAnalyzer
from app.core.monitoring import record_model_prediction
import time

logger = logging.getLogger(__name__)
router = APIRouter()

# Request/Response Models
class AdRequest(BaseModel):
    customer_id: int
    page_url: Optional[str] = None
    device_type: str = "desktop"
    user_agent: Optional[str] = None

class AdResponse(BaseModel):
    ad_id: str
    campaign_id: int
    creative_id: int
    serving_id: int
    variant_type: str
    content: Dict[str, Any]
    tracking: Dict[str, str]
    metadata: Dict[str, Any]

class CampaignCreate(BaseModel):
    name: str
    description: Optional[str] = None
    target_audience: AudienceSegment
    min_age: int = 40
    max_age: int = 50
    min_income: int = 10000000
    target_locations: List[str] = ["tokyo", "osaka", "nagoya"]
    daily_budget: float
    total_budget: float
    start_date: datetime
    end_date: Optional[datetime] = None
    ab_test_enabled: bool = True
    traffic_split: Dict[str, int] = {"control": 50, "variant_a": 50}

class CreativeCreate(BaseModel):
    campaign_id: int
    variant_type: AdVariantType
    headline: str
    description: str
    call_to_action: str
    image_url: Optional[str] = None
    landing_page_url: str
    creative_type: str = "banner"
    dimensions: Dict[str, int] = {"width": 728, "height": 90}

class InteractionTrack(BaseModel):
    interaction_type: str  # impression, click, conversion
    value: Optional[float] = None
    page_url: Optional[str] = None
    referrer_url: Optional[str] = None
    session_id: Optional[str] = None

class ABTestAnalysisResponse(BaseModel):
    campaign_id: int
    test_duration_days: int
    variant_results: Dict[str, Any]
    statistical_results: Dict[str, Any]
    winner: Dict[str, Any]
    recommendation: str

# Initialize services
ad_serving_engine = AdServingEngine()
ab_test_analyzer = ABTestAnalyzer()

@router.post("/serve", response_model=AdResponse)
async def serve_ad(
    ad_request: AdRequest,
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Serve targeted ad to customer based on GP MLOps' criteria:
    - Age: 40-50 years old
    - Income: >10M yen annually  
    - Target: Salaryman professionals
    - A/B testing enabled
    """
    try:
        start_time = time.time()
        
        # Get customer data
        customer = db.query(Customer).filter(Customer.id == ad_request.customer_id).first()
        if not customer:
            raise HTTPException(status_code=404, detail="Customer not found")
        
        # Build customer data dictionary
        customer_data = {
            'id': customer.id,
            'age': customer.age,
            'income': customer.income,
            'profession': customer.profession,
            'location': customer.location,
            'family_size': customer.family_size
        }
        
        # Build request context
        request_context = {
            'ip_address': request.client.host if request.client else "unknown",
            'user_agent': ad_request.user_agent or request.headers.get("user-agent", ""),
            'device_type': ad_request.device_type,
            'page_url': ad_request.page_url,
            'timestamp': datetime.now()
        }
        
        # Serve ad using targeting engine
        ad_response = ad_serving_engine.serve_ad(customer_data, request_context, db)
        
        if not ad_response:
            raise HTTPException(
                status_code=204, 
                detail="No suitable ads available for this customer"
            )
        
        # Record successful serving
        duration = time.time() - start_time
        record_model_prediction("ad_serving_request", duration, True)
        
        return AdResponse(**ad_response)
        
    except HTTPException:
        raise
    except Exception as e:
        duration = time.time() - start_time
        record_model_prediction("ad_serving_request", duration, False)
        logger.error(f"Ad serving failed: {e}")
        raise HTTPException(status_code=500, detail="Ad serving failed")

@router.post("/track/{serving_id}")
async def track_interaction(
    serving_id: int,
    interaction: InteractionTrack,
    db: Session = Depends(get_db)
):
    """Track ad interactions for performance measurement and attribution"""
    try:
        interaction_context = {
            'value': interaction.value,
            'page_url': interaction.page_url,
            'referrer_url': interaction.referrer_url,
            'session_id': interaction.session_id
        }
        
        success = ad_serving_engine.track_interaction(
            serving_id, 
            interaction.interaction_type, 
            interaction_context, 
            db
        )
        
        if success:
            return {"status": "tracked", "serving_id": serving_id, "type": interaction.interaction_type}
        else:
            raise HTTPException(status_code=400, detail="Interaction tracking failed")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Interaction tracking failed: {e}")
        raise HTTPException(status_code=500, detail="Tracking failed")

@router.get("/campaigns", response_model=List[Dict[str, Any]])
async def get_campaigns(
    status: Optional[AdCampaignStatus] = None,
    audience: Optional[AudienceSegment] = None,
    limit: int = Query(default=50, le=100),
    db: Session = Depends(get_db)
):
    """Get advertising campaigns with optional filters"""
    
    query = db.query(AdCampaign)
    
    if status:
        query = query.filter(AdCampaign.status == status)
    if audience:
        query = query.filter(AdCampaign.target_audience == audience)
    
    campaigns = query.limit(limit).all()
    
    # Build response with performance metrics
    campaign_data = []
    for campaign in campaigns:
        ctr = campaign.clicks / max(1, campaign.impressions)
        cvr = campaign.conversions / max(1, campaign.clicks)
        
        campaign_data.append({
            "id": campaign.id,
            "name": campaign.name,
            "status": campaign.status.value,
            "target_audience": campaign.target_audience.value,
            "daily_budget": campaign.daily_budget,
            "total_spend": campaign.spend,
            "impressions": campaign.impressions,
            "clicks": campaign.clicks,
            "conversions": campaign.conversions,
            "ctr": round(ctr * 100, 2),
            "cvr": round(cvr * 100, 2),
            "start_date": campaign.start_date,
            "end_date": campaign.end_date,
            "ab_test_enabled": campaign.ab_test_enabled
        })
    
    return campaign_data

@router.post("/campaigns", response_model=Dict[str, Any])
async def create_campaign(
    campaign: CampaignCreate,
    db: Session = Depends(get_db)
):
    """Create new advertising campaign for GP MLOps' target audience"""
    try:
        # Validate targeting criteria matches GA's requirements
        if campaign.min_age < 35 or campaign.max_age > 55:
            raise HTTPException(
                status_code=400, 
                detail="Age targeting must focus on 40-50 demographic (35-55 allowed)"
            )
        
        if campaign.min_income < 8000000:  # 8M yen minimum
            raise HTTPException(
                status_code=400,
                detail="Income targeting must be at least 8M yen for GP MLOps' market"
            )
        
        # Create campaign
        db_campaign = AdCampaign(
            name=campaign.name,
            description=campaign.description,
            target_audience=campaign.target_audience,
            min_age=campaign.min_age,
            max_age=campaign.max_age,
            min_income=campaign.min_income,
            target_locations=campaign.target_locations,
            daily_budget=campaign.daily_budget,
            total_budget=campaign.total_budget,
            start_date=campaign.start_date,
            end_date=campaign.end_date,
            ab_test_enabled=campaign.ab_test_enabled,
            traffic_split=campaign.traffic_split,
            status=AdCampaignStatus.DRAFT
        )
        
        db.add(db_campaign)
        db.commit()
        db.refresh(db_campaign)
        
        return {
            "campaign_id": db_campaign.id,
            "name": db_campaign.name,
            "status": db_campaign.status.value,
            "message": "Campaign created successfully. Add creatives and activate to start serving."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Campaign creation failed: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Campaign creation failed")

@router.post("/creatives", response_model=Dict[str, Any])
async def create_creative(
    creative: CreativeCreate,
    db: Session = Depends(get_db)
):
    """Create ad creative for A/B testing"""
    try:
        # Verify campaign exists
        campaign = db.query(AdCampaign).filter(AdCampaign.id == creative.campaign_id).first()
        if not campaign:
            raise HTTPException(status_code=404, detail="Campaign not found")
        
        # Create creative
        db_creative = AdCreative(
            campaign_id=creative.campaign_id,
            variant_type=creative.variant_type,
            headline=creative.headline,
            description=creative.description,
            call_to_action=creative.call_to_action,
            image_url=creative.image_url,
            landing_page_url=creative.landing_page_url,
            creative_type=creative.creative_type,
            dimensions=creative.dimensions
        )
        
        db.add(db_creative)
        db.commit()
        db.refresh(db_creative)
        
        return {
            "creative_id": db_creative.id,
            "campaign_id": db_creative.campaign_id,
            "variant_type": db_creative.variant_type.value,
            "headline": db_creative.headline,
            "message": "Creative created successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Creative creation failed: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Creative creation failed")

@router.get("/campaigns/{campaign_id}/ab-test", response_model=ABTestAnalysisResponse)
async def analyze_ab_test(
    campaign_id: int,
    db: Session = Depends(get_db)
):
    """Analyze A/B test results for campaign performance"""
    try:
        analysis_results = ab_test_analyzer.analyze_campaign_results(campaign_id, db)
        
        if "error" in analysis_results:
            raise HTTPException(status_code=400, detail=analysis_results["error"])
        
        return ABTestAnalysisResponse(**analysis_results)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"A/B test analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Analysis failed")

@router.put("/campaigns/{campaign_id}/status")
async def update_campaign_status(
    campaign_id: int,
    status: AdCampaignStatus,
    db: Session = Depends(get_db)
):
    """Update campaign status (draft, active, paused, completed)"""
    try:
        campaign = db.query(AdCampaign).filter(AdCampaign.id == campaign_id).first()
        if not campaign:
            raise HTTPException(status_code=404, detail="Campaign not found")
        
        # Validate status transition
        if status == AdCampaignStatus.ACTIVE:
            # Check if campaign has creatives
            creative_count = db.query(AdCreative).filter(
                AdCreative.campaign_id == campaign_id,
                AdCreative.is_active == True
            ).count()
            
            if creative_count == 0:
                raise HTTPException(
                    status_code=400, 
                    detail="Cannot activate campaign without active creatives"
                )
        
        campaign.status = status
        db.commit()
        
        return {
            "campaign_id": campaign_id,
            "status": status.value,
            "message": f"Campaign status updated to {status.value}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Status update failed: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Status update failed")

@router.get("/performance/summary")
async def get_performance_summary(
    days: int = Query(default=7, ge=1, le=90),
    campaign_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Get advertising performance summary"""
    try:
        date_from = datetime.now() - timedelta(days=days)
        
        # Build query
        query = db.query(AdCampaign)
        if campaign_id:
            query = query.filter(AdCampaign.id == campaign_id)
        
        campaigns = query.filter(AdCampaign.created_at >= date_from).all()
        
        # Aggregate metrics
        total_impressions = sum(c.impressions for c in campaigns)
        total_clicks = sum(c.clicks for c in campaigns)
        total_conversions = sum(c.conversions for c in campaigns)
        total_spend = sum(c.spend for c in campaigns)
        
        overall_ctr = total_clicks / max(1, total_impressions)
        overall_cvr = total_conversions / max(1, total_clicks)
        avg_cpc = total_spend / max(1, total_clicks)
        avg_cpa = total_spend / max(1, total_conversions)
        
        # Performance by audience segment
        segment_performance = {}
        for segment in AudienceSegment:
            segment_campaigns = [c for c in campaigns if c.target_audience == segment]
            if segment_campaigns:
                seg_impressions = sum(c.impressions for c in segment_campaigns)
                seg_clicks = sum(c.clicks for c in segment_campaigns)
                seg_conversions = sum(c.conversions for c in segment_campaigns)
                
                segment_performance[segment.value] = {
                    "campaigns": len(segment_campaigns),
                    "impressions": seg_impressions,
                    "clicks": seg_clicks,
                    "conversions": seg_conversions,
                    "ctr": seg_clicks / max(1, seg_impressions),
                    "cvr": seg_conversions / max(1, seg_clicks)
                }
        
        return {
            "period_days": days,
            "summary": {
                "total_campaigns": len(campaigns),
                "total_impressions": total_impressions,
                "total_clicks": total_clicks,
                "total_conversions": total_conversions,
                "total_spend": round(total_spend, 2),
                "overall_ctr": round(overall_ctr * 100, 3),
                "overall_cvr": round(overall_cvr * 100, 3),
                "avg_cpc": round(avg_cpc, 2),
                "avg_cpa": round(avg_cpa, 2)
            },
            "by_audience_segment": segment_performance
        }
        
    except Exception as e:
        logger.error(f"Performance summary failed: {e}")
        raise HTTPException(status_code=500, detail="Performance summary failed")

@router.get("/audience/segments")
async def get_audience_segments(db: Session = Depends(get_db)):
    """Get available audience segments for targeting"""
    
    segments = []
    for segment in AudienceSegment:
        # Get segment performance from database
        profile = db.query(AudienceProfile).filter(
            AudienceProfile.segment_name == segment
        ).first()
        
        if profile:
            segments.append({
                "segment": segment.value,
                "name": segment.value.replace('_', ' ').title(),
                "criteria": {
                    "age_range": f"{profile.age_min}-{profile.age_max}",
                    "min_income": profile.income_min,
                    "locations": profile.locations,
                    "professions": profile.professions
                },
                "estimated_size": profile.estimated_size,
                "avg_conversion_rate": round(profile.average_conversion_rate * 100, 2),
                "performance": {
                    "total_served": profile.total_served_ads,
                    "avg_ctr": round(profile.average_ctr * 100, 3),
                    "avg_cvr": round(profile.average_cvr * 100, 3)
                }
            })
        else:
            # Default segment info for GP MLOps' primary target
            if segment == AudienceSegment.HIGH_INCOME_SALARYMAN:
                segments.append({
                    "segment": segment.value,
                    "name": "High Income Salaryman (Primary Target)",
                    "criteria": {
                        "age_range": "40-50",
                        "min_income": 10000000,
                        "locations": ["tokyo", "osaka", "nagoya", "yokohama"],
                        "professions": ["engineer", "manager", "executive", "consultant"]
                    },
                    "estimated_size": 50000,
                    "avg_conversion_rate": 2.5,
                    "performance": {
                        "total_served": 0,
                        "avg_ctr": 0,
                        "avg_cvr": 0
                    }
                })
    
    return {"audience_segments": segments}

# Pixel tracking endpoint for conversion attribution
@router.get("/pixel/{serving_id}")
async def conversion_pixel(serving_id: int, db: Session = Depends(get_db)):
    """1x1 pixel for conversion tracking"""
    try:
        # Track conversion
        success = ad_serving_engine.track_interaction(
            serving_id, "conversion", {}, db
        )
        
        # Return 1x1 transparent pixel
        pixel_data = b'\x47\x49\x46\x38\x39\x61\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff\x21\xf9\x04\x01\x00\x00\x00\x00\x2c\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02\x04\x01\x00\x3b'
        
        return Response(content=pixel_data, media_type="image/gif")
        
    except Exception as e:
        logger.error(f"Conversion pixel tracking failed: {e}")
        return Response(content=pixel_data, media_type="image/gif")