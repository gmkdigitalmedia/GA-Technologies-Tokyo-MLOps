from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import Response
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime
import logging
import uuid

from app.core.database import get_db
from app.services.conversion_api import ConversionTracker
from app.models.customer import Customer
from app.core.monitoring import record_model_prediction
import time

logger = logging.getLogger(__name__)
router = APIRouter()

class ConversionEvent(BaseModel):
    event_type: str  # property_inquiry, property_purchase, mortgage_application, etc.
    customer_id: int
    event_time: Optional[datetime] = None
    value: Optional[float] = None
    currency: str = "JPY"
    
    # Attribution data
    google_click_id: Optional[str] = None  # gclid
    facebook_click_id: Optional[str] = None  # fbclid
    twitter_click_id: Optional[str] = None  # twclid
    
    # User data for enhanced matching (hashed on client side)
    email_hash: Optional[str] = None
    phone_hash: Optional[str] = None
    
    # Event properties
    properties: Dict[str, Any] = {}
    user_properties: Dict[str, Any] = {}
    
    # Request context
    page_url: Optional[str] = None
    referrer_url: Optional[str] = None
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None

class ConversionResponse(BaseModel):
    conversion_id: int
    event_id: str
    attributed: bool
    attribution: Optional[Dict[str, Any]]
    capi_results: Dict[str, Any]
    status: str

class FunnelAnalysisResponse(BaseModel):
    campaign_id: int
    date_range: str
    funnel_metrics: Dict[str, Any]
    conversion_paths: Dict[str, Any]
    total_conversion_value: float

# Initialize service
conversion_tracker = ConversionTracker()

@router.post("/track", response_model=ConversionResponse)
async def track_conversion_event(
    conversion_event: ConversionEvent,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Track conversion event and send to advertising platforms
    
    This endpoint receives conversion events from GA Technologies' website/app
    and handles attribution to ad campaigns plus sending data to advertising
    platforms via their Conversion APIs.
    """
    try:
        start_time = time.time()
        
        # Validate customer exists
        customer = db.query(Customer).filter(Customer.id == conversion_event.customer_id).first()
        if not customer:
            raise HTTPException(status_code=404, detail="Customer not found")
        
        # Generate unique event ID if not provided
        event_id = str(uuid.uuid4())
        
        # Prepare conversion data
        conversion_data = {
            'event_id': event_id,
            'event_type': conversion_event.event_type,
            'event_time': conversion_event.event_time or datetime.now(),
            'customer_id': conversion_event.customer_id,
            'value': conversion_event.value or 0,
            'currency': conversion_event.currency,
            
            # Attribution data
            'google_click_id': conversion_event.google_click_id,
            'facebook_click_id': conversion_event.facebook_click_id, 
            'twitter_click_id': conversion_event.twitter_click_id,
            
            # User data (already hashed)
            'email': conversion_event.email_hash,
            'phone': conversion_event.phone_hash,
            
            # Context
            'page_url': conversion_event.page_url,
            'referrer_url': conversion_event.referrer_url,
            'user_agent': conversion_event.user_agent,
            'ip_address': conversion_event.ip_address,
            
            # Properties
            'properties': conversion_event.properties,
            'user_properties': conversion_event.user_properties
        }
        
        # Track conversion
        tracking_result = conversion_tracker.track_conversion(conversion_data, db)
        
        # Record metrics
        duration = time.time() - start_time
        record_model_prediction("conversion_tracking", duration, 
                               tracking_result['status'] == 'success')
        
        if tracking_result['status'] == 'success':
            logger.info(f"Conversion tracked successfully: {event_id} for customer {conversion_event.customer_id}")
            return ConversionResponse(**tracking_result)
        else:
            logger.error(f"Conversion tracking failed: {tracking_result.get('error')}")
            raise HTTPException(status_code=500, detail=tracking_result.get('error'))
        
    except HTTPException:
        raise
    except Exception as e:
        duration = time.time() - start_time
        record_model_prediction("conversion_tracking", duration, False)
        logger.error(f"Conversion tracking failed: {e}")
        raise HTTPException(status_code=500, detail="Conversion tracking failed")

@router.get("/funnel/{campaign_id}", response_model=FunnelAnalysisResponse)
async def get_conversion_funnel(
    campaign_id: int,
    days: int = 30,
    db: Session = Depends(get_db)
):
    """Get conversion funnel analysis for a specific campaign"""
    try:
        funnel_data = conversion_tracker.get_conversion_funnel(campaign_id, days, db)
        
        if 'error' in funnel_data:
            raise HTTPException(status_code=400, detail=funnel_data['error'])
        
        return FunnelAnalysisResponse(**funnel_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Funnel analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Funnel analysis failed")

@router.get("/attribution/{customer_id}")
async def get_customer_attribution_history(
    customer_id: int,
    days: int = 30,
    db: Session = Depends(get_db)
):
    """Get attribution history for a specific customer"""
    try:
        from datetime import timedelta
        date_from = datetime.now() - timedelta(days=days)
        
        # Get customer's conversions with attribution
        conversions = db.query(ConversionAPI).filter(
            and_(
                ConversionAPI.customer_id == customer_id,
                ConversionAPI.event_time >= date_from
            )
        ).all()
        
        # Get customer's ad interactions
        interactions = db.query(AdInteraction).join(AdServing).filter(
            and_(
                AdServing.customer_id == customer_id,
                AdInteraction.interaction_time >= date_from
            )
        ).all()
        
        # Build attribution timeline
        timeline = []
        
        # Add interactions
        for interaction in interactions:
            timeline.append({
                'timestamp': interaction.interaction_time,
                'type': 'interaction',
                'action': interaction.interaction_type,
                'campaign_id': interaction.campaign_id,
                'creative_id': interaction.creative_id
            })
        
        # Add conversions
        for conversion in conversions:
            timeline.append({
                'timestamp': conversion.event_time,
                'type': 'conversion',
                'event_type': conversion.event_type,
                'value': conversion.conversion_value,
                'attributed_campaign_id': conversion.attributed_campaign_id,
                'attributed_creative_id': conversion.attributed_creative_id
            })
        
        # Sort by timestamp
        timeline.sort(key=lambda x: x['timestamp'])
        
        return {
            'customer_id': customer_id,
            'date_range': f'{days} days',
            'total_interactions': len(interactions),
            'total_conversions': len(conversions),
            'total_conversion_value': sum(c.conversion_value or 0 for c in conversions),
            'attribution_timeline': timeline
        }
        
    except Exception as e:
        logger.error(f"Attribution history failed: {e}")
        raise HTTPException(status_code=500, detail="Attribution history failed")

@router.post("/test-event")
async def send_test_conversion_event(
    platform: str = "facebook",  # facebook, google, twitter
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    """Send test conversion event to verify CAPI integration"""
    try:
        test_conversion_data = {
            'event_id': f'test_{int(time.time())}',
            'event_type': 'property_inquiry',
            'event_time': datetime.now(),
            'customer_id': 99999,  # Test customer ID
            'value': 100000,  # 100K yen test value
            'currency': 'JPY',
            'page_url': 'https://renosy.com/test',
            'user_agent': 'GA-Technologies-Test-Agent/1.0'
        }
        
        # Send to specified platform
        capi_results = conversion_tracker.capi_client.send_conversion_event(
            test_conversion_data, 
            platforms=[platform]
        )
        
        return {
            'test_event': True,
            'platform': platform,
            'results': capi_results,
            'message': f'Test event sent to {platform}'
        }
        
    except Exception as e:
        logger.error(f"Test event failed: {e}")
        raise HTTPException(status_code=500, detail=f"Test event failed: {str(e)}")

@router.get("/platforms/status")
async def get_platform_status():
    """Get status of connected advertising platforms"""
    
    platforms_status = {}
    
    # Check Facebook
    facebook_configured = bool(
        conversion_tracker.capi_client.platforms['facebook']['access_token'] and
        conversion_tracker.capi_client.platforms['facebook']['pixel_id']
    )
    platforms_status['facebook'] = {
        'configured': facebook_configured,
        'status': 'ready' if facebook_configured else 'not_configured',
        'features': ['conversion_tracking', 'attribution', 'lookalike_audiences']
    }
    
    # Check Google Ads
    google_configured = bool(
        conversion_tracker.capi_client.platforms['google']['conversion_id']
    )
    platforms_status['google'] = {
        'configured': google_configured,
        'status': 'ready' if google_configured else 'not_configured', 
        'features': ['conversion_tracking', 'attribution', 'enhanced_conversions']
    }
    
    # Check Twitter
    twitter_configured = bool(
        conversion_tracker.capi_client.platforms['twitter']['api_key']
    )
    platforms_status['twitter'] = {
        'configured': twitter_configured,
        'status': 'ready' if twitter_configured else 'not_configured',
        'features': ['conversion_tracking', 'attribution']
    }
    
    return {
        'platforms': platforms_status,
        'ga_technology_integration': {
            'primary_platforms': ['facebook', 'google'],
            'target_audience': 'high_income_salaryman_40_50',
            'conversion_events': [
                'property_inquiry', 'property_purchase', 'mortgage_application', 
                'consultation_booking', 'signup'
            ]
        }
    }

# JavaScript pixel endpoint for client-side tracking
@router.get("/pixel.js")
async def conversion_tracking_pixel():
    """Return JavaScript pixel for client-side conversion tracking"""
    
    js_pixel = """
// GA Technologies Conversion Tracking Pixel
(function() {
    var ga_tech = window.ga_tech || {};
    
    ga_tech.track = function(eventType, eventData) {
        eventData = eventData || {};
        
        // Get URL parameters for attribution
        var urlParams = new URLSearchParams(window.location.search);
        var conversionData = {
            event_type: eventType,
            event_time: new Date().toISOString(),
            value: eventData.value || null,
            properties: eventData.properties || {},
            
            // Attribution parameters
            google_click_id: urlParams.get('gclid'),
            facebook_click_id: urlParams.get('fbclid'),
            twitter_click_id: urlParams.get('twclid'),
            
            // Page context
            page_url: window.location.href,
            referrer_url: document.referrer,
            user_agent: navigator.userAgent
        };
        
        // Add customer ID if available
        if (eventData.customer_id) {
            conversionData.customer_id = eventData.customer_id;
        } else if (window.ga_tech_customer_id) {
            conversionData.customer_id = window.ga_tech_customer_id;
        }
        
        // Send to conversion API
        fetch('/api/v1/capi/track', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(conversionData)
        }).catch(function(error) {
            console.warn('GA Technologies conversion tracking failed:', error);
        });
    };
    
    // Auto-track page views for property pages
    if (window.location.pathname.includes('/property/')) {
        ga_tech.track('property_view', {
            properties: {
                property_id: window.location.pathname.split('/').pop()
            }
        });
    }
    
    window.ga_tech = ga_tech;
})();
"""
    
    return Response(content=js_pixel, media_type="application/javascript")

# Import required models
from app.models.advertising import ConversionAPI, AdServing, AdInteraction
from sqlalchemy import and_