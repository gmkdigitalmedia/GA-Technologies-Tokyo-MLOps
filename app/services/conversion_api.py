import requests
import json
import hashlib
import hmac
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import logging
from sqlalchemy.orm import Session
from app.models.advertising import ConversionAPI, AdServing, AdInteraction
from app.core.config import settings

logger = logging.getLogger(__name__)

class ConversionAPIClient:
    """
    Conversion API (CAPI) client for sending server-side conversion events
    to advertising platforms (Facebook, Google, etc.)
    """
    
    def __init__(self):
        # Platform configurations
        self.platforms = {
            'facebook': {
                'url': 'https://graph.facebook.com/v18.0/{pixel_id}/events',
                'access_token': settings.FACEBOOK_ACCESS_TOKEN if hasattr(settings, 'FACEBOOK_ACCESS_TOKEN') else None,
                'pixel_id': settings.FACEBOOK_PIXEL_ID if hasattr(settings, 'FACEBOOK_PIXEL_ID') else None
            },
            'google': {
                'url': 'https://www.googleadservices.com/pagead/conversion/{conversion_id}/',
                'conversion_id': settings.GOOGLE_CONVERSION_ID if hasattr(settings, 'GOOGLE_CONVERSION_ID') else None,
                'conversion_label': settings.GOOGLE_CONVERSION_LABEL if hasattr(settings, 'GOOGLE_CONVERSION_LABEL') else None
            },
            'twitter': {
                'url': 'https://ads-api.twitter.com/12/measurement/conversions',
                'api_key': settings.TWITTER_API_KEY if hasattr(settings, 'TWITTER_API_KEY') else None
            }
        }
    
    def send_conversion_event(self, conversion_data: Dict[str, Any], 
                            platforms: List[str] = None) -> Dict[str, Any]:
        """
        Send conversion event to specified advertising platforms
        
        Args:
            conversion_data: Conversion event data
            platforms: List of platforms to send to (default: all configured)
            
        Returns:
            Results from each platform
        """
        if platforms is None:
            platforms = ['facebook', 'google']  # GP MLOps' primary platforms
        
        results = {}
        
        for platform in platforms:
            try:
                if platform == 'facebook':
                    result = self._send_facebook_event(conversion_data)
                elif platform == 'google':
                    result = self._send_google_event(conversion_data)
                elif platform == 'twitter':
                    result = self._send_twitter_event(conversion_data)
                else:
                    result = {'success': False, 'error': f'Unsupported platform: {platform}'}
                
                results[platform] = result
                
            except Exception as e:
                logger.error(f"Failed to send conversion to {platform}: {e}")
                results[platform] = {'success': False, 'error': str(e)}
        
        return results
    
    def _send_facebook_event(self, conversion_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send conversion event to Facebook Conversions API"""
        
        if not self.platforms['facebook']['access_token'] or not self.platforms['facebook']['pixel_id']:
            return {'success': False, 'error': 'Facebook credentials not configured'}
        
        # Build Facebook event data
        event_data = {
            'data': [{
                'event_name': self._map_event_name(conversion_data['event_type']),
                'event_time': int(conversion_data['event_time'].timestamp()),
                'action_source': 'website',
                'event_source_url': conversion_data.get('page_url', ''),
                'user_data': {
                    'client_ip_address': conversion_data.get('ip_address'),
                    'client_user_agent': conversion_data.get('user_agent'),
                    'fbc': conversion_data.get('facebook_click_id'),  # Facebook click ID
                    'fbp': conversion_data.get('facebook_browser_id')  # Facebook browser ID
                },
                'custom_data': {
                    'currency': 'JPY',
                    'value': conversion_data.get('value', 0),
                    'content_category': 'real_estate',
                    'content_type': 'property_inquiry'
                },
                'event_id': conversion_data.get('event_id')  # For deduplication
            }],
            'access_token': self.platforms['facebook']['access_token']
        }
        
        # Hash user data for privacy
        if conversion_data.get('email'):
            event_data['data'][0]['user_data']['em'] = [
                hashlib.sha256(conversion_data['email'].lower().encode()).hexdigest()
            ]
        
        if conversion_data.get('phone'):
            event_data['data'][0]['user_data']['ph'] = [
                hashlib.sha256(conversion_data['phone'].encode()).hexdigest()
            ]
        
        # Send to Facebook
        url = self.platforms['facebook']['url'].format(
            pixel_id=self.platforms['facebook']['pixel_id']
        )
        
        response = requests.post(url, json=event_data, timeout=10)
        
        if response.status_code == 200:
            return {
                'success': True, 
                'platform': 'facebook',
                'response': response.json()
            }
        else:
            return {
                'success': False, 
                'platform': 'facebook',
                'error': f'HTTP {response.status_code}: {response.text}'
            }
    
    def _send_google_event(self, conversion_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send conversion event to Google Ads Conversions API"""
        
        if not self.platforms['google']['conversion_id']:
            return {'success': False, 'error': 'Google Ads credentials not configured'}
        
        # Build Google Ads conversion data
        conversion_payload = {
            'conversion_action': f"customers/{settings.GOOGLE_ADS_CUSTOMER_ID}/conversionActions/{self.platforms['google']['conversion_id']}",
            'conversion_date_time': conversion_data['event_time'].strftime('%Y-%m-%d %H:%M:%S%z'),
            'conversion_value': conversion_data.get('value', 0),
            'currency_code': 'JPY',
            'click_identifier': {
                'gclid': conversion_data.get('google_click_id')  # Google Click ID
            } if conversion_data.get('google_click_id') else None,
            'order_id': conversion_data.get('event_id'),
            'user_identifiers': []
        }
        
        # Add user identifiers (hashed)
        if conversion_data.get('email'):
            conversion_payload['user_identifiers'].append({
                'hashed_email': hashlib.sha256(conversion_data['email'].lower().encode()).hexdigest()
            })
        
        if conversion_data.get('phone'):
            conversion_payload['user_identifiers'].append({
                'hashed_phone_number': hashlib.sha256(conversion_data['phone'].encode()).hexdigest()
            })
        
        # For Google Ads API, you would typically use the Google Ads Python client
        # This is a simplified version for demonstration
        try:
            # Mock successful response for Google Ads
            return {
                'success': True,
                'platform': 'google',
                'message': 'Conversion sent to Google Ads (mock implementation)'
            }
        except Exception as e:
            return {
                'success': False,
                'platform': 'google', 
                'error': str(e)
            }
    
    def _send_twitter_event(self, conversion_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send conversion event to Twitter Ads API"""
        
        if not self.platforms['twitter']['api_key']:
            return {'success': False, 'error': 'Twitter credentials not configured'}
        
        # Twitter conversion event format
        event_payload = {
            'conversions': [{
                'conversion_time': conversion_data['event_time'].isoformat(),
                'event_id': conversion_data.get('event_id'),
                'identifiers': [{
                    'twclid': conversion_data.get('twitter_click_id')
                }] if conversion_data.get('twitter_click_id') else [],
                'conversion_value': conversion_data.get('value', 0),
                'currency': 'JPY',
                'conversion_type': 'PURCHASE'
            }]
        }
        
        # Mock implementation for Twitter
        return {
            'success': True,
            'platform': 'twitter',
            'message': 'Conversion sent to Twitter (mock implementation)'
        }
    
    def _map_event_name(self, event_type: str) -> str:
        """Map internal event types to platform-specific event names"""
        
        event_mapping = {
            'property_inquiry': 'Lead',
            'property_purchase': 'Purchase', 
            'property_view': 'ViewContent',
            'mortgage_application': 'SubmitApplication',
            'consultation_booking': 'Schedule',
            'signup': 'CompleteRegistration'
        }
        
        return event_mapping.get(event_type, 'CustomEvent')

class ConversionTracker:
    """
    Service to track and attribute conversions from GP MLOps' ad campaigns
    """
    
    def __init__(self):
        self.capi_client = ConversionAPIClient()
        self.attribution_window_hours = 168  # 7 days default
    
    def track_conversion(self, conversion_data: Dict[str, Any], db: Session) -> Dict[str, Any]:
        """
        Track conversion event and send to advertising platforms
        
        Args:
            conversion_data: Conversion event details
            db: Database session
            
        Returns:
            Tracking results
        """
        try:
            # Create conversion record
            conversion_record = ConversionAPI(
                event_id=conversion_data['event_id'],
                event_type=conversion_data['event_type'],
                event_time=conversion_data['event_time'],
                customer_id=conversion_data['customer_id'],
                conversion_value=conversion_data.get('value', 0),
                event_properties=conversion_data.get('properties', {}),
                user_properties=conversion_data.get('user_properties', {}),
                api_version='v1.0',
                source_platform='ga_technology_platform'
            )
            
            # Perform attribution analysis
            attribution_result = self._perform_attribution(conversion_data, db)
            
            if attribution_result:
                conversion_record.attributed_campaign_id = attribution_result['campaign_id']
                conversion_record.attributed_creative_id = attribution_result['creative_id']
                conversion_record.attribution_window_hours = self.attribution_window_hours
            
            # Save to database
            db.add(conversion_record)
            db.commit()
            db.refresh(conversion_record)
            
            # Send to advertising platforms via CAPI
            capi_results = self.capi_client.send_conversion_event(conversion_data)
            
            # Update campaign metrics if attributed
            if attribution_result:
                self._update_campaign_metrics(attribution_result, conversion_data['value'], db)
            
            return {
                'conversion_id': conversion_record.id,
                'event_id': conversion_record.event_id,
                'attributed': attribution_result is not None,
                'attribution': attribution_result,
                'capi_results': capi_results,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Conversion tracking failed: {e}")
            db.rollback()
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _perform_attribution(self, conversion_data: Dict[str, Any], db: Session) -> Optional[Dict[str, Any]]:
        """
        Perform attribution analysis to link conversion to ad interactions
        Uses last-click attribution within the attribution window
        """
        try:
            customer_id = conversion_data['customer_id']
            conversion_time = conversion_data['event_time']
            attribution_window_start = conversion_time - timezone.utc.localize(
                datetime.utcnow() - datetime.utcfromtimestamp(
                    conversion_time.timestamp() - (self.attribution_window_hours * 3600)
                )
            ).replace(tzinfo=None)
            
            # Find ad interactions within attribution window
            interactions = db.query(AdInteraction).join(AdServing).filter(
                and_(
                    AdServing.customer_id == customer_id,
                    AdInteraction.interaction_time >= attribution_window_start,
                    AdInteraction.interaction_time <= conversion_time,
                    AdInteraction.interaction_type.in_(['click', 'impression'])
                )
            ).order_by(AdInteraction.interaction_time.desc()).all()
            
            if not interactions:
                logger.info(f"No ad interactions found for customer {customer_id} within attribution window")
                return None
            
            # Last-click attribution
            last_click = next((i for i in interactions if i.interaction_type == 'click'), None)
            
            if last_click:
                # Attribution to last click
                time_to_conversion = int((conversion_time - last_click.interaction_time).total_seconds())
                
                return {
                    'campaign_id': last_click.campaign_id,
                    'creative_id': last_click.creative_id,
                    'serving_id': last_click.serving_id,
                    'attribution_type': 'last_click',
                    'interaction_time': last_click.interaction_time,
                    'time_to_conversion_seconds': time_to_conversion
                }
            else:
                # Attribution to last impression (view-through conversion)
                last_impression = interactions[0]  # Most recent interaction
                time_to_conversion = int((conversion_time - last_impression.interaction_time).total_seconds())
                
                return {
                    'campaign_id': last_impression.campaign_id,
                    'creative_id': last_impression.creative_id,
                    'serving_id': last_impression.serving_id,
                    'attribution_type': 'view_through',
                    'interaction_time': last_impression.interaction_time,
                    'time_to_conversion_seconds': time_to_conversion
                }
            
        except Exception as e:
            logger.error(f"Attribution analysis failed: {e}")
            return None
    
    def _update_campaign_metrics(self, attribution_result: Dict[str, Any], 
                               conversion_value: float, db: Session):
        """Update campaign and creative metrics with attribution data"""
        try:
            # Update campaign conversion count
            campaign = db.query(AdCampaign).filter(
                AdCampaign.id == attribution_result['campaign_id']
            ).first()
            
            if campaign:
                campaign.conversions += 1
            
            # Update creative conversion count  
            creative = db.query(AdCreative).filter(
                AdCreative.id == attribution_result['creative_id']
            ).first()
            
            if creative:
                creative.conversions += 1
                # Recalculate conversion rate
                creative.cvr = creative.conversions / max(1, creative.clicks)
            
            db.commit()
            
            logger.info(f"Updated metrics for campaign {attribution_result['campaign_id']}, "
                       f"creative {attribution_result['creative_id']}")
            
        except Exception as e:
            logger.error(f"Failed to update campaign metrics: {e}")
            db.rollback()
    
    def get_conversion_funnel(self, campaign_id: int, days: int, db: Session) -> Dict[str, Any]:
        """
        Get conversion funnel metrics for a campaign
        
        Args:
            campaign_id: Campaign ID to analyze
            days: Number of days to look back
            db: Database session
            
        Returns:
            Funnel metrics and conversion paths
        """
        try:
            date_from = datetime.now() - timedelta(days=days)
            
            # Get campaign serving data
            servings = db.query(AdServing).filter(
                and_(
                    AdServing.campaign_id == campaign_id,
                    AdServing.serving_time >= date_from
                )
            ).all()
            
            # Get interactions
            interactions = db.query(AdInteraction).filter(
                and_(
                    AdInteraction.campaign_id == campaign_id,
                    AdInteraction.interaction_time >= date_from
                )
            ).all()
            
            # Get conversions
            conversions = db.query(ConversionAPI).filter(
                and_(
                    ConversionAPI.attributed_campaign_id == campaign_id,
                    ConversionAPI.event_time >= date_from
                )
            ).all()
            
            # Build funnel metrics
            impressions = len([i for i in interactions if i.interaction_type == 'impression'])
            clicks = len([i for i in interactions if i.interaction_type == 'click'])
            total_conversions = len(conversions)
            
            # Conversion paths analysis
            conversion_paths = {}
            for conversion in conversions:
                path_key = f"{conversion.attributed_campaign_id}_{conversion.attributed_creative_id}"
                if path_key not in conversion_paths:
                    conversion_paths[path_key] = {
                        'conversions': 0,
                        'total_value': 0,
                        'avg_time_to_conversion': 0
                    }
                
                conversion_paths[path_key]['conversions'] += 1
                conversion_paths[path_key]['total_value'] += conversion.conversion_value or 0
            
            return {
                'campaign_id': campaign_id,
                'date_range': f'{days} days',
                'funnel_metrics': {
                    'served_ads': len(servings),
                    'impressions': impressions,
                    'clicks': clicks,
                    'conversions': total_conversions,
                    'impression_rate': impressions / max(1, len(servings)),
                    'click_through_rate': clicks / max(1, impressions),
                    'conversion_rate': total_conversions / max(1, clicks),
                    'overall_conversion_rate': total_conversions / max(1, len(servings))
                },
                'conversion_paths': conversion_paths,
                'total_conversion_value': sum(c.conversion_value or 0 for c in conversions)
            }
            
        except Exception as e:
            logger.error(f"Conversion funnel analysis failed: {e}")
            return {'error': str(e)}