import random
import hashlib
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from app.models.advertising import (
    AdCampaign, AdCreative, AdServing, AdInteraction, AudienceProfile,
    AdCampaignStatus, AdVariantType, AudienceSegment
)
from app.models.customer import Customer, CustomerPrediction
from app.services.customer_inference import CustomerInferenceService
from app.core.monitoring import record_model_prediction
import time

logger = logging.getLogger(__name__)

class AdServingEngine:
    def __init__(self):
        self.customer_inference = CustomerInferenceService()
        
        # GP MLOps' primary target criteria
        self.PRIMARY_TARGET_CRITERIA = {
            'min_age': 40,
            'max_age': 50,
            'min_income': 10000000,  # 10M yen annually
            'target_professions': ['engineer', 'manager', 'executive', 'consultant', 'doctor'],
            'target_locations': ['tokyo', 'osaka', 'yokohama', 'kyoto', 'kobe', 'nagoya']
        }
    
    def serve_ad(self, customer_data: Dict[str, Any], request_context: Dict[str, Any], 
                 db: Session) -> Optional[Dict[str, Any]]:
        """
        Main ad serving logic with targeting and A/B testing
        
        Args:
            customer_data: Customer profile data
            request_context: Request metadata (IP, user agent, etc.)
            db: Database session
            
        Returns:
            Ad creative to serve or None if no match
        """
        try:
            start_time = time.time()
            
            # Step 1: Check if customer matches primary target criteria
            if not self._matches_primary_target(customer_data):
                logger.info(f"Customer {customer_data.get('id')} doesn't match primary target criteria")
                return None
            
            # Step 2: Get customer predictions for targeting
            predictions = self._get_customer_predictions(customer_data, db)
            if not predictions:
                logger.warning(f"No predictions available for customer {customer_data.get('id')}")
                return None
            
            # Step 3: Find eligible campaigns
            eligible_campaigns = self._get_eligible_campaigns(customer_data, predictions, db)
            if not eligible_campaigns:
                logger.info(f"No eligible campaigns for customer {customer_data.get('id')}")
                return None
            
            # Step 4: Select best campaign based on relevance and budget
            selected_campaign = self._select_campaign(eligible_campaigns, predictions)
            
            # Step 5: A/B test variant assignment
            selected_creative = self._assign_ab_test_variant(
                selected_campaign, customer_data, db
            )
            
            # Step 6: Log ad serving decision
            serving_record = self._log_ad_serving(
                selected_campaign, selected_creative, customer_data, 
                request_context, predictions, db
            )
            
            # Step 7: Build response
            ad_response = self._build_ad_response(
                selected_campaign, selected_creative, serving_record
            )
            
            # Record metrics
            duration = time.time() - start_time
            record_model_prediction("ad_serving", duration, True)
            
            logger.info(f"Served ad for customer {customer_data.get('id')}: "
                       f"Campaign {selected_campaign.id}, Creative {selected_creative.id}")
            
            return ad_response
            
        except Exception as e:
            duration = time.time() - start_time
            record_model_prediction("ad_serving", duration, False)
            logger.error(f"Ad serving failed: {e}")
            return None
    
    def _matches_primary_target(self, customer_data: Dict[str, Any]) -> bool:
        """Check if customer matches GP MLOps' primary targeting criteria"""
        
        age = customer_data.get('age', 0)
        income = customer_data.get('income', 0)
        profession = customer_data.get('profession', '').lower()
        location = customer_data.get('location', '').lower()
        
        # Age check (40-50 years old)
        if not (self.PRIMARY_TARGET_CRITERIA['min_age'] <= age <= self.PRIMARY_TARGET_CRITERIA['max_age']):
            return False
        
        # Income check (>10M yen annually)
        if income < self.PRIMARY_TARGET_CRITERIA['min_income']:
            return False
        
        # Profession check (salaryman in target roles)
        if profession not in self.PRIMARY_TARGET_CRITERIA['target_professions']:
            return False
        
        # Location check (major Japanese cities)
        if location not in self.PRIMARY_TARGET_CRITERIA['target_locations']:
            return False
        
        return True
    
    def _get_customer_predictions(self, customer_data: Dict[str, Any], db: Session) -> Optional[Dict[str, Any]]:
        """Get customer predictions for ad targeting"""
        try:
            # Check for recent predictions in database
            customer_id = customer_data.get('id')
            recent_prediction = db.query(CustomerPrediction).filter(
                and_(
                    CustomerPrediction.customer_id == customer_id,
                    CustomerPrediction.created_at >= datetime.now() - timedelta(hours=24)
                )
            ).order_by(CustomerPrediction.created_at.desc()).first()
            
            if recent_prediction:
                return {
                    'conversion_probability': recent_prediction.prediction_value,
                    'confidence_score': recent_prediction.confidence_score,
                    'lifetime_value': 500000  # Mock value, should be from separate model
                }
            
            # Generate new predictions if none exist
            # In production, this would be called asynchronously
            mock_customer = type('Customer', (), customer_data)()
            predictions = self.customer_inference.predict_customer_value(mock_customer, [])
            
            return predictions
            
        except Exception as e:
            logger.error(f"Failed to get customer predictions: {e}")
            return None
    
    def _get_eligible_campaigns(self, customer_data: Dict[str, Any], 
                              predictions: Dict[str, Any], db: Session) -> List[AdCampaign]:
        """Find campaigns eligible for the customer"""
        
        age = customer_data.get('age')
        income = customer_data.get('income')
        location = customer_data.get('location', '').lower()
        
        # Query active campaigns with targeting criteria
        eligible_campaigns = db.query(AdCampaign).filter(
            and_(
                AdCampaign.status == AdCampaignStatus.ACTIVE,
                AdCampaign.start_date <= datetime.now(),
                or_(AdCampaign.end_date.is_(None), AdCampaign.end_date >= datetime.now()),
                AdCampaign.min_age <= age,
                AdCampaign.max_age >= age,
                AdCampaign.min_income <= income,
                # Check if daily budget not exceeded
                AdCampaign.daily_budget > AdCampaign.spend / 24  # Simplified daily spend check
            )
        ).all()
        
        # Filter by location targeting
        location_filtered = []
        for campaign in eligible_campaigns:
            target_locations = campaign.target_locations or []
            if not target_locations or location in target_locations:
                location_filtered.append(campaign)
        
        # Filter by conversion probability threshold
        probability_filtered = []
        min_conversion_prob = 0.3  # Minimum threshold for serving ads
        
        if predictions['conversion_probability'] >= min_conversion_prob:
            probability_filtered = location_filtered
        
        return probability_filtered
    
    def _select_campaign(self, eligible_campaigns: List[AdCampaign], 
                        predictions: Dict[str, Any]) -> AdCampaign:
        """Select the best campaign based on relevance scoring"""
        
        if len(eligible_campaigns) == 1:
            return eligible_campaigns[0]
        
        # Score campaigns based on multiple factors
        campaign_scores = []
        
        for campaign in eligible_campaigns:
            score = 0.0
            
            # Factor 1: Conversion probability alignment (40% weight)
            conv_prob = predictions['conversion_probability']
            if campaign.target_audience == AudienceSegment.HIGH_INCOME_SALARYMAN:
                if conv_prob >= 0.7:
                    score += 0.4
                elif conv_prob >= 0.5:
                    score += 0.3
                else:
                    score += 0.2
            
            # Factor 2: Budget availability (30% weight)
            daily_budget_remaining = max(0, campaign.daily_budget - (campaign.spend / 24))
            budget_factor = min(1.0, daily_budget_remaining / campaign.daily_budget)
            score += budget_factor * 0.3
            
            # Factor 3: Campaign performance (30% weight)
            if campaign.impressions > 100:  # Has enough data
                campaign_ctr = campaign.clicks / campaign.impressions
                campaign_cvr = campaign.conversions / max(1, campaign.clicks)
                performance_score = (campaign_ctr * 0.5 + campaign_cvr * 0.5)
                score += min(performance_score, 0.3)
            else:
                score += 0.15  # Default score for new campaigns
            
            campaign_scores.append((campaign, score))
        
        # Select campaign with highest score
        campaign_scores.sort(key=lambda x: x[1], reverse=True)
        selected_campaign = campaign_scores[0][0]
        
        logger.info(f"Selected campaign {selected_campaign.id} with score {campaign_scores[0][1]:.3f}")
        
        return selected_campaign
    
    def _assign_ab_test_variant(self, campaign: AdCampaign, customer_data: Dict[str, Any], 
                               db: Session) -> AdCreative:
        """Assign A/B test variant using consistent hashing"""
        
        # Get available creatives for the campaign
        creatives = db.query(AdCreative).filter(
            and_(
                AdCreative.campaign_id == campaign.id,
                AdCreative.is_active == True
            )
        ).all()
        
        if not creatives:
            raise ValueError(f"No active creatives found for campaign {campaign.id}")
        
        if not campaign.ab_test_enabled or len(creatives) == 1:
            return creatives[0]
        
        # Use consistent hashing for stable variant assignment
        customer_id = customer_data.get('id', 0)
        hash_input = f"{campaign.id}_{customer_id}".encode('utf-8')
        hash_value = int(hashlib.md5(hash_input).hexdigest(), 16) % 100
        
        # Get traffic split configuration
        traffic_split = campaign.traffic_split or {"control": 100}
        
        # Assign variant based on hash and traffic split
        cumulative_percentage = 0
        for variant_name, percentage in traffic_split.items():
            cumulative_percentage += percentage
            if hash_value < cumulative_percentage:
                # Find creative with matching variant type
                for creative in creatives:
                    if creative.variant_type.value == variant_name:
                        logger.info(f"Assigned variant {variant_name} to customer {customer_id}")
                        return creative
        
        # Fallback to control variant
        control_creative = next((c for c in creatives if c.variant_type == AdVariantType.CONTROL), creatives[0])
        return control_creative
    
    def _log_ad_serving(self, campaign: AdCampaign, creative: AdCreative, 
                       customer_data: Dict[str, Any], request_context: Dict[str, Any],
                       predictions: Dict[str, Any], db: Session) -> AdServing:
        """Log the ad serving decision"""
        
        serving_record = AdServing(
            campaign_id=campaign.id,
            creative_id=creative.id,
            customer_id=customer_data.get('id'),
            user_agent=request_context.get('user_agent', ''),
            ip_address=request_context.get('ip_address', ''),
            device_type=request_context.get('device_type', 'unknown'),
            user_age=customer_data.get('age'),
            user_income=customer_data.get('income'),
            user_location=customer_data.get('location', ''),
            user_segment=AudienceSegment.HIGH_INCOME_SALARYMAN.value,
            conversion_probability=predictions['conversion_probability'],
            estimated_lifetime_value=predictions.get('lifetime_value', 0),
            relevance_score=0.8,  # Could be computed based on targeting match
            was_served=True,
            serving_reason='targeting_match_ab_test',
            ab_test_variant=creative.variant_type
        )
        
        db.add(serving_record)
        db.commit()
        db.refresh(serving_record)
        
        return serving_record
    
    def _build_ad_response(self, campaign: AdCampaign, creative: AdCreative, 
                          serving_record: AdServing) -> Dict[str, Any]:
        """Build the ad response for the client"""
        
        return {
            'ad_id': f"{campaign.id}_{creative.id}_{serving_record.id}",
            'campaign_id': campaign.id,
            'creative_id': creative.id,
            'serving_id': serving_record.id,
            'variant_type': creative.variant_type.value,
            'content': {
                'headline': creative.headline,
                'description': creative.description,
                'call_to_action': creative.call_to_action,
                'image_url': creative.image_url,
                'landing_page_url': creative.landing_page_url
            },
            'tracking': {
                'impression_url': f"/api/v1/ads/track/impression/{serving_record.id}",
                'click_url': f"/api/v1/ads/track/click/{serving_record.id}",
                'conversion_pixel': f"/api/v1/ads/track/conversion/{serving_record.id}"
            },
            'metadata': {
                'campaign_name': campaign.name,
                'target_audience': campaign.target_audience.value,
                'creative_type': creative.creative_type,
                'dimensions': creative.dimensions
            }
        }
    
    def track_interaction(self, serving_id: int, interaction_type: str, 
                         interaction_context: Dict[str, Any], db: Session) -> bool:
        """Track ad interactions (impressions, clicks, conversions)"""
        try:
            # Get serving record
            serving_record = db.query(AdServing).filter(AdServing.id == serving_id).first()
            if not serving_record:
                logger.warning(f"Serving record {serving_id} not found")
                return False
            
            # Create interaction record
            interaction = AdInteraction(
                serving_id=serving_id,
                campaign_id=serving_record.campaign_id,
                creative_id=serving_record.creative_id,
                customer_id=serving_record.customer_id,
                interaction_type=interaction_type,
                interaction_value=interaction_context.get('value', 0),
                page_url=interaction_context.get('page_url', ''),
                referrer_url=interaction_context.get('referrer_url', ''),
                session_id=interaction_context.get('session_id', '')
            )
            
            db.add(interaction)
            
            # Update serving record flags
            if interaction_type == 'impression' and not serving_record.impression_tracked:
                serving_record.impression_tracked = True
                
                # Update campaign and creative impression counts
                campaign = db.query(AdCampaign).filter(AdCampaign.id == serving_record.campaign_id).first()
                creative = db.query(AdCreative).filter(AdCreative.id == serving_record.creative_id).first()
                
                if campaign:
                    campaign.impressions += 1
                if creative:
                    creative.impressions += 1
                    
            elif interaction_type == 'click' and not serving_record.click_tracked:
                serving_record.click_tracked = True
                
                # Update campaign and creative click counts
                campaign = db.query(AdCampaign).filter(AdCampaign.id == serving_record.campaign_id).first()
                creative = db.query(AdCreative).filter(AdCreative.id == serving_record.creative_id).first()
                
                if campaign:
                    campaign.clicks += 1
                if creative:
                    creative.clicks += 1
                    creative.ctr = creative.clicks / max(1, creative.impressions)
                    
            elif interaction_type == 'conversion' and not serving_record.conversion_tracked:
                serving_record.conversion_tracked = True
                
                # Update campaign and creative conversion counts
                campaign = db.query(AdCampaign).filter(AdCampaign.id == serving_record.campaign_id).first()
                creative = db.query(AdCreative).filter(AdCreative.id == serving_record.creative_id).first()
                
                if campaign:
                    campaign.conversions += 1
                if creative:
                    creative.conversions += 1
                    creative.cvr = creative.conversions / max(1, creative.clicks)
            
            db.commit()
            
            logger.info(f"Tracked {interaction_type} for serving {serving_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to track interaction: {e}")
            db.rollback()
            return False

class ABTestAnalyzer:
    def __init__(self):
        self.min_sample_size = 100
        self.confidence_level = 0.95
        
    def analyze_campaign_results(self, campaign_id: int, db: Session) -> Dict[str, Any]:
        """Analyze A/B test results for a campaign"""
        try:
            # Get campaign and creatives
            campaign = db.query(AdCampaign).filter(AdCampaign.id == campaign_id).first()
            if not campaign:
                raise ValueError(f"Campaign {campaign_id} not found")
            
            creatives = db.query(AdCreative).filter(
                AdCreative.campaign_id == campaign_id
            ).all()
            
            if len(creatives) < 2:
                return {"error": "Need at least 2 variants for A/B testing"}
            
            # Calculate metrics for each variant
            variant_results = {}
            for creative in creatives:
                if creative.impressions >= self.min_sample_size:
                    variant_results[creative.variant_type.value] = {
                        'impressions': creative.impressions,
                        'clicks': creative.clicks,
                        'conversions': creative.conversions,
                        'ctr': creative.ctr,
                        'cvr': creative.cvr,
                        'creative_id': creative.id
                    }
            
            if len(variant_results) < 2:
                return {"error": "Insufficient data for statistical analysis"}
            
            # Perform statistical significance test
            significance_results = self._calculate_significance(variant_results)
            
            # Determine winner
            winner = self._determine_winner(variant_results, significance_results)
            
            return {
                'campaign_id': campaign_id,
                'test_duration_days': (datetime.now() - campaign.start_date).days,
                'variant_results': variant_results,
                'statistical_results': significance_results,
                'winner': winner,
                'recommendation': self._generate_recommendation(variant_results, winner)
            }
            
        except Exception as e:
            logger.error(f"A/B test analysis failed: {e}")
            return {"error": str(e)}
    
    def _calculate_significance(self, variant_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate statistical significance between variants"""
        
        # Simplified z-test for conversion rates
        variants = list(variant_results.keys())
        if len(variants) < 2:
            return {}
        
        control = variant_results[variants[0]]
        variant = variant_results[variants[1]]
        
        # Calculate conversion rates
        p1 = control['conversions'] / max(1, control['clicks'])
        p2 = variant['conversions'] / max(1, variant['clicks'])
        
        n1 = control['clicks']
        n2 = variant['clicks']
        
        # Pooled standard error
        p_pool = (control['conversions'] + variant['conversions']) / max(1, n1 + n2)
        se = np.sqrt(p_pool * (1 - p_pool) * (1/max(1, n1) + 1/max(1, n2)))
        
        # Z-score
        z_score = (p2 - p1) / max(se, 0.0001)
        
        # P-value (simplified, assumes normal distribution)
        p_value = 2 * (1 - abs(z_score) / 1.96) if abs(z_score) <= 1.96 else 0.05 if abs(z_score) > 1.96 else 1.0
        
        return {
            'z_score': z_score,
            'p_value': min(max(p_value, 0.0001), 1.0),
            'significant': abs(z_score) > 1.96,
            'confidence_interval': [p2 - 1.96 * se, p2 + 1.96 * se],
            'effect_size': (p2 - p1) / max(p1, 0.0001) if p1 > 0 else 0
        }
    
    def _determine_winner(self, variant_results: Dict[str, Any], 
                         significance_results: Dict[str, Any]) -> Dict[str, Any]:
        """Determine the winning variant"""
        
        # Find variant with best conversion rate
        best_variant = max(variant_results.items(), key=lambda x: x[1]['cvr'])
        
        is_significant = significance_results.get('significant', False)
        
        return {
            'variant': best_variant[0],
            'creative_id': best_variant[1]['creative_id'],
            'cvr': best_variant[1]['cvr'],
            'improvement': significance_results.get('effect_size', 0) * 100,
            'is_statistically_significant': is_significant,
            'confidence': 95 if is_significant else 50
        }
    
    def _generate_recommendation(self, variant_results: Dict[str, Any], 
                               winner: Dict[str, Any]) -> str:
        """Generate actionable recommendation"""
        
        if winner['is_statistically_significant']:
            if winner['improvement'] > 10:
                return f"Strong recommendation: Deploy {winner['variant']} (Creative {winner['creative_id']}) " \
                       f"with {winner['improvement']:.1f}% improvement in conversion rate."
            else:
                return f"Moderate recommendation: Consider deploying {winner['variant']} " \
                       f"with {winner['improvement']:.1f}% improvement, but continue monitoring."
        else:
            return "Inconclusive: Continue testing. No statistically significant difference found. " \
                   "Consider running the test longer or increasing traffic."