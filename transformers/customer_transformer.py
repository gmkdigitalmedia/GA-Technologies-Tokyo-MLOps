import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Union
import logging
import os
from kserve import Model, ModelServer
import asyncio

logger = logging.getLogger(__name__)

class CustomerValueTransformer(Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.ready = False
        self.predictor_host = os.environ.get("PREDICTOR_HOST", "customer-value-model-predictor")
        
        # Feature engineering components
        self.feature_columns = [
            'age', 'income', 'family_size', 'total_interactions',
            'unique_properties_viewed', 'avg_session_duration',
            'engagement_score', 'profession_encoded', 'location_encoded'
        ]
        
        # Mock label encoders (in production, load from model artifacts)
        self.profession_mapping = {
            'engineer': 0, 'manager': 1, 'sales': 2, 'teacher': 3,
            'doctor': 4, 'lawyer': 5, 'consultant': 6, 'other': 7
        }
        
        self.location_mapping = {
            'tokyo': 0, 'osaka': 1, 'nagoya': 2, 'yokohama': 3,
            'kyoto': 4, 'kobe': 5, 'sapporo': 6, 'other': 7
        }

    def load(self):
        """Load transformer components"""
        self.ready = True
        logger.info(f"CustomerValueTransformer {self.name} loaded successfully")

    def preprocess(self, inputs: Dict[str, Any], headers: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Transform input data for customer value inference
        
        Expected input format:
        {
            "instances": [
                {
                    "customer_id": 12345,
                    "age": 45,
                    "income": 8000000,
                    "profession": "engineer",
                    "location": "tokyo",
                    "family_size": 3,
                    "interactions": [
                        {"type": "view", "property_id": "prop123", "duration": 300},
                        {"type": "inquiry", "property_id": "prop456", "duration": 180}
                    ]
                }
            ]
        }
        """
        try:
            instances = inputs.get("instances", [])
            if not instances:
                raise ValueError("No instances provided")
            
            processed_instances = []
            
            for instance in instances:
                # Extract basic features
                features = {
                    'age': instance.get('age', 40),
                    'income': instance.get('income', 5000000),
                    'family_size': instance.get('family_size', 1)
                }
                
                # Process interactions
                interactions = instance.get('interactions', [])
                features.update(self._process_interactions(interactions))
                
                # Encode categorical features
                features['profession_encoded'] = self.profession_mapping.get(
                    instance.get('profession', 'other').lower(), 7
                )
                features['location_encoded'] = self.location_mapping.get(
                    instance.get('location', 'other').lower(), 7
                )
                
                # Calculate engagement score
                features['engagement_score'] = min(
                    features['total_interactions'] * 0.3 + 
                    features['unique_properties_viewed'] * 0.7,
                    100
                )
                
                # Create feature array in correct order
                feature_array = [features[col] for col in self.feature_columns]
                processed_instances.append(feature_array)
            
            return {"instances": processed_instances}
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise
    
    def postprocess(self, inputs: Dict[str, Any], headers: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Transform model output to business-friendly format
        
        Model output format: {"predictions": [[conversion_prob, lifetime_value], ...]}
        Business output format: Structured response with confidence scores
        """
        try:
            predictions = inputs.get("predictions", [])
            if not predictions:
                return {"predictions": []}
            
            processed_predictions = []
            
            for i, prediction in enumerate(predictions):
                if len(prediction) >= 2:
                    conversion_prob = float(prediction[0])
                    lifetime_value = float(prediction[1])
                    
                    # Calculate confidence score
                    confidence = min(0.95, max(0.1, conversion_prob * 0.6 + 0.3))
                    
                    # Determine customer segment
                    segment = self._determine_segment(conversion_prob, lifetime_value)
                    
                    processed_prediction = {
                        "customer_index": i,
                        "conversion_probability": round(conversion_prob, 4),
                        "estimated_lifetime_value": round(lifetime_value, 2),
                        "confidence_score": round(confidence, 4),
                        "customer_segment": segment,
                        "recommendations": self._generate_recommendations(
                            conversion_prob, lifetime_value
                        ),
                        "model_version": "v1.2",
                        "prediction_timestamp": pd.Timestamp.now().isoformat()
                    }
                    
                    processed_predictions.append(processed_prediction)
                else:
                    logger.warning(f"Invalid prediction format at index {i}")
                    processed_predictions.append({
                        "customer_index": i,
                        "error": "Invalid prediction format"
                    })
            
            return {"predictions": processed_predictions}
            
        except Exception as e:
            logger.error(f"Postprocessing failed: {e}")
            raise
    
    def _process_interactions(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process customer interactions to extract features"""
        if not interactions:
            return {
                'total_interactions': 0,
                'unique_properties_viewed': 0,
                'avg_session_duration': 0.0
            }
        
        # Calculate interaction features
        total_interactions = len(interactions)
        unique_properties = len(set(
            i.get('property_id') for i in interactions 
            if i.get('property_id')
        ))
        
        durations = [i.get('duration', 0) for i in interactions]
        avg_duration = np.mean(durations) if durations else 0.0
        
        return {
            'total_interactions': total_interactions,
            'unique_properties_viewed': unique_properties,
            'avg_session_duration': float(avg_duration)
        }
    
    def _determine_segment(self, conversion_prob: float, lifetime_value: float) -> str:
        """Determine customer segment based on predictions"""
        if conversion_prob >= 0.8 and lifetime_value >= 1000000:
            return "high_value_high_conversion"
        elif conversion_prob >= 0.8:
            return "high_conversion"
        elif lifetime_value >= 1000000:
            return "high_value"
        elif conversion_prob >= 0.6:
            return "medium_conversion"
        elif lifetime_value >= 500000:
            return "medium_value"
        else:
            return "low_priority"
    
    def _generate_recommendations(self, conversion_prob: float, lifetime_value: float) -> List[str]:
        """Generate actionable recommendations based on predictions"""
        recommendations = []
        
        if conversion_prob >= 0.8:
            recommendations.append("prioritize_immediate_contact")
            recommendations.append("offer_premium_properties")
        elif conversion_prob >= 0.6:
            recommendations.append("nurture_with_targeted_content")
            recommendations.append("schedule_property_viewing")
        else:
            recommendations.append("engage_with_educational_content")
            recommendations.append("build_relationship_slowly")
        
        if lifetime_value >= 1000000:
            recommendations.append("assign_senior_agent")
            recommendations.append("provide_vip_service")
        elif lifetime_value >= 500000:
            recommendations.append("offer_personalized_search")
            
        return recommendations

if __name__ == "__main__":
    model = CustomerValueTransformer("customer-value-transformer")
    model.load()
    ModelServer().start([model])