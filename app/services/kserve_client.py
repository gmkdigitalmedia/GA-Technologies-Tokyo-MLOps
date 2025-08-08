import requests
import json
import base64
import numpy as np
from typing import Dict, List, Any, Optional, Union
import logging
from PIL import Image
import io
from app.core.config import settings

logger = logging.getLogger(__name__)

class KServeClient:
    def __init__(self, base_url: str = None):
        self.base_url = base_url or "http://istio-ingressgateway.istio-system.svc.cluster.local"
        self.timeout = 30
        
    def predict_customer_value(self, customer_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Send customer data to KServe for value prediction
        
        Args:
            customer_data: List of customer instances with features
            
        Returns:
            Prediction response with conversion probability and lifetime value
        """
        try:
            url = f"{self.base_url}/v1/models/customer-value:predict"
            headers = {
                "Content-Type": "application/json",
                "Host": "ga-mlops-inference.example.com"
            }
            
            payload = {"instances": customer_data}
            
            response = requests.post(
                url, 
                json=payload, 
                headers=headers, 
                timeout=self.timeout
            )
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"KServe customer value prediction failed: {e}")
            raise
    
    def predict_floorplan(self, floorplan_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Send floorplan images to KServe for layout detection
        
        Args:
            floorplan_data: List of floorplan instances with image data
            
        Returns:
            Floorplan analysis response
        """
        try:
            url = f"{self.base_url}/v1/models/floorplan:predict"
            headers = {
                "Content-Type": "application/json",
                "Host": "ga-mlops-inference.example.com"
            }
            
            payload = {"instances": floorplan_data}
            
            response = requests.post(
                url, 
                json=payload, 
                headers=headers, 
                timeout=self.timeout * 2  # Longer timeout for image processing
            )
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"KServe floorplan prediction failed: {e}")
            raise
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """
        Encode image file to base64 string
        
        Args:
            image_path: Path to image file
            
        Returns:
            Base64 encoded image string
        """
        try:
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                base64_string = base64.b64encode(image_data).decode('utf-8')
                return base64_string
                
        except Exception as e:
            logger.error(f"Image encoding failed: {e}")
            raise
    
    def prepare_customer_data(self, customers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prepare customer data for KServe prediction
        
        Args:
            customers: Raw customer data
            
        Returns:
            Formatted customer data for KServe
        """
        formatted_data = []
        
        for customer in customers:
            formatted_customer = {
                "customer_id": customer.get("id"),
                "age": customer.get("age", 40),
                "income": customer.get("income", 5000000),
                "profession": customer.get("profession", "other"),
                "location": customer.get("location", "other"),
                "family_size": customer.get("family_size", 1),
                "interactions": customer.get("interactions", [])
            }
            formatted_data.append(formatted_customer)
        
        return formatted_data
    
    def prepare_floorplan_data(self, properties: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prepare floorplan data for KServe prediction
        
        Args:
            properties: Property data with image paths
            
        Returns:
            Formatted floorplan data for KServe
        """
        formatted_data = []
        
        for property_data in properties:
            image_path = property_data.get("floorplan_image_path")
            if not image_path:
                logger.warning(f"No floorplan image for property {property_data.get('id')}")
                continue
            
            try:
                # Encode image to base64
                base64_image = self.encode_image_to_base64(image_path)
                
                formatted_property = {
                    "property_id": property_data.get("id"),
                    "image_data": base64_image,
                    "image_format": image_path.split('.')[-1].lower(),
                    "metadata": {
                        "property_type": property_data.get("property_type"),
                        "listed_rooms": property_data.get("rooms"),
                        "size_sqm": property_data.get("size_sqm")
                    }
                }
                formatted_data.append(formatted_property)
                
            except Exception as e:
                logger.error(f"Failed to prepare floorplan data for property {property_data.get('id')}: {e}")
                continue
        
        return formatted_data
    
    def health_check(self, model_name: str) -> bool:
        """
        Check if KServe model is healthy and ready
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            True if model is healthy, False otherwise
        """
        try:
            url = f"{self.base_url}/v1/models/{model_name}"
            headers = {"Host": "ga-mlops-inference.example.com"}
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            status = response.json()
            return status.get("status", {}).get("state") == "Ready"
            
        except Exception as e:
            logger.error(f"Health check failed for model {model_name}: {e}")
            return False
    
    def get_model_metadata(self, model_name: str) -> Dict[str, Any]:
        """
        Get model metadata from KServe
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model metadata
        """
        try:
            url = f"{self.base_url}/v1/models/{model_name}/metadata"
            headers = {"Host": "ga-mlops-inference.example.com"}
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Failed to get metadata for model {model_name}: {e}")
            return {}

# Integration with existing services
class KServeIntegratedCustomerService:
    def __init__(self):
        self.kserve_client = KServeClient()
    
    def predict_customer_value_kserve(self, customer, interactions) -> Dict[str, Any]:
        """
        Use KServe for customer value prediction instead of local models
        """
        try:
            # Check if KServe is available
            if not self.kserve_client.health_check("customer-value"):
                logger.warning("KServe customer-value model not available, falling back to local prediction")
                # Fall back to local model or heuristics
                return self._fallback_prediction(customer, interactions)
            
            # Prepare data for KServe
            customer_data = [{
                "customer_id": customer.id,
                "age": customer.age,
                "income": customer.income,
                "profession": customer.profession,
                "location": customer.location,
                "family_size": customer.family_size,
                "interactions": [
                    {
                        "type": i.interaction_type,
                        "property_id": i.property_id,
                        "duration": 300  # Mock duration
                    }
                    for i in interactions
                ]
            }]
            
            # Get predictions from KServe
            response = self.kserve_client.predict_customer_value(customer_data)
            
            if response.get("predictions"):
                prediction = response["predictions"][0]
                return {
                    "customer_id": customer.id,
                    "conversion_probability": prediction["conversion_probability"],
                    "estimated_lifetime_value": prediction["estimated_lifetime_value"],
                    "confidence_score": prediction["confidence_score"],
                    "customer_segment": prediction["customer_segment"],
                    "recommended_properties": [],
                    "features": prediction.get("recommendations", [])
                }
            else:
                raise ValueError("No predictions returned from KServe")
                
        except Exception as e:
            logger.error(f"KServe prediction failed: {e}")
            return self._fallback_prediction(customer, interactions)
    
    def _fallback_prediction(self, customer, interactions):
        """Fallback prediction when KServe is unavailable"""
        # Simple heuristic prediction
        age_score = 1.0 if 35 <= customer.age <= 55 else 0.7
        income_score = min(customer.income / 100000.0, 1.0) if customer.income else 0.5
        engagement_score = min(len(interactions) / 10.0, 1.0)
        
        conversion_prob = (age_score * 0.3 + income_score * 0.4 + engagement_score * 0.3)
        estimated_value = customer.income * 0.5 if customer.income else 25000
        
        return {
            "customer_id": customer.id,
            "conversion_probability": conversion_prob,
            "estimated_lifetime_value": estimated_value,
            "confidence_score": 0.6,
            "customer_segment": "fallback_prediction",
            "recommended_properties": [],
            "features": []
        }

class KServeIntegratedFloorplanService:
    def __init__(self):
        self.kserve_client = KServeClient()
    
    def analyze_floorplan_kserve(self, image_path: str, property_id: int) -> Dict[str, Any]:
        """
        Use KServe for floorplan analysis instead of local models
        """
        try:
            # Check if KServe is available
            if not self.kserve_client.health_check("floorplan"):
                logger.warning("KServe floorplan model not available, falling back to local analysis")
                return self._fallback_analysis(image_path)
            
            # Prepare data for KServe
            floorplan_data = [{
                "property_id": property_id,
                "image_data": self.kserve_client.encode_image_to_base64(image_path),
                "image_format": image_path.split('.')[-1].lower(),
                "metadata": {
                    "property_type": "apartment",  # Could be retrieved from database
                    "listed_rooms": 3  # Could be retrieved from database
                }
            }]
            
            # Get predictions from KServe
            response = self.kserve_client.predict_floorplan(floorplan_data)
            
            if response.get("predictions"):
                prediction = response["predictions"][0]
                return {
                    "detected_layout": prediction["detected_layout"],
                    "room_count": prediction["room_count"],
                    "estimated_flow": prediction["estimated_flow"],
                    "accessibility_score": prediction["accessibility_score"],
                    "family_friendliness": prediction["family_friendliness"],
                    "confidence_scores": prediction["confidence_scores"]
                }
            else:
                raise ValueError("No predictions returned from KServe")
                
        except Exception as e:
            logger.error(f"KServe floorplan analysis failed: {e}")
            return self._fallback_analysis(image_path)
    
    def _fallback_analysis(self, image_path: str):
        """Fallback analysis when KServe is unavailable"""
        return {
            "detected_layout": {
                "rooms": [{"type": "unknown", "confidence": 0.1}],
                "connections": [],
                "spatial_arrangement": {"layout_type": "unknown"}
            },
            "room_count": {"unknown": 1},
            "estimated_flow": {
                "traffic_flow_score": 0.5,
                "openness": 0.5,
                "circulation_paths": []
            },
            "accessibility_score": 0.5,
            "family_friendliness": 0.5,
            "confidence_scores": {
                "room_detection": 0.1,
                "layout_analysis": 0.1,
                "overall": 0.1
            }
        }