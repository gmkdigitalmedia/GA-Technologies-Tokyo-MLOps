import json
import base64
import numpy as np
import cv2
from PIL import Image
import io
from typing import Dict, List, Any, Union
import logging
import os
from kserve import Model, ModelServer
import torch
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)

class FloorplanTransformer(Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.ready = False
        self.predictor_host = os.environ.get("PREDICTOR_HOST", "floorplan-detection-model-predictor")
        
        # Room type mappings
        self.room_types = [
            "bedroom", "living_room", "kitchen", "bathroom", "dining_room",
            "hallway", "closet", "balcony", "study", "storage", "utility"
        ]
        
        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

    def load(self):
        """Load transformer components"""
        self.ready = True
        logger.info(f"FloorplanTransformer {self.name} loaded successfully")

    def preprocess(self, inputs: Dict[str, Any], headers: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Transform input images for floorplan detection
        
        Expected input format:
        {
            "instances": [
                {
                    "property_id": 12345,
                    "image_data": "base64_encoded_image",
                    "image_format": "jpg",
                    "metadata": {
                        "property_type": "apartment",
                        "listed_rooms": 3
                    }
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
                # Decode image
                image_tensor = self._decode_and_preprocess_image(instance)
                
                # Add metadata for context
                metadata = instance.get("metadata", {})
                
                # Create input for PyTorch model
                model_input = {
                    "image": image_tensor.tolist(),
                    "property_id": instance.get("property_id"),
                    "metadata": metadata
                }
                
                processed_instances.append(model_input)
            
            return {"instances": processed_instances}
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise
    
    def postprocess(self, inputs: Dict[str, Any], headers: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Transform model output to structured floorplan analysis
        
        Model output: {"predictions": [[room_scores, layout_features], ...]}
        Business output: Detailed floorplan analysis
        """
        try:
            predictions = inputs.get("predictions", [])
            if not predictions:
                return {"predictions": []}
            
            processed_predictions = []
            
            for i, prediction in enumerate(predictions):
                if len(prediction) >= 2:
                    room_scores = np.array(prediction[0])
                    layout_features = np.array(prediction[1])
                    
                    # Process room detection
                    room_analysis = self._analyze_rooms(room_scores)
                    
                    # Process layout features
                    layout_analysis = self._analyze_layout(layout_features)
                    
                    # Calculate derived metrics
                    accessibility_score = self._calculate_accessibility(
                        room_analysis, layout_analysis
                    )
                    family_friendliness = self._calculate_family_friendliness(
                        room_analysis, layout_analysis
                    )
                    
                    # Generate recommendations
                    recommendations = self._generate_recommendations(
                        room_analysis, layout_analysis, accessibility_score, family_friendliness
                    )
                    
                    processed_prediction = {
                        "property_index": i,
                        "detected_layout": {
                            "rooms": room_analysis["detected_rooms"],
                            "connections": layout_analysis["connections"],
                            "spatial_arrangement": layout_analysis["spatial_arrangement"]
                        },
                        "room_count": room_analysis["room_counts"],
                        "estimated_flow": layout_analysis["flow_analysis"],
                        "accessibility_score": round(accessibility_score, 3),
                        "family_friendliness": round(family_friendliness, 3),
                        "confidence_scores": {
                            "room_detection": round(room_analysis["confidence"], 3),
                            "layout_analysis": round(layout_analysis["confidence"], 3),
                            "overall": round((room_analysis["confidence"] + layout_analysis["confidence"]) / 2, 3)
                        },
                        "recommendations": recommendations,
                        "model_version": "v1.0",
                        "analysis_timestamp": pd.Timestamp.now().isoformat()
                    }
                    
                    processed_predictions.append(processed_prediction)
                else:
                    logger.warning(f"Invalid prediction format at index {i}")
                    processed_predictions.append({
                        "property_index": i,
                        "error": "Invalid prediction format"
                    })
            
            return {"predictions": processed_predictions}
            
        except Exception as e:
            logger.error(f"Postprocessing failed: {e}")
            raise
    
    def _decode_and_preprocess_image(self, instance: Dict[str, Any]) -> torch.Tensor:
        """Decode base64 image and preprocess for model"""
        try:
            image_data = instance.get("image_data")
            if not image_data:
                raise ValueError("No image data provided")
            
            # Decode base64
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # Apply transforms
            image_tensor = self.transform(image)
            
            return image_tensor
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            raise
    
    def _analyze_rooms(self, room_scores: np.ndarray) -> Dict[str, Any]:
        """Analyze room detection scores"""
        # Apply softmax to get probabilities
        room_probs = self._softmax(room_scores)
        
        # Detect rooms above threshold
        threshold = 0.3
        detected_rooms = []
        room_counts = {}
        
        for i, prob in enumerate(room_probs):
            if prob > threshold:
                room_type = self.room_types[i]
                detected_rooms.append({
                    "type": room_type,
                    "confidence": float(prob),
                    "area_percentage": min(prob * 100, 100)  # Rough area estimation
                })
                room_counts[room_type] = room_counts.get(room_type, 0) + 1
        
        # Calculate overall confidence
        confidence = float(np.mean([r["confidence"] for r in detected_rooms])) if detected_rooms else 0.1
        
        return {
            "detected_rooms": detected_rooms,
            "room_counts": room_counts,
            "confidence": confidence
        }
    
    def _analyze_layout(self, layout_features: np.ndarray) -> Dict[str, Any]:
        """Analyze layout features from model"""
        # Extract layout metrics from features
        # These mappings would be learned during training
        flow_score = self._sigmoid(layout_features[0])
        openness_score = self._sigmoid(layout_features[1])
        connectivity_score = self._sigmoid(layout_features[2])
        natural_light_score = self._sigmoid(layout_features[3])
        
        # Generate layout analysis
        flow_analysis = {
            "traffic_flow_score": float(flow_score),
            "openness": float(openness_score),
            "connectivity": float(connectivity_score),
            "natural_light_access": float(natural_light_score),
            "bottlenecks": ["hallway"] if flow_score < 0.4 else [],
            "circulation_efficiency": "good" if flow_score > 0.6 else "needs_improvement"
        }
        
        # Determine connections (simplified)
        connections = self._infer_connections(connectivity_score)
        
        # Spatial arrangement
        spatial_arrangement = {
            "layout_type": "open_plan" if openness_score > 0.6 else "traditional",
            "room_clustering": "efficient" if connectivity_score > 0.5 else "compartmentalized",
            "flow_pattern": "circular" if flow_score > 0.7 else "linear"
        }
        
        confidence = float(np.mean([flow_score, openness_score, connectivity_score]))
        
        return {
            "flow_analysis": flow_analysis,
            "connections": connections,
            "spatial_arrangement": spatial_arrangement,
            "confidence": confidence
        }
    
    def _calculate_accessibility(self, room_analysis: Dict, layout_analysis: Dict) -> float:
        """Calculate accessibility score"""
        base_score = 0.7
        
        # Bonus for good flow
        flow_score = layout_analysis["flow_analysis"]["traffic_flow_score"]
        if flow_score > 0.7:
            base_score += 0.2
        elif flow_score < 0.4:
            base_score -= 0.2
        
        # Bonus for bathroom accessibility
        bathrooms = [r for r in room_analysis["detected_rooms"] if r["type"] == "bathroom"]
        if bathrooms:
            base_score += 0.1
        
        # Penalty for too many levels (not applicable for floorplans)
        
        return max(0.0, min(1.0, base_score))
    
    def _calculate_family_friendliness(self, room_analysis: Dict, layout_analysis: Dict) -> float:
        """Calculate family friendliness score"""
        base_score = 0.6
        room_counts = room_analysis["room_counts"]
        
        # Bonus for bedrooms
        bedrooms = room_counts.get("bedroom", 0)
        if bedrooms >= 2:
            base_score += 0.2
        if bedrooms >= 3:
            base_score += 0.1
        
        # Bonus for family areas
        if room_counts.get("living_room", 0) > 0:
            base_score += 0.1
        if room_counts.get("dining_room", 0) > 0:
            base_score += 0.1
        
        # Bonus for storage
        storage_count = room_counts.get("storage", 0) + room_counts.get("closet", 0)
        if storage_count > 0:
            base_score += 0.1
        
        # Bonus for good flow (kids need safe circulation)
        flow_score = layout_analysis["flow_analysis"]["traffic_flow_score"]
        if flow_score > 0.7:
            base_score += 0.1
        
        return max(0.0, min(1.0, base_score))
    
    def _generate_recommendations(self, room_analysis: Dict, layout_analysis: Dict, 
                                accessibility_score: float, family_friendliness: float) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        room_counts = room_analysis["room_counts"]
        flow_analysis = layout_analysis["flow_analysis"]
        
        # Room-based recommendations
        if room_counts.get("bedroom", 0) >= 3:
            recommendations.append("suitable_for_families")
        if room_counts.get("study", 0) > 0:
            recommendations.append("good_for_remote_work")
        if room_counts.get("storage", 0) + room_counts.get("closet", 0) >= 2:
            recommendations.append("excellent_storage_space")
        
        # Layout recommendations
        if flow_analysis["traffic_flow_score"] > 0.7:
            recommendations.append("excellent_flow_design")
        elif flow_analysis["traffic_flow_score"] < 0.4:
            recommendations.append("consider_furniture_placement")
        
        if flow_analysis["openness"] > 0.6:
            recommendations.append("open_spacious_feel")
        
        # Accessibility recommendations
        if accessibility_score > 0.8:
            recommendations.append("wheelchair_accessible")
        elif accessibility_score < 0.5:
            recommendations.append("accessibility_improvements_needed")
        
        # Family recommendations
        if family_friendliness > 0.8:
            recommendations.append("ideal_for_families_with_children")
        
        return recommendations
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Apply softmax function"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def _sigmoid(self, x: float) -> float:
        """Apply sigmoid function"""
        return 1.0 / (1.0 + np.exp(-x))
    
    def _infer_connections(self, connectivity_score: float) -> List[Dict[str, str]]:
        """Infer room connections based on connectivity score"""
        # Simplified connection inference
        connections = []
        
        if connectivity_score > 0.7:
            connections.extend([
                {"from": "living_room", "to": "kitchen", "type": "open"},
                {"from": "living_room", "to": "dining_room", "type": "open"},
                {"from": "hallway", "to": "bedroom", "type": "door"}
            ])
        elif connectivity_score > 0.4:
            connections.extend([
                {"from": "living_room", "to": "kitchen", "type": "doorway"},
                {"from": "hallway", "to": "bedroom", "type": "door"}
            ])
        else:
            connections.extend([
                {"from": "hallway", "to": "bedroom", "type": "door"},
                {"from": "hallway", "to": "bathroom", "type": "door"}
            ])
        
        return connections

if __name__ == "__main__":
    import pandas as pd
    model = FloorplanTransformer("floorplan-transformer")
    model.load()
    ModelServer().start([model])