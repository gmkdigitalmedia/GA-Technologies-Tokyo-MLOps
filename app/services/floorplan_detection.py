import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
from typing import Dict, List, Any, Tuple, Optional
import logging
from sqlalchemy.orm import Session
from app.services.aws_services import S3DataService
from app.models.property import FloorplanAnalysis
import time

logger = logging.getLogger(__name__)

class FloorplanDetectionModel(nn.Module):
    """CNN model for floorplan room detection and layout analysis"""
    
    def __init__(self, num_room_types=11):
        super(FloorplanDetectionModel, self).__init__()
        
        # Feature extraction backbone
        self.backbone = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth conv block
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Room detection head
        self.room_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_room_types)
        )
        
        # Layout analysis head
        self.layout_analyzer = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(512 * 16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 10)  # Layout features
        )
    
    def forward(self, x):
        features = self.backbone(x)
        room_scores = self.room_classifier(features)
        layout_features = self.layout_analyzer(features)
        
        return room_scores, layout_features

class FloorplanDetectionService:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.s3_service = S3DataService()
        
        # Room type mappings
        self.room_types = [
            "bedroom", "living_room", "kitchen", "bathroom", "dining_room",
            "hallway", "closet", "balcony", "study", "storage", "utility"
        ]
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load model if available
        self._load_model()
    
    def _load_model(self):
        """Load pre-trained floorplan detection model"""
        try:
            self.model = FloorplanDetectionModel(num_room_types=len(self.room_types))
            
            # In a real implementation, load weights from S3
            # For now, initialize with random weights
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Floorplan detection model loaded successfully")
            
        except Exception as e:
            logger.warning(f"Could not load pre-trained model: {e}")
            self.model = None
    
    def analyze_floorplan(self, image_path: str) -> Dict[str, Any]:
        """Analyze floorplan image and extract layout information"""
        try:
            if self.model is None:
                return self._fallback_analysis(image_path)
            
            # Load and preprocess image
            if image_path.startswith('s3://'):
                # Download from S3 first
                local_path = self._download_from_s3(image_path)
            else:
                local_path = image_path
            
            image = Image.open(local_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                room_scores, layout_features = self.model(input_tensor)
            
            # Process results
            analysis_result = self._process_model_output(
                image, room_scores, layout_features
            )
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Floorplan analysis failed: {e}")
            return self._fallback_analysis(image_path)
    
    def _download_from_s3(self, s3_path: str) -> str:
        """Download image from S3 to local temp file"""
        import urllib.parse
        
        # Parse S3 path
        parsed = urllib.parse.urlparse(s3_path)
        bucket = parsed.netloc
        key = parsed.path.lstrip('/')
        
        # Download to temp file
        temp_path = f"/tmp/{key.split('/')[-1]}"
        
        try:
            self.s3_service.aws_service.s3.download_file(bucket, key, temp_path)
            return temp_path
        except Exception as e:
            logger.error(f"Failed to download from S3: {e}")
            raise
    
    def _process_model_output(self, image: Image.Image, room_scores: torch.Tensor, 
                            layout_features: torch.Tensor) -> Dict[str, Any]:
        """Process model output to extract meaningful information"""
        
        # Convert tensors to numpy
        room_scores_np = torch.softmax(room_scores, dim=1).cpu().numpy()[0]
        layout_features_np = layout_features.cpu().numpy()[0]
        
        # Detect rooms based on scores
        room_threshold = 0.3
        detected_rooms = []
        room_counts = {}
        
        for i, score in enumerate(room_scores_np):
            if score > room_threshold:
                room_type = self.room_types[i]
                detected_rooms.append({
                    'type': room_type,
                    'confidence': float(score)
                })
                room_counts[room_type] = room_counts.get(room_type, 0) + 1
        
        # Analyze layout features
        layout_analysis = self._analyze_layout_features(layout_features_np)
        
        # Calculate derived metrics
        accessibility_score = self._calculate_accessibility_score(detected_rooms, layout_analysis)
        family_friendliness = self._calculate_family_friendliness(room_counts, layout_analysis)
        
        # Generate confidence scores
        confidence_scores = {
            'room_detection': float(np.mean([r['confidence'] for r in detected_rooms])) if detected_rooms else 0.0,
            'layout_analysis': min(1.0, float(np.mean(np.abs(layout_features_np)))),
            'overall': 0.8  # Mock overall confidence
        }
        
        return {
            'detected_layout': {
                'rooms': detected_rooms,
                'connections': layout_analysis.get('connections', []),
                'spatial_arrangement': layout_analysis.get('spatial_arrangement', {})
            },
            'room_count': room_counts,
            'estimated_flow': layout_analysis.get('flow_analysis', {}),
            'accessibility_score': accessibility_score,
            'family_friendliness': family_friendliness,
            'confidence_scores': confidence_scores
        }
    
    def _analyze_layout_features(self, layout_features: np.ndarray) -> Dict[str, Any]:
        """Analyze layout features from model output"""
        
        # Mock analysis based on feature values
        # In a real implementation, these would be learned mappings
        
        flow_score = np.clip(layout_features[0] * 0.5 + 0.5, 0, 1)
        openness_score = np.clip(layout_features[1] * 0.5 + 0.5, 0, 1)
        connectivity_score = np.clip(layout_features[2] * 0.5 + 0.5, 0, 1)
        
        return {
            'flow_analysis': {
                'traffic_flow_score': float(flow_score),
                'openness': float(openness_score),
                'bottlenecks': ['hallway'] if flow_score < 0.4 else [],
                'circulation_paths': ['main_corridor', 'secondary_path']
            },
            'connections': [
                {'from': 'living_room', 'to': 'kitchen', 'type': 'open'},
                {'from': 'hallway', 'to': 'bedroom', 'type': 'door'}
            ],
            'spatial_arrangement': {
                'layout_type': 'linear' if connectivity_score < 0.4 else 'open_plan',
                'room_clustering': 'private_public_separation'
            }
        }
    
    def _calculate_accessibility_score(self, detected_rooms: List[Dict], 
                                     layout_analysis: Dict[str, Any]) -> float:
        """Calculate accessibility score based on layout"""
        
        base_score = 0.7
        
        # Bonus for single-level layout (assumed for floorplans)
        accessibility_score = base_score + 0.1
        
        # Penalty for narrow passages
        flow_score = layout_analysis.get('flow_analysis', {}).get('traffic_flow_score', 0.5)
        if flow_score < 0.4:
            accessibility_score -= 0.2
        
        # Bonus for bathroom accessibility
        bathrooms = [r for r in detected_rooms if r['type'] == 'bathroom']
        if len(bathrooms) > 0:
            accessibility_score += 0.1
        
        return max(0.0, min(1.0, accessibility_score))
    
    def _calculate_family_friendliness(self, room_counts: Dict[str, int], 
                                     layout_analysis: Dict[str, Any]) -> float:
        """Calculate family friendliness score"""
        
        base_score = 0.6
        
        # Bonus for multiple bedrooms
        bedrooms = room_counts.get('bedroom', 0)
        if bedrooms >= 2:
            base_score += 0.2
        elif bedrooms >= 3:
            base_score += 0.3
        
        # Bonus for family areas
        if room_counts.get('living_room', 0) > 0:
            base_score += 0.1
        
        if room_counts.get('dining_room', 0) > 0:
            base_score += 0.1
        
        # Bonus for storage
        storage_rooms = room_counts.get('storage', 0) + room_counts.get('closet', 0)
        if storage_rooms > 0:
            base_score += 0.1
        
        return max(0.0, min(1.0, base_score))
    
    def _fallback_analysis(self, image_path: str) -> Dict[str, Any]:
        """Fallback analysis using traditional computer vision"""
        try:
            # Load image with OpenCV
            if image_path.startswith('s3://'):
                local_path = self._download_from_s3(image_path)
            else:
                local_path = image_path
            
            image = cv2.imread(local_path)
            if image is None:
                raise ValueError("Could not load image")
            
            # Simple room detection using contours and heuristics
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Estimate number of rooms based on contours
            large_contours = [c for c in contours if cv2.contourArea(c) > 1000]
            estimated_rooms = min(len(large_contours), 8)  # Cap at reasonable number
            
            # Mock room distribution
            room_types_detected = ['living_room', 'bedroom', 'kitchen', 'bathroom'][:estimated_rooms]
            
            detected_rooms = [
                {'type': room_type, 'confidence': 0.6}
                for room_type in room_types_detected
            ]
            
            room_counts = {room_type: 1 for room_type in room_types_detected}
            
            return {
                'detected_layout': {
                    'rooms': detected_rooms,
                    'connections': [],
                    'spatial_arrangement': {'layout_type': 'traditional'}
                },
                'room_count': room_counts,
                'estimated_flow': {
                    'traffic_flow_score': 0.6,
                    'openness': 0.5,
                    'circulation_paths': ['main_corridor']
                },
                'accessibility_score': 0.7,
                'family_friendliness': 0.6,
                'confidence_scores': {
                    'room_detection': 0.6,
                    'layout_analysis': 0.5,
                    'overall': 0.55
                }
            }
            
        except Exception as e:
            logger.error(f"Fallback analysis failed: {e}")
            return self._default_analysis()
    
    def _default_analysis(self) -> Dict[str, Any]:
        """Return default analysis when all else fails"""
        return {
            'detected_layout': {
                'rooms': [{'type': 'unknown', 'confidence': 0.1}],
                'connections': [],
                'spatial_arrangement': {'layout_type': 'unknown'}
            },
            'room_count': {'unknown': 1},
            'estimated_flow': {
                'traffic_flow_score': 0.5,
                'openness': 0.5,
                'circulation_paths': []
            },
            'accessibility_score': 0.5,
            'family_friendliness': 0.5,
            'confidence_scores': {
                'room_detection': 0.1,
                'layout_analysis': 0.1,
                'overall': 0.1
            }
        }
    
    def batch_analyze(self, property_ids: List[int], db: Session):
        """Batch analyze floorplans for multiple properties"""
        try:
            for property_id in property_ids:
                # Get property floorplan path from database
                # This would typically query the property table
                mock_image_path = f"s3://gp-mlops-mlops/floorplans/{property_id}.jpg"
                
                try:
                    # Perform analysis
                    analysis_result = self.analyze_floorplan(mock_image_path)
                    
                    # Store results
                    db_analysis = FloorplanAnalysis(
                        property_id=property_id,
                        image_path=mock_image_path,
                        detected_layout=analysis_result["detected_layout"],
                        room_count=analysis_result["room_count"],
                        estimated_flow=analysis_result["estimated_flow"],
                        accessibility_score=analysis_result["accessibility_score"],
                        family_friendliness=analysis_result["family_friendliness"],
                        model_version="v1.0",
                        confidence_scores=analysis_result["confidence_scores"]
                    )
                    
                    db.add(db_analysis)
                    
                except Exception as e:
                    logger.error(f"Failed to analyze property {property_id}: {e}")
                    continue
            
            db.commit()
            logger.info(f"Batch analysis completed for {len(property_ids)} properties")
            
        except Exception as e:
            logger.error(f"Batch analysis failed: {e}")
            db.rollback()
            raise