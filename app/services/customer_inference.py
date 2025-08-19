import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import joblib
import logging
from datetime import datetime, timedelta
import json

from app.services.snowflake_service import SnowflakeService
from app.services.aws_services import S3DataService, SageMakerService
from app.models.customer import Customer, CustomerInteraction
from app.core.config import settings

logger = logging.getLogger(__name__)

class CustomerInferenceService:
    def __init__(self):
        self.snowflake_service = SnowflakeService()
        self.s3_service = S3DataService()
        self.sagemaker_service = SageMakerService()
        
        # Model components
        self.conversion_model = None
        self.value_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
        # Load models if they exist
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models from S3"""
        try:
            # Try to download and load models
            conversion_model_path = f"{settings.S3_MODEL_PREFIX}conversion_model.pkl"
            value_model_path = f"{settings.S3_MODEL_PREFIX}value_model.pkl"
            scaler_path = f"{settings.S3_MODEL_PREFIX}scaler.pkl"
            
            # Download from S3 (implement S3 download logic)
            # For now, will train fresh models if not available
            logger.info("Models will be trained fresh or loaded from local storage")
            
        except Exception as e:
            logger.info(f"Models not found, will train fresh: {e}")
    
    def prepare_training_data(self) -> pd.DataFrame:
        """Prepare training data from Snowflake"""
        try:
            # Get conversion data
            conversion_data = self.snowflake_service.get_conversion_data()
            
            # Get interaction data for feature engineering
            interaction_data = self.snowflake_service.get_property_interaction_data(days_back=90)
            
            # Feature engineering
            features_df = self._engineer_features(conversion_data, interaction_data)
            
            return features_df
            
        except Exception as e:
            logger.error(f"Failed to prepare training data: {e}")
            raise
    
    def _engineer_features(self, customer_data: pd.DataFrame, interaction_data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for ML models"""
        try:
            # Aggregate interaction features by customer
            interaction_features = interaction_data.groupby('customer_id').agg({
                'interaction_type': ['count', 'nunique'],
                'session_duration': ['mean', 'sum', 'std'],
                'property_id': 'nunique',
                'price': ['mean', 'std'],
                'size_sqm': ['mean', 'std']
            }).round(2)
            
            # Flatten column names
            interaction_features.columns = ['_'.join(col) for col in interaction_features.columns]
            interaction_features = interaction_features.fillna(0)
            
            # Merge with customer data
            features_df = customer_data.merge(
                interaction_features, 
                left_on='customer_id', 
                right_index=True, 
                how='left'
            )
            
            # Fill missing values
            features_df = features_df.fillna(0)
            
            # Create additional features
            features_df['age_group'] = pd.cut(features_df['age'], 
                                           bins=[0, 30, 40, 50, 60, 100], 
                                           labels=['young', 'early_career', 'mid_career', 'senior', 'retirement'])
            
            features_df['income_tier'] = pd.qcut(features_df['income'], 
                                               q=5, 
                                               labels=['low', 'low_mid', 'mid', 'high_mid', 'high'])
            
            features_df['engagement_score'] = (
                features_df['interaction_type_count'] * 0.3 +
                features_df['property_id_nunique'] * 0.4 +
                features_df['session_duration_mean'] * 0.3
            )
            
            return features_df
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            raise
    
    def train_conversion_model(self, training_data: pd.DataFrame) -> Dict[str, float]:
        """Train conversion probability model"""
        try:
            # Prepare features and target
            feature_columns = [
                'age', 'income', 'family_size', 'total_interactions',
                'unique_properties_viewed', 'avg_session_duration',
                'engagement_score'
            ]
            
            # Handle categorical features
            categorical_features = ['profession', 'location', 'age_group', 'income_tier']
            
            # Encode categorical features
            for cat_feature in categorical_features:
                if cat_feature in training_data.columns:
                    le = LabelEncoder()
                    training_data[f'{cat_feature}_encoded'] = le.fit_transform(
                        training_data[cat_feature].astype(str)
                    )
                    self.label_encoders[cat_feature] = le
                    feature_columns.append(f'{cat_feature}_encoded')
            
            X = training_data[feature_columns].fillna(0)
            y = training_data['converted']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.conversion_model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            
            self.conversion_model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = self.conversion_model.predict(X_test_scaled)
            y_pred_proba = self.conversion_model.predict_proba(X_test_scaled)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"Conversion model trained with accuracy: {accuracy:.4f}")
            
            return {
                'accuracy': accuracy,
                'feature_importance': dict(zip(feature_columns, self.conversion_model.feature_importances_))
            }
            
        except Exception as e:
            logger.error(f"Conversion model training failed: {e}")
            raise
    
    def train_value_model(self, training_data: pd.DataFrame) -> Dict[str, float]:
        """Train lifetime value prediction model"""
        try:
            # Prepare features and target
            feature_columns = [
                'age', 'income', 'family_size', 'total_interactions',
                'unique_properties_viewed', 'avg_session_duration',
                'engagement_score'
            ]
            
            # Add categorical features
            categorical_features = ['profession', 'location', 'age_group', 'income_tier']
            for cat_feature in categorical_features:
                if f'{cat_feature}_encoded' in training_data.columns:
                    feature_columns.append(f'{cat_feature}_encoded')
            
            # Filter for converted customers only for value prediction
            converted_data = training_data[training_data['converted'] == 1].copy()
            
            if len(converted_data) < 10:
                logger.warning("Insufficient converted customers for value model training")
                return {'r2_score': 0.0, 'mse': float('inf')}
            
            X = converted_data[feature_columns].fillna(0)
            y = converted_data['conversion_value']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features (using same scaler as conversion model)
            X_train_scaled = self.scaler.transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.value_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            self.value_model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = self.value_model.predict(X_test_scaled)
            
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            logger.info(f"Value model trained with R² score: {r2:.4f}")
            
            return {
                'r2_score': r2,
                'mse': mse,
                'feature_importance': dict(zip(feature_columns, self.value_model.feature_importances_))
            }
            
        except Exception as e:
            logger.error(f"Value model training failed: {e}")
            raise
    
    def predict_customer_value(self, customer: Customer, interactions: List[CustomerInteraction]) -> Dict[str, Any]:
        """Predict customer conversion probability and lifetime value"""
        try:
            # Extract features from customer and interactions
            features = self._extract_customer_features(customer, interactions)
            
            # If models are not trained, use simple heuristics
            if self.conversion_model is None or self.value_model is None:
                return self._heuristic_prediction(customer, interactions)
            
            # Prepare features for prediction
            feature_array = self._prepare_prediction_features(features)
            
            # Predict conversion probability
            conversion_prob = self.conversion_model.predict_proba([feature_array])[0][1]
            
            # Predict lifetime value
            estimated_value = self.value_model.predict([feature_array])[0]
            
            # Calculate confidence based on feature completeness and model certainty
            confidence = min(
                0.9,  # Max confidence
                conversion_prob * 0.5 + 0.3 + (len(interactions) / 100.0) * 0.2
            )
            
            # Get property recommendations
            recommended_properties = self._get_property_recommendations(customer, interactions)
            
            return {
                'customer_id': customer.id,
                'conversion_probability': float(conversion_prob),
                'estimated_lifetime_value': float(estimated_value),
                'confidence_score': float(confidence),
                'recommended_properties': recommended_properties,
                'features': features
            }
            
        except Exception as e:
            logger.error(f"Customer prediction failed: {e}")
            raise
    
    def _extract_customer_features(self, customer: Customer, interactions: List[CustomerInteraction]) -> Dict[str, Any]:
        """Extract features from customer and interaction data"""
        
        # Basic customer features
        features = {
            'age': customer.age or 40,  # Default if missing
            'income': customer.income or 50000,
            'family_size': customer.family_size or 1,
            'profession': customer.profession or 'unknown',
            'location': customer.location or 'unknown'
        }
        
        # Interaction features
        if interactions:
            features.update({
                'total_interactions': len(interactions),
                'unique_properties_viewed': len(set(i.property_id for i in interactions if i.property_id)),
                'recent_activity_days': min(
                    (datetime.now() - max(i.timestamp for i in interactions)).days, 
                    30
                ) if interactions else 30
            })
        else:
            features.update({
                'total_interactions': 0,
                'unique_properties_viewed': 0,
                'recent_activity_days': 30
            })
        
        # Derived features
        features['engagement_score'] = min(
            features['total_interactions'] * 0.3 + 
            features['unique_properties_viewed'] * 0.7,
            100
        )
        
        return features
    
    def _prepare_prediction_features(self, features: Dict[str, Any]) -> np.ndarray:
        """Prepare features for model prediction"""
        
        # Create feature array in same order as training
        feature_array = [
            features['age'],
            features['income'],
            features['family_size'],
            features['total_interactions'],
            features['unique_properties_viewed'],
            features.get('avg_session_duration', 5.0),  # Default 5 minutes
            features['engagement_score']
        ]
        
        # Add encoded categorical features
        for cat_feature in ['profession', 'location', 'age_group', 'income_tier']:
            if cat_feature in self.label_encoders:
                try:
                    if cat_feature in features:
                        encoded_value = self.label_encoders[cat_feature].transform([features[cat_feature]])[0]
                    else:
                        encoded_value = 0  # Default encoding
                    feature_array.append(encoded_value)
                except ValueError:
                    # Unknown category, use default
                    feature_array.append(0)
        
        return np.array(feature_array)
    
    def _heuristic_prediction(self, customer: Customer, interactions: List[CustomerInteraction]) -> Dict[str, Any]:
        """Simple heuristic prediction when models are not available"""
        
        # Simple scoring based on customer profile
        age_score = 1.0 if 35 <= customer.age <= 55 else 0.7
        income_score = min(customer.income / 100000.0, 1.0) if customer.income else 0.5
        engagement_score = min(len(interactions) / 10.0, 1.0)
        
        conversion_prob = (age_score * 0.3 + income_score * 0.4 + engagement_score * 0.3)
        estimated_value = customer.income * 0.5 if customer.income else 25000
        
        return {
            'customer_id': customer.id,
            'conversion_probability': conversion_prob,
            'estimated_lifetime_value': estimated_value,
            'confidence_score': 0.6,  # Lower confidence for heuristics
            'recommended_properties': [],
            'features': self._extract_customer_features(customer, interactions)
        }
    
    def _get_property_recommendations(self, customer: Customer, interactions: List[CustomerInteraction]) -> List[int]:
        """Get property recommendations based on customer profile"""
        # Simplified recommendation logic
        # In production, this would use a separate recommendation model
        
        try:
            # Get properties from Snowflake based on customer preferences
            property_data = self.snowflake_service.get_property_features()
            
            # Simple filtering based on customer profile
            if customer.income:
                max_price = customer.income * 3  # 3x income rule
                filtered_properties = property_data[
                    property_data['price'] <= max_price
                ].head(5)
            else:
                filtered_properties = property_data.head(5)
            
            return filtered_properties['property_id'].tolist()
            
        except Exception as e:
            logger.error(f"Property recommendation failed: {e}")
            return []
    
    def save_models_to_s3(self):
        """Save trained models to S3"""
        try:
            if self.conversion_model:
                # Save locally first
                joblib.dump(self.conversion_model, '/tmp/conversion_model.pkl')
                joblib.dump(self.value_model, '/tmp/value_model.pkl')
                joblib.dump(self.scaler, '/tmp/scaler.pkl')
                joblib.dump(self.label_encoders, '/tmp/label_encoders.pkl')
                
                # Upload to S3
                self.s3_service.upload_model_artifact('/tmp/conversion_model.pkl', 'conversion_model.pkl')
                self.s3_service.upload_model_artifact('/tmp/value_model.pkl', 'value_model.pkl')
                self.s3_service.upload_model_artifact('/tmp/scaler.pkl', 'scaler.pkl')
                self.s3_service.upload_model_artifact('/tmp/label_encoders.pkl', 'label_encoders.pkl')
                
                logger.info("Models saved to S3 successfully")
                
        except Exception as e:
            logger.error(f"Failed to save models to S3: {e}")
            raise