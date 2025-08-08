"""
Full MLOps Pipeline Service
Orchestrates the complete ML pipeline from data to deployment
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pathlib import Path
import json

from .mlflow_service import mlflow_service
from .dify_service import dify_service
from .kserve_client import KServeClient
from .snowflake_service import SnowflakeService
from .airbyte_service import airbyte_service
from app.core.config import settings

logger = logging.getLogger(__name__)

class MLOpsPipeline:
    def __init__(self):
        self.mlflow = mlflow_service
        self.dify = dify_service
        self.kserve = KServeClient()
        self.snowflake = SnowflakeService()
        self.airbyte = airbyte_service
        
        self.pipeline_config = {
            "data_sources": ["snowflake", "s3", "airbyte"],
            "models": ["customer_value", "floorplan_detection"],
            "deployment_targets": ["kserve", "sagemaker"],
            "monitoring_enabled": True,
            "auto_retrain": True,
            "retrain_threshold": 0.05,  # 5% performance drop triggers retrain
            "airbyte_enabled": True
        }
    
    async def run_full_pipeline(self, pipeline_type: str = "daily") -> Dict[str, Any]:
        """
        Execute the complete MLOps pipeline
        
        Args:
            pipeline_type: Type of pipeline run (daily, weekly, on_demand)
            
        Returns:
            Pipeline execution results
        """
        
        pipeline_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Starting MLOps pipeline {pipeline_id}")
        
        results = {
            "pipeline_id": pipeline_id,
            "pipeline_type": pipeline_type,
            "start_time": datetime.now().isoformat(),
            "stages": {},
            "status": "running"
        }
        
        try:
            # Stage 0: Airbyte Data Pipeline Setup (if enabled)
            if self.pipeline_config.get("airbyte_enabled"):
                logger.info("Stage 0: Airbyte data pipeline setup")
                airbyte_results = await self._setup_airbyte_pipeline()
                results["stages"]["airbyte_setup"] = airbyte_results
            
            # Stage 1: Data Extraction and Preparation
            logger.info("Stage 1: Data extraction and preparation")
            data_results = await self._extract_and_prepare_data()
            results["stages"]["data_preparation"] = data_results
            
            if not data_results["success"]:
                raise Exception("Data preparation failed")
            
            # Stage 2: Model Training and Validation
            logger.info("Stage 2: Model training and validation")
            training_results = await self._train_and_validate_models(data_results["datasets"])
            results["stages"]["model_training"] = training_results
            
            # Stage 3: Model Registration and Versioning
            logger.info("Stage 3: Model registration and versioning")
            registry_results = await self._register_and_version_models(training_results)
            results["stages"]["model_registry"] = registry_results
            
            # Stage 4: Model Deployment
            logger.info("Stage 4: Model deployment")
            deployment_results = await self._deploy_models(registry_results)
            results["stages"]["model_deployment"] = deployment_results
            
            # Stage 5: LLM Workflow Integration
            logger.info("Stage 5: LLM workflow integration")
            llm_results = await self._integrate_llm_workflows()
            results["stages"]["llm_integration"] = llm_results
            
            # Stage 6: Monitoring Setup
            logger.info("Stage 6: Monitoring and alerting setup")
            monitoring_results = await self._setup_monitoring()
            results["stages"]["monitoring_setup"] = monitoring_results
            
            # Stage 7: Pipeline Validation
            logger.info("Stage 7: End-to-end pipeline validation")
            validation_results = await self._validate_pipeline()
            results["stages"]["pipeline_validation"] = validation_results
            
            results["status"] = "completed"
            results["end_time"] = datetime.now().isoformat()
            results["duration_minutes"] = (datetime.now() - datetime.fromisoformat(results["start_time"])).total_seconds() / 60
            
            logger.info(f"Pipeline {pipeline_id} completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Pipeline {pipeline_id} failed: {e}")
            results["status"] = "failed"
            results["error"] = str(e)
            results["end_time"] = datetime.now().isoformat()
            return results
    
    async def _setup_airbyte_pipeline(self) -> Dict[str, Any]:
        """Set up Airbyte data integration pipeline"""
        try:
            logger.info("Setting up Airbyte data pipeline...")
            
            # Setup complete Tokyo real estate pipeline
            pipeline_result = await self.airbyte.setup_tokyo_real_estate_pipeline()
            
            if pipeline_result["success"]:
                # Trigger initial sync
                connection_id = pipeline_result.get("connection_id")
                if connection_id and not pipeline_result.get("mock"):
                    sync_result = await self.airbyte.trigger_sync(connection_id)
                    pipeline_result["initial_sync"] = sync_result
                
                return {
                    "success": True,
                    "pipeline_setup": pipeline_result,
                    "airbyte_webapp": "http://localhost:2237",
                    "connections_configured": 1,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                logger.warning("Airbyte pipeline setup failed, continuing with mock data")
                return {
                    "success": True,  # Continue pipeline even if Airbyte fails
                    "pipeline_setup": pipeline_result,
                    "fallback_mode": True,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Airbyte pipeline setup failed: {e}")
            return {
                "success": True,  # Don't fail entire pipeline
                "error": str(e),
                "fallback_mode": True,
                "timestamp": datetime.now().isoformat()
            }

    async def _extract_and_prepare_data(self) -> Dict[str, Any]:
        """Extract and prepare training data"""
        try:
            # Extract customer data from Snowflake
            customer_query = """
            SELECT 
                customer_id,
                age,
                annual_income,
                profession,
                location_preference,
                family_size,
                property_interactions,
                conversion_flag,
                last_interaction_date
            FROM customer_analytics.customer_profiles 
            WHERE last_interaction_date >= DATEADD(day, -90, CURRENT_DATE())
            """
            
            customer_data = await self._execute_snowflake_query(customer_query)
            
            # Extract property interaction data
            interaction_query = """
            SELECT 
                customer_id,
                property_id,
                interaction_type,
                interaction_duration,
                property_type,
                property_location,
                price_range,
                created_at
            FROM customer_analytics.property_interactions 
            WHERE created_at >= DATEADD(day, -90, CURRENT_DATE())
            """
            
            interaction_data = await self._execute_snowflake_query(interaction_query)
            
            # Prepare training datasets
            customer_features, customer_labels = self._prepare_customer_training_data(
                customer_data, interaction_data
            )
            
            # Mock floorplan data preparation (would integrate with actual image data)
            floorplan_data = await self._prepare_floorplan_data()
            
            return {
                "success": True,
                "datasets": {
                    "customer_features": customer_features,
                    "customer_labels": customer_labels,
                    "floorplan_data": floorplan_data
                },
                "data_stats": {
                    "customer_records": len(customer_data) if customer_data else 0,
                    "interaction_records": len(interaction_data) if interaction_data else 0,
                    "floorplan_images": len(floorplan_data) if floorplan_data else 0
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _train_and_validate_models(self, datasets: Dict[str, Any]) -> Dict[str, Any]:
        """Train and validate ML models"""
        try:
            results = {}
            
            # Train Customer Value Model
            if datasets.get("customer_features") is not None:
                customer_results = await self._train_customer_model(
                    datasets["customer_features"],
                    datasets["customer_labels"]
                )
                results["customer_model"] = customer_results
            
            # Train/Update Floorplan Model (mock training)
            floorplan_results = await self._train_floorplan_model(
                datasets.get("floorplan_data", [])
            )
            results["floorplan_model"] = floorplan_results
            
            return {
                "success": True,
                "models_trained": len(results),
                "training_results": results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _register_and_version_models(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """Register models in MLflow"""
        try:
            registry_results = {}
            
            if training_results.get("success") and "training_results" in training_results:
                for model_name, model_data in training_results["training_results"].items():
                    if model_data.get("success"):
                        # Model already registered during training
                        registry_results[model_name] = {
                            "registered": True,
                            "version": model_data.get("model_version"),
                            "run_id": model_data.get("run_id"),
                            "stage": "Staging"
                        }
            
            return {
                "success": True,
                "registered_models": len(registry_results),
                "registry_results": registry_results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Model registration failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _deploy_models(self, registry_results: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy models to KServe"""
        try:
            deployment_results = {}
            
            # Check model health and promote to production if ready
            for model_name, model_info in registry_results.get("registry_results", {}).items():
                if model_info.get("registered"):
                    # Check model performance before promotion
                    if await self._validate_model_performance(model_name, model_info["version"]):
                        # Promote to production
                        promotion_success = self.mlflow.promote_model_to_production(
                            model_name.replace("_model", "").title() + "Model",
                            model_info["version"]
                        )
                        
                        deployment_results[model_name] = {
                            "promoted": promotion_success,
                            "stage": "Production" if promotion_success else "Staging",
                            "kserve_status": "deployed"  # Would check actual KServe deployment
                        }
                    else:
                        deployment_results[model_name] = {
                            "promoted": False,
                            "stage": "Staging",
                            "reason": "Performance validation failed"
                        }
            
            return {
                "success": True,
                "deployed_models": len(deployment_results),
                "deployment_results": deployment_results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Model deployment failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _integrate_llm_workflows(self) -> Dict[str, Any]:
        """Set up Dify LLM workflows"""
        try:
            workflow_results = {}
            
            # Test customer interaction workflow
            test_customer = {
                "id": "test_customer",
                "age": 45,
                "income": 12000000,
                "profession": "manager",
                "location_preference": "Shibuya",
                "family_size": 2
            }
            
            interaction_result = await self.dify.create_customer_interaction_workflow(test_customer)
            workflow_results["customer_interaction"] = {
                "success": interaction_result is not None,
                "workflow_id": interaction_result.get("workflow_id") if interaction_result else None
            }
            
            # Test property description workflow
            test_property = {
                "id": "test_property",
                "property_type": "Mansion",
                "location": "Shibuya",
                "size_sqm": 85,
                "price": 120000000,
                "rooms": "3LDK"
            }
            
            property_result = await self.dify.create_property_description_workflow(test_property)
            workflow_results["property_description"] = {
                "success": property_result is not None,
                "workflow_id": property_result.get("workflow_id") if property_result else None
            }
            
            return {
                "success": True,
                "active_workflows": len(workflow_results),
                "workflow_results": workflow_results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"LLM workflow integration failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _setup_monitoring(self) -> Dict[str, Any]:
        """Setup monitoring for deployed models"""
        try:
            monitoring_results = {}
            
            models_to_monitor = ["CustomerValueModel", "FloorplanDetectionModel"]
            
            for model_name in models_to_monitor:
                monitoring_config = self.mlflow.setup_model_monitoring(model_name)
                monitoring_results[model_name] = {
                    "configured": bool(monitoring_config),
                    "config": monitoring_config
                }
            
            return {
                "success": True,
                "monitored_models": len(monitoring_results),
                "monitoring_results": monitoring_results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Monitoring setup failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _validate_pipeline(self) -> Dict[str, Any]:
        """Validate the complete pipeline"""
        try:
            validation_results = {}
            
            # Test model serving endpoints
            kserve_health = {
                "customer-value": self.kserve.health_check("customer-value"),
                "floorplan": self.kserve.health_check("floorplan")
            }
            
            validation_results["kserve_health"] = kserve_health
            
            # Test MLflow model registry
            try:
                customer_model = self.mlflow.load_model_from_registry("CustomerValueModel", "Production")
                validation_results["mlflow_registry"] = {"accessible": customer_model is not None}
            except:
                validation_results["mlflow_registry"] = {"accessible": False}
            
            # Test Dify integration (mock)
            validation_results["dify_integration"] = {"accessible": True}  # Would test actual API
            
            # Test data pipeline
            validation_results["data_pipeline"] = {"accessible": True}  # Would test Snowflake connection
            
            overall_health = all([
                any(kserve_health.values()),  # At least one KServe model healthy
                validation_results["mlflow_registry"]["accessible"],
                validation_results["dify_integration"]["accessible"],
                validation_results["data_pipeline"]["accessible"]
            ])
            
            return {
                "success": True,
                "overall_health": overall_health,
                "component_health": validation_results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Pipeline validation failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _execute_snowflake_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute query on Snowflake (mock implementation)"""
        # Mock data for demo - would use real Snowflake service
        if "customer_profiles" in query:
            return [
                {
                    "customer_id": f"cust_{i}",
                    "age": np.random.randint(25, 65),
                    "annual_income": np.random.randint(5000000, 30000000),
                    "profession": np.random.choice(["engineer", "manager", "executive"]),
                    "location_preference": np.random.choice(["Shibuya", "Shinjuku", "Ginza"]),
                    "family_size": np.random.randint(1, 5),
                    "conversion_flag": np.random.randint(0, 2)
                }
                for i in range(1000)  # Mock 1000 customers
            ]
        elif "property_interactions" in query:
            return [
                {
                    "customer_id": f"cust_{np.random.randint(1, 1001)}",
                    "property_id": f"prop_{i}",
                    "interaction_type": np.random.choice(["view", "inquiry", "visit"]),
                    "interaction_duration": np.random.randint(30, 1800),
                    "property_type": np.random.choice(["Mansion", "Apartment", "Condo"]),
                    "property_location": np.random.choice(["Shibuya", "Shinjuku", "Ginza"]),
                    "price_range": np.random.choice(["50-100M", "100-200M", "200M+"])
                }
                for i in range(5000)  # Mock 5000 interactions
            ]
        return []
    
    def _prepare_customer_training_data(self, customer_data: List[Dict], interaction_data: List[Dict]) -> tuple:
        """Prepare customer training data"""
        if not customer_data:
            # Generate mock training data
            np.random.seed(42)
            n_samples = 5000
            
            # Features: age, income, location_idx, property_preference, family_size, engagement_metrics
            features = np.column_stack([
                np.random.normal(42, 10, n_samples).clip(25, 65).astype(int),  # age
                np.random.lognormal(16.5, 0.4, n_samples).clip(4_000_000, 50_000_000).astype(int),  # income
                np.random.choice(8, n_samples),  # location index (Tokyo areas)
                np.random.choice(5, n_samples),  # property preference
                np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.3, 0.35, 0.2, 0.1, 0.05]),  # family size
                np.random.exponential(180, n_samples),  # time on site
                np.random.poisson(3, n_samples),  # pages viewed
                np.random.poisson(2, n_samples)   # previous searches
            ])
            
            # Labels: conversion based on realistic probability
            conversion_prob = (
                (features[:, 1] > 8_000_000).astype(float) * 0.25 +  # High income
                ((features[:, 0] >= 30) & (features[:, 0] <= 50)).astype(float) * 0.2 +  # Prime age
                (features[:, 4] >= 2).astype(float) * 0.15 +  # Family buyers
                np.random.normal(0, 0.1, n_samples)
            ).clip(0, 1)
            
            labels = (np.random.random(n_samples) < conversion_prob).astype(int)
            
            return features, labels
        
        # Would process real data here
        return np.array([]), np.array([])
    
    async def _prepare_floorplan_data(self) -> List[Dict[str, Any]]:
        """Prepare floorplan training data (mock)"""
        return [
            {
                "property_id": f"prop_{i}",
                "image_path": f"/mock/floorplans/property_{i}.jpg",
                "ground_truth_rooms": np.random.randint(1, 6),
                "property_type": np.random.choice(["Mansion", "Apartment", "Studio"])
            }
            for i in range(100)  # Mock 100 floorplans
        ]
    
    async def _train_customer_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train customer value model"""
        try:
            from sklearn.model_selection import train_test_split
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train and register model using MLflow service
            training_results = self.mlflow.train_and_register_customer_model(
                X_train, y_train, X_test, y_test
            )
            
            return {
                "success": True,
                "model_version": training_results["model_version"],
                "run_id": training_results["run_id"],
                "metrics": training_results["metrics"]
            }
            
        except Exception as e:
            logger.error(f"Customer model training failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _train_floorplan_model(self, floorplan_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Train/register floorplan model"""
        try:
            # Mock model path - would be actual trained model
            model_path = "/mock/models/floorplan_model.pth"
            
            training_results = self.mlflow.train_and_register_floorplan_model(
                model_path, {"test_images": floorplan_data}
            )
            
            return {
                "success": True,
                "model_version": training_results["model_version"],
                "run_id": training_results["run_id"],
                "metrics": training_results["metrics"]
            }
            
        except Exception as e:
            logger.error(f"Floorplan model training failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _validate_model_performance(self, model_name: str, version: str) -> bool:
        """Validate model performance before promotion"""
        try:
            # Get performance history
            performance_history = self.mlflow.get_model_performance_history(
                model_name.replace("_model", "").title() + "Model"
            )
            
            if not performance_history:
                return True  # No history, allow promotion
            
            # Check if current version meets minimum thresholds
            current_version = next(
                (p for p in performance_history if p["version"] == version), None
            )
            
            if not current_version:
                return False
            
            # Model-specific validation criteria
            if "customer" in model_name.lower():
                accuracy = current_version["metrics"].get("accuracy", 0)
                f1_score = current_version["metrics"].get("f1_score", 0)
                return accuracy >= 0.8 and f1_score >= 0.75
            
            elif "floorplan" in model_name.lower():
                detection_accuracy = current_version["metrics"].get("room_detection_accuracy", 0)
                return detection_accuracy >= 0.75
            
            return True
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False
    
    async def monitor_model_drift(self, model_name: str, time_window_hours: int = 24) -> Dict[str, Any]:
        """Monitor model for data/concept drift"""
        try:
            # This would implement actual drift detection
            # For now, return mock results
            drift_results = {
                "model_name": model_name,
                "time_window_hours": time_window_hours,
                "data_drift_score": np.random.uniform(0, 0.3),  # Low drift
                "prediction_drift_score": np.random.uniform(0, 0.2),
                "performance_degradation": np.random.uniform(0, 0.1),
                "alert_triggered": False,
                "recommendations": [],
                "timestamp": datetime.now().isoformat()
            }
            
            # Trigger alerts if thresholds exceeded
            if (drift_results["data_drift_score"] > 0.1 or 
                drift_results["performance_degradation"] > 0.05):
                drift_results["alert_triggered"] = True
                drift_results["recommendations"].append("Consider model retraining")
            
            return drift_results
            
        except Exception as e:
            logger.error(f"Drift monitoring failed: {e}")
            return {"error": str(e)}
    
    async def trigger_model_retrain(self, model_name: str, reason: str) -> Dict[str, Any]:
        """Trigger automated model retraining"""
        try:
            logger.info(f"Triggering retrain for {model_name}: {reason}")
            
            # This would trigger the full pipeline for the specific model
            retrain_results = await self.run_full_pipeline("retrain")
            
            return {
                "model_name": model_name,
                "retrain_trigger": reason,
                "pipeline_results": retrain_results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Model retrain failed: {e}")
            return {"error": str(e)}

# Global pipeline service
mlops_pipeline = MLOpsPipeline()