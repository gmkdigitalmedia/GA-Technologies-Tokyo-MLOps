import boto3
import pandas as pd
from typing import Dict, List, Optional, Any
import json
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

class AWSService:
    def __init__(self):
        self.session = boto3.Session(
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION
        )
        self.s3 = self.session.client('s3')
        self.sagemaker = self.session.client('sagemaker')
        self.sagemaker_runtime = self.session.client('sagemaker-runtime')

class S3DataService:
    def __init__(self):
        self.aws_service = AWSService()
        self.bucket = settings.S3_BUCKET_NAME
    
    def upload_data(self, data: pd.DataFrame, key: str) -> str:
        """Upload DataFrame to S3 as parquet"""
        try:
            s3_path = f"s3://{self.bucket}/{settings.S3_DATA_PREFIX}{key}.parquet"
            # Convert DataFrame to parquet bytes and upload
            parquet_buffer = data.to_parquet(index=False)
            s3_key = f"{settings.S3_DATA_PREFIX}{key}.parquet"
            self.aws_service.s3.put_object(
                Bucket=self.bucket,
                Key=s3_key,
                Body=parquet_buffer
            )
            logger.info(f"Data uploaded to {s3_path}")
            return s3_path
        except Exception as e:
            logger.error(f"Failed to upload data to S3: {e}")
            raise
    
    def download_data(self, key: str) -> pd.DataFrame:
        """Download parquet data from S3"""
        try:
            s3_path = f"s3://{self.bucket}/{settings.S3_DATA_PREFIX}{key}.parquet"
            s3_key = f"{settings.S3_DATA_PREFIX}{key}.parquet"
            response = self.aws_service.s3.get_object(Bucket=self.bucket, Key=s3_key)
            data = pd.read_parquet(response['Body'])
            logger.info(f"Data downloaded from {s3_path}")
            return data
        except Exception as e:
            logger.error(f"Failed to download data from S3: {e}")
            raise
    
    def upload_model_artifact(self, local_path: str, model_name: str) -> str:
        """Upload model artifacts to S3"""
        try:
            s3_key = f"{settings.S3_MODEL_PREFIX}{model_name}"
            self.aws_service.s3.upload_file(local_path, self.bucket, s3_key)
            s3_path = f"s3://{self.bucket}/{s3_key}"
            logger.info(f"Model uploaded to {s3_path}")
            return s3_path
        except Exception as e:
            logger.error(f"Failed to upload model to S3: {e}")
            raise

class SnowflakeAWSService:
    def __init__(self):
        self.s3_service = S3DataService()
    
    def extract_from_snowflake_to_s3(self, query: str, s3_key: str) -> str:
        """Extract data from Snowflake and store in S3"""
        try:
            # TODO: Implement Snowflake connection using snowflake-connector-python
            # For now, return placeholder
            logger.warning("Snowflake extraction not implemented yet")
            return f"s3://{self.s3_service.bucket}/{settings.S3_DATA_PREFIX}{s3_key}.parquet"
            
        except Exception as e:
            logger.error(f"Failed to extract from Snowflake to S3: {e}")
            raise
    
    def load_s3_to_snowflake(self, s3_path: str, table_name: str, mode: str = "overwrite"):
        """Load data from S3 to Snowflake"""
        try:
            # TODO: Implement Snowflake loading using snowflake-connector-python
            # For now, log placeholder
            logger.warning(f"Snowflake loading not implemented yet for {s3_path} to {table_name}")
            
        except Exception as e:
            logger.error(f"Failed to load S3 data to Snowflake: {e}")
            raise

class SageMakerService:
    def __init__(self):
        self.aws_service = AWSService()
        self.s3_service = S3DataService()
    
    def deploy_model(self, model_name: str, model_s3_path: str, instance_type: str = None) -> str:
        """Deploy model to SageMaker endpoint"""
        try:
            if not instance_type:
                instance_type = settings.SAGEMAKER_INSTANCE_TYPE
            
            # Create model
            model_response = self.aws_service.sagemaker.create_model(
                ModelName=model_name,
                PrimaryContainer={
                    'Image': '763104351884.dkr.ecr.us-east-1.amazonaws.com/sklearn-inference:1.0-1-cpu-py3',
                    'ModelDataUrl': model_s3_path,
                    'Environment': {
                        'SAGEMAKER_PROGRAM': 'inference.py',
                        'SAGEMAKER_SUBMIT_DIRECTORY': model_s3_path
                    }
                },
                ExecutionRoleArn=settings.SAGEMAKER_EXECUTION_ROLE
            )
            
            # Create endpoint configuration
            endpoint_config_name = f"{model_name}-config"
            self.aws_service.sagemaker.create_endpoint_config(
                EndpointConfigName=endpoint_config_name,
                ProductionVariants=[
                    {
                        'VariantName': 'primary',
                        'ModelName': model_name,
                        'InitialInstanceCount': 1,
                        'InstanceType': instance_type,
                        'InitialVariantWeight': 1
                    }
                ]
            )
            
            # Create endpoint
            endpoint_name = f"{model_name}-endpoint"
            self.aws_service.sagemaker.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name
            )
            
            logger.info(f"Model {model_name} deployed to endpoint {endpoint_name}")
            return endpoint_name
            
        except Exception as e:
            logger.error(f"Failed to deploy model to SageMaker: {e}")
            raise
    
    def invoke_endpoint(self, endpoint_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke SageMaker endpoint for predictions"""
        try:
            response = self.aws_service.sagemaker_runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType='application/json',
                Body=json.dumps(payload)
            )
            
            result = json.loads(response['Body'].read().decode())
            return result
            
        except Exception as e:
            logger.error(f"Failed to invoke endpoint {endpoint_name}: {e}")
            raise