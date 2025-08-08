from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql://postgres:password@localhost:5432/ga_platform"
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379"
    
    # Snowflake
    SNOWFLAKE_ACCOUNT: Optional[str] = None
    SNOWFLAKE_USER: Optional[str] = None
    SNOWFLAKE_PASSWORD: Optional[str] = None
    SNOWFLAKE_DATABASE: Optional[str] = None
    SNOWFLAKE_WAREHOUSE: Optional[str] = None
    SNOWFLAKE_SCHEMA: Optional[str] = None
    
    # AWS Configuration
    AWS_REGION: str = "us-east-1"
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    
    # S3 Configuration
    S3_BUCKET_NAME: str = "ga-technology-mlops"
    S3_DATA_PREFIX: str = "data/"
    S3_MODEL_PREFIX: str = "models/"
    
    # SageMaker Configuration
    SAGEMAKER_EXECUTION_ROLE: Optional[str] = None
    SAGEMAKER_INSTANCE_TYPE: str = "ml.m5.large"
    
    # MLflow (on AWS)
    MLFLOW_TRACKING_URI: str = "http://localhost:2226"
    MLFLOW_S3_ARTIFACT_ROOT: str = "s3://ga-technology-mlops/mlflow-artifacts/"
    
    # API Keys
    JWT_SECRET_KEY: str = "your-secret-key-here"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # ML Models
    CUSTOMER_VALUE_MODEL_PATH: str = "models/customer_value_model.pkl"
    FLOORPLAN_MODEL_PATH: str = "models/floorplan_detection_model.pth"
    
    # Feature flags
    ENABLE_CONVERSION_API: bool = True
    ENABLE_FLOORPLAN_DETECTION: bool = True
    ENABLE_CUSTOMER_INFERENCE: bool = True
    
    # Performance
    MAX_WORKERS: int = 4
    BATCH_SIZE: int = 32
    
    class Config:
        env_file = ".env"

settings = Settings()