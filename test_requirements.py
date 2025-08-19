#!/usr/bin/env python3
"""
Test script to verify all required packages can be imported
"""

import sys

def test_imports():
    """Test that all critical packages can be imported"""
    
    try:
        # Core web framework
        import fastapi
        import uvicorn
        import pydantic
        print("PASS Web framework packages OK")
        
        # Database
        import sqlalchemy
        import alembic
        import psycopg2
        print("PASS Database packages OK")
        
        # Data processing
        import pandas
        import numpy
        print("PASS Data processing packages OK")
        
        # ML
        import sklearn
        print("PASS ML packages OK")
        
        # Snowflake
        import snowflake.connector
        import snowflake.sqlalchemy
        print("PASS Snowflake packages OK")
        
        # MLOps
        import mlflow
        print("PASS MLflow OK")
        
        # Cache/Queue
        import redis
        import celery
        print("PASS Redis/Celery OK")
        
        # AWS
        import boto3
        print("PASS AWS packages OK")
        
        # API
        import aiohttp
        import httpx
        print("PASS API packages OK")
        
        print("\n🎉 All required packages imported successfully!")
        print("Dependencies are properly resolved.")
        return True
        
    except ImportError as e:
        print(f"FAIL Import error: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)