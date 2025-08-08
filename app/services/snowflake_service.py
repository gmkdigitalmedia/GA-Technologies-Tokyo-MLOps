import snowflake.connector
import pandas as pd
from typing import List, Dict, Any, Optional
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

class SnowflakeService:
    def __init__(self):
        self.connection_params = {
            'account': settings.SNOWFLAKE_ACCOUNT,
            'user': settings.SNOWFLAKE_USER,
            'password': settings.SNOWFLAKE_PASSWORD,
            'database': settings.SNOWFLAKE_DATABASE,
            'warehouse': settings.SNOWFLAKE_WAREHOUSE,
            'schema': settings.SNOWFLAKE_SCHEMA
        }
    
    def get_connection(self):
        """Get Snowflake connection"""
        try:
            conn = snowflake.connector.connect(**self.connection_params)
            return conn
        except Exception as e:
            logger.error(f"Failed to connect to Snowflake: {e}")
            raise
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute query and return DataFrame"""
        try:
            with self.get_connection() as conn:
                return pd.read_sql(query, conn)
        except Exception as e:
            logger.error(f"Failed to execute query: {e}")
            raise
    
    def get_customer_data(self, customer_ids: Optional[List[int]] = None, 
                         limit: int = 1000) -> pd.DataFrame:
        """Get customer data from Snowflake"""
        base_query = """
        SELECT 
            customer_id,
            age,
            income,
            profession,
            location,
            family_size,
            property_preferences,
            registration_date,
            last_login_date
        FROM customers
        """
        
        if customer_ids:
            id_list = ','.join(map(str, customer_ids))
            query = f"{base_query} WHERE customer_id IN ({id_list})"
        else:
            query = f"{base_query} ORDER BY registration_date DESC LIMIT {limit}"
        
        return self.execute_query(query)
    
    def get_property_interaction_data(self, days_back: int = 30) -> pd.DataFrame:
        """Get property interaction data"""
        query = f"""
        SELECT 
            pi.customer_id,
            pi.property_id,
            pi.interaction_type,
            pi.interaction_timestamp,
            pi.session_duration,
            p.property_type,
            p.price,
            p.location,
            p.size_sqm,
            p.rooms
        FROM property_interactions pi
        JOIN properties p ON pi.property_id = p.property_id
        WHERE pi.interaction_timestamp >= DATEADD(day, -{days_back}, CURRENT_DATE())
        ORDER BY pi.interaction_timestamp DESC
        """
        
        return self.execute_query(query)
    
    def get_conversion_data(self) -> pd.DataFrame:
        """Get conversion data for ML training"""
        query = """
        SELECT 
            c.customer_id,
            c.age,
            c.income,
            c.profession,
            c.location,
            c.family_size,
            COUNT(pi.interaction_id) as total_interactions,
            COUNT(DISTINCT pi.property_id) as unique_properties_viewed,
            AVG(pi.session_duration) as avg_session_duration,
            CASE WHEN co.customer_id IS NOT NULL THEN 1 ELSE 0 END as converted,
            COALESCE(co.conversion_value, 0) as conversion_value
        FROM customers c
        LEFT JOIN property_interactions pi ON c.customer_id = pi.customer_id
        LEFT JOIN conversions co ON c.customer_id = co.customer_id
        WHERE c.registration_date >= DATEADD(month, -12, CURRENT_DATE())
        GROUP BY c.customer_id, c.age, c.income, c.profession, c.location, 
                 c.family_size, co.customer_id, co.conversion_value
        """
        
        return self.execute_query(query)
    
    def get_property_features(self) -> pd.DataFrame:
        """Get property features for recommendation models"""
        query = """
        SELECT 
            property_id,
            property_type,
            price,
            location,
            size_sqm,
            rooms,
            floor,
            building_age,
            amenities,
            avg_rating,
            view_count,
            inquiry_count
        FROM properties
        WHERE is_active = TRUE
        """
        
        return self.execute_query(query)
    
    def store_predictions(self, predictions: pd.DataFrame, table_name: str = "ml_predictions"):
        """Store ML predictions back to Snowflake"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Create table if not exists
                create_table_query = f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    prediction_id STRING,
                    customer_id INTEGER,
                    model_name STRING,
                    prediction_type STRING,
                    prediction_value FLOAT,
                    confidence_score FLOAT,
                    features_used VARIANT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
                )
                """
                cursor.execute(create_table_query)
                
                # Insert predictions
                for _, row in predictions.iterrows():
                    insert_query = f"""
                    INSERT INTO {table_name} 
                    (prediction_id, customer_id, model_name, prediction_type, 
                     prediction_value, confidence_score, features_used)
                    VALUES (%(prediction_id)s, %(customer_id)s, %(model_name)s, 
                            %(prediction_type)s, %(prediction_value)s, 
                            %(confidence_score)s, %(features_used)s)
                    """
                    cursor.execute(insert_query, row.to_dict())
                
                conn.commit()
                logger.info(f"Stored {len(predictions)} predictions to {table_name}")
                
        except Exception as e:
            logger.error(f"Failed to store predictions: {e}")
            raise
    
    def get_customer_segments(self) -> Dict[str, pd.DataFrame]:
        """Get predefined customer segments"""
        segments = {}
        
        # High-value customers (top 20% by lifetime value)
        high_value_query = """
        SELECT customer_id, lifetime_value, conversion_probability
        FROM customer_scores 
        WHERE lifetime_value_percentile >= 80
        ORDER BY lifetime_value DESC
        """
        segments['high_value'] = self.execute_query(high_value_query)
        
        # High conversion probability customers
        high_conversion_query = """
        SELECT customer_id, conversion_probability, estimated_value
        FROM customer_scores 
        WHERE conversion_probability >= 0.7
        ORDER BY conversion_probability DESC
        """
        segments['high_conversion'] = self.execute_query(high_conversion_query)
        
        # Recently active customers
        active_query = """
        SELECT DISTINCT c.customer_id, c.location, c.property_preferences
        FROM customers c
        JOIN property_interactions pi ON c.customer_id = pi.customer_id
        WHERE pi.interaction_timestamp >= DATEADD(day, -7, CURRENT_DATE())
        """
        segments['recently_active'] = self.execute_query(active_query)
        
        return segments