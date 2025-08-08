"""
Dify LLM Workflow Integration Service
Manages LLM workflows for real estate customer interaction and property analysis
"""

import requests
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio
import aiohttp
from app.core.config import settings

logger = logging.getLogger(__name__)

class DifyWorkflowService:
    def __init__(self):
        self.dify_api_url = "http://localhost:2229"
        self.api_key = "app-" + "dify-gp-mlops"  # This would be generated in Dify console
        self.timeout = 30
        
    async def create_customer_interaction_workflow(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create an LLM workflow for personalized customer interaction
        
        Args:
            customer_data: Customer profile and preferences
            
        Returns:
            Generated customer interaction strategy
        """
        try:
            workflow_prompt = self._build_customer_interaction_prompt(customer_data)
            
            payload = {
                "inputs": {
                    "customer_profile": json.dumps(customer_data),
                    "interaction_goal": "real_estate_engagement",
                    "market_focus": "tokyo_premium"
                },
                "query": workflow_prompt,
                "response_mode": "blocking",
                "user": f"customer_{customer_data.get('id', 'unknown')}"
            }
            
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                async with session.post(
                    f"{self.dify_api_url}/v1/chat-messages", 
                    json=payload, 
                    headers=headers,
                    timeout=self.timeout
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    
                    return {
                        "customer_id": customer_data.get('id'),
                        "interaction_strategy": result.get('answer', ''),
                        "personalization_score": self._calculate_personalization_score(result),
                        "recommended_properties": self._extract_property_recommendations(result),
                        "follow_up_actions": self._extract_follow_up_actions(result),
                        "conversation_id": result.get('conversation_id'),
                        "workflow_id": "customer_interaction",
                        "timestamp": datetime.now().isoformat()
                    }
                    
        except Exception as e:
            logger.error(f"Dify customer interaction workflow failed: {e}")
            return self._fallback_customer_interaction(customer_data)
    
    async def create_property_description_workflow(self, property_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate compelling property descriptions using LLM workflows
        
        Args:
            property_data: Property details and features
            
        Returns:
            Generated property marketing content
        """
        try:
            workflow_prompt = self._build_property_description_prompt(property_data)
            
            payload = {
                "inputs": {
                    "property_details": json.dumps(property_data),
                    "target_audience": "tokyo_premium_buyers",
                    "marketing_style": "professional_persuasive"
                },
                "query": workflow_prompt,
                "response_mode": "blocking",
                "user": f"property_{property_data.get('id', 'unknown')}"
            }
            
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                async with session.post(
                    f"{self.dify_api_url}/v1/chat-messages", 
                    json=payload, 
                    headers=headers,
                    timeout=self.timeout
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    
                    return {
                        "property_id": property_data.get('id'),
                        "marketing_headline": self._extract_headline(result),
                        "full_description": result.get('answer', ''),
                        "key_selling_points": self._extract_selling_points(result),
                        "target_keywords": self._extract_seo_keywords(result),
                        "social_media_variants": self._generate_social_variants(result),
                        "conversation_id": result.get('conversation_id'),
                        "workflow_id": "property_marketing",
                        "timestamp": datetime.now().isoformat()
                    }
                    
        except Exception as e:
            logger.error(f"Dify property description workflow failed: {e}")
            return self._fallback_property_description(property_data)
    
    async def create_market_analysis_workflow(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate market analysis and insights using LLM workflows
        
        Args:
            market_data: Market trends and pricing data
            
        Returns:
            AI-generated market analysis
        """
        try:
            workflow_prompt = self._build_market_analysis_prompt(market_data)
            
            payload = {
                "inputs": {
                    "market_data": json.dumps(market_data),
                    "analysis_type": "tokyo_real_estate_trends",
                    "audience": "investors_and_buyers"
                },
                "query": workflow_prompt,
                "response_mode": "blocking",
                "user": "market_analyst"
            }
            
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                async with session.post(
                    f"{self.dify_api_url}/v1/chat-messages", 
                    json=payload, 
                    headers=headers,
                    timeout=self.timeout
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    
                    return {
                        "market_overview": self._extract_market_overview(result),
                        "key_trends": self._extract_market_trends(result),
                        "price_predictions": self._extract_price_predictions(result),
                        "investment_recommendations": self._extract_investment_advice(result),
                        "risk_factors": self._extract_risk_factors(result),
                        "full_analysis": result.get('answer', ''),
                        "confidence_score": self._calculate_analysis_confidence(result),
                        "conversation_id": result.get('conversation_id'),
                        "workflow_id": "market_analysis",
                        "timestamp": datetime.now().isoformat()
                    }
                    
        except Exception as e:
            logger.error(f"Dify market analysis workflow failed: {e}")
            return self._fallback_market_analysis(market_data)
    
    def _build_customer_interaction_prompt(self, customer_data: Dict[str, Any]) -> str:
        """Build prompt for customer interaction workflow"""
        age = customer_data.get('age', 45)
        income = customer_data.get('income', 12000000)
        location = customer_data.get('location_preference', 'Tokyo')
        
        return f"""
        Create a personalized real estate engagement strategy for a customer with the following profile:
        - Age: {age}
        - Annual Income: ¥{income:,}
        - Location Preference: {location}
        - Occupation: {customer_data.get('profession', 'Professional')}
        - Family Size: {customer_data.get('family_size', 2)}
        
        Focus on Tokyo premium real estate market. Provide:
        1. Personalized communication approach
        2. Property type recommendations
        3. Budget optimization suggestions
        4. Timeline recommendations
        5. Follow-up strategy
        
        Keep it professional and focused on high-value properties in Tokyo.
        """
    
    def _build_property_description_prompt(self, property_data: Dict[str, Any]) -> str:
        """Build prompt for property description workflow"""
        return f"""
        Create compelling marketing content for this Tokyo property:
        - Property Type: {property_data.get('property_type', 'Apartment')}
        - Location: {property_data.get('location', 'Tokyo')}
        - Size: {property_data.get('size_sqm', 80)} sqm
        - Price: ¥{property_data.get('price', 80000000):,}
        - Rooms: {property_data.get('rooms', '2LDK')}
        
        Target audience: High-income professionals (40-50 years old, 10M+ yen income)
        
        Generate:
        1. Compelling headline
        2. Detailed description highlighting luxury and convenience
        3. Key selling points
        4. SEO keywords for Tokyo real estate
        5. Social media variants
        
        Emphasize Tokyo location advantages, investment potential, and lifestyle benefits.
        """
    
    def _build_market_analysis_prompt(self, market_data: Dict[str, Any]) -> str:
        """Build prompt for market analysis workflow"""
        return f"""
        Analyze the Tokyo real estate market based on this data:
        - Average Price per sqm: ¥{market_data.get('avg_price_per_sqm', 800000):,}
        - Market Trend: {market_data.get('trend', 'stable')}
        - Inventory Levels: {market_data.get('inventory', 'moderate')}
        - Interest Rates: {market_data.get('interest_rate', 1.2)}%
        
        Provide comprehensive analysis including:
        1. Market overview and current conditions
        2. Key trends affecting Tokyo real estate
        3. Price predictions for next 12 months
        4. Investment recommendations by area
        5. Risk factors to consider
        
        Focus on premium market segments and high-value properties.
        """
    
    def _calculate_personalization_score(self, result: Dict) -> float:
        """Calculate personalization score based on response quality"""
        answer = result.get('answer', '')
        # Simple scoring based on content length and keywords
        score = min(len(answer) / 1000, 1.0)  # Length factor
        tokyo_keywords = ['tokyo', 'shibuya', 'shinjuku', 'ginza', 'premium']
        keyword_score = sum(1 for kw in tokyo_keywords if kw.lower() in answer.lower()) / len(tokyo_keywords)
        return (score * 0.6 + keyword_score * 0.4)
    
    def _extract_property_recommendations(self, result: Dict) -> List[str]:
        """Extract property recommendations from LLM response"""
        # This would use NLP to extract property types mentioned
        answer = result.get('answer', '')
        properties = []
        
        property_types = ['mansion', 'apartment', 'condo', 'studio', 'house']
        for prop_type in property_types:
            if prop_type in answer.lower():
                properties.append(prop_type.capitalize())
        
        return properties[:3]  # Top 3 recommendations
    
    def _extract_follow_up_actions(self, result: Dict) -> List[str]:
        """Extract follow-up actions from LLM response"""
        # Extract actionable items from the response
        answer = result.get('answer', '')
        actions = []
        
        action_indicators = ['schedule', 'contact', 'send', 'show', 'arrange', 'follow up']
        sentences = answer.split('.')
        
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in action_indicators):
                actions.append(sentence.strip())
        
        return actions[:3]  # Top 3 actions
    
    def _extract_headline(self, result: Dict) -> str:
        """Extract marketing headline from property description"""
        answer = result.get('answer', '')
        lines = answer.split('\n')
        
        # Look for headline patterns
        for line in lines:
            if ('headline' in line.lower() or 
                line.startswith('**') or 
                len(line.strip()) < 100 and len(line.strip()) > 20):
                return line.strip().replace('*', '').replace('Headline:', '').strip()
        
        # Fallback: use first meaningful line
        return lines[0][:80] + "..." if len(lines[0]) > 80 else lines[0]
    
    def _extract_selling_points(self, result: Dict) -> List[str]:
        """Extract key selling points from property description"""
        answer = result.get('answer', '')
        points = []
        
        # Look for bullet points or numbered lists
        lines = answer.split('\n')
        for line in lines:
            line = line.strip()
            if (line.startswith('-') or line.startswith('•') or 
                line.startswith('*') or line[0:3].replace('.', '').isdigit()):
                cleaned = line.lstrip('-•*0123456789. ').strip()
                if cleaned and len(cleaned) > 10:
                    points.append(cleaned)
        
        return points[:5]  # Top 5 selling points
    
    def _extract_seo_keywords(self, result: Dict) -> List[str]:
        """Extract SEO keywords for Tokyo real estate"""
        answer = result.get('answer', '')
        
        # Common Tokyo real estate keywords
        base_keywords = [
            'tokyo real estate', 'premium property', 'luxury apartment',
            'shibuya', 'shinjuku', 'ginza', 'investment property',
            'high-rise', 'modern', 'convenient location'
        ]
        
        # Find which keywords appear in the response
        relevant_keywords = []
        for keyword in base_keywords:
            if keyword.lower() in answer.lower():
                relevant_keywords.append(keyword)
        
        return relevant_keywords
    
    def _generate_social_variants(self, result: Dict) -> List[str]:
        """Generate social media variants from property description"""
        answer = result.get('answer', '')
        
        # Create shortened versions for different platforms
        full_text = answer.replace('\n', ' ').strip()
        
        variants = [
            full_text[:280] + "..." if len(full_text) > 280 else full_text,  # Twitter
            full_text[:1000] + "..." if len(full_text) > 1000 else full_text,  # LinkedIn
            full_text[:500] + "..." if len(full_text) > 500 else full_text   # Instagram
        ]
        
        return variants
    
    def _extract_market_overview(self, result: Dict) -> str:
        """Extract market overview from analysis"""
        answer = result.get('answer', '')
        paragraphs = answer.split('\n\n')
        
        # First substantial paragraph is usually the overview
        for paragraph in paragraphs:
            if len(paragraph.strip()) > 100:
                return paragraph.strip()[:500]
        
        return answer[:500] + "..." if len(answer) > 500 else answer
    
    def _extract_market_trends(self, result: Dict) -> List[str]:
        """Extract key market trends"""
        answer = result.get('answer', '')
        trends = []
        
        trend_keywords = ['trend', 'growth', 'decline', 'increase', 'decrease', 'rising', 'falling']
        sentences = answer.split('.')
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in trend_keywords):
                trends.append(sentence.strip())
        
        return trends[:4]  # Top 4 trends
    
    def _extract_price_predictions(self, result: Dict) -> Dict[str, str]:
        """Extract price predictions from market analysis"""
        answer = result.get('answer', '')
        
        # Look for price-related predictions
        predictions = {
            "short_term": "Stable growth expected",
            "medium_term": "Continued upward trend",
            "long_term": "Strong investment potential"
        }
        
        # This would be enhanced with better NLP parsing
        if 'increase' in answer.lower() or 'growth' in answer.lower():
            predictions["outlook"] = "Positive"
        elif 'decrease' in answer.lower() or 'decline' in answer.lower():
            predictions["outlook"] = "Cautious"
        else:
            predictions["outlook"] = "Stable"
        
        return predictions
    
    def _extract_investment_advice(self, result: Dict) -> List[str]:
        """Extract investment recommendations"""
        answer = result.get('answer', '')
        advice = []
        
        recommendation_keywords = ['recommend', 'suggest', 'consider', 'invest', 'buy', 'sell']
        sentences = answer.split('.')
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in recommendation_keywords):
                advice.append(sentence.strip())
        
        return advice[:3]  # Top 3 recommendations
    
    def _extract_risk_factors(self, result: Dict) -> List[str]:
        """Extract risk factors from market analysis"""
        answer = result.get('answer', '')
        risks = []
        
        risk_keywords = ['risk', 'caution', 'warning', 'concern', 'challenge', 'volatile']
        sentences = answer.split('.')
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in risk_keywords):
                risks.append(sentence.strip())
        
        return risks[:3]  # Top 3 risks
    
    def _calculate_analysis_confidence(self, result: Dict) -> float:
        """Calculate confidence score for market analysis"""
        answer = result.get('answer', '')
        
        # Score based on length, data references, and certainty language
        length_score = min(len(answer) / 2000, 1.0)
        
        data_keywords = ['data', 'statistics', 'analysis', 'research', 'study']
        data_score = sum(1 for kw in data_keywords if kw.lower() in answer.lower()) / len(data_keywords)
        
        uncertainty_keywords = ['might', 'possibly', 'perhaps', 'uncertain']
        certainty_penalty = sum(1 for kw in uncertainty_keywords if kw.lower() in answer.lower()) * 0.1
        
        return max(0.0, min(1.0, (length_score * 0.4 + data_score * 0.6) - certainty_penalty))
    
    # Fallback methods when Dify is unavailable
    def _fallback_customer_interaction(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback customer interaction when Dify is unavailable"""
        return {
            "customer_id": customer_data.get('id'),
            "interaction_strategy": "Personalized consultation recommended based on profile",
            "personalization_score": 0.5,
            "recommended_properties": ["Premium Apartment", "Luxury Condo"],
            "follow_up_actions": ["Schedule consultation", "Send property listings"],
            "conversation_id": None,
            "workflow_id": "fallback_interaction",
            "timestamp": datetime.now().isoformat()
        }
    
    def _fallback_property_description(self, property_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback property description when Dify is unavailable"""
        return {
            "property_id": property_data.get('id'),
            "marketing_headline": f"Premium {property_data.get('property_type', 'Property')} in Tokyo",
            "full_description": "Luxury property in prime Tokyo location with modern amenities",
            "key_selling_points": ["Prime location", "Modern design", "Excellent transport links"],
            "target_keywords": ["tokyo real estate", "premium property", "luxury apartment"],
            "social_media_variants": ["Premium Tokyo property available", "Luxury living in Tokyo"],
            "conversation_id": None,
            "workflow_id": "fallback_property",
            "timestamp": datetime.now().isoformat()
        }
    
    def _fallback_market_analysis(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback market analysis when Dify is unavailable"""
        return {
            "market_overview": "Tokyo real estate market showing stable growth trends",
            "key_trends": ["Continued demand in central areas", "Premium segment remains strong"],
            "price_predictions": {"outlook": "Stable", "short_term": "Stable growth"},
            "investment_recommendations": ["Focus on central Tokyo locations"],
            "risk_factors": ["Interest rate changes", "Economic volatility"],
            "full_analysis": "Market analysis unavailable - using baseline assessment",
            "confidence_score": 0.3,
            "conversation_id": None,
            "workflow_id": "fallback_analysis",
            "timestamp": datetime.now().isoformat()
        }

# Service instance
dify_service = DifyWorkflowService()