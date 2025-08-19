#!/usr/bin/env python3
"""
GP MLOps Ad Serving Example

This example demonstrates how to use the ad serving platform for
targeting high-income salaryman (40-50 years old, >10M yen income).

Run this after setting up the platform to see the complete workflow.
"""

import requests
import json
import time
from datetime import datetime, timedelta
import random

# Configuration
API_BASE_URL = "http://localhost:8000"
API_ENDPOINTS = {
    'create_campaign': f"{API_BASE_URL}/api/v1/ads/campaigns",
    'create_creative': f"{API_BASE_URL}/api/v1/ads/creatives",
    'activate_campaign': f"{API_BASE_URL}/api/v1/ads/campaigns",
    'serve_ad': f"{API_BASE_URL}/api/v1/ads/serve",
    'track_interaction': f"{API_BASE_URL}/api/v1/ads/track",
    'ab_test_analysis': f"{API_BASE_URL}/api/v1/ads/campaigns",
    'conversion_track': f"{API_BASE_URL}/api/v1/capi/track"
}

def create_sample_campaign():
    """Create a sample campaign targeting GP MLOps' demographic"""
    
    print("🎯 Creating ad campaign for high-income salaryman...")
    
    campaign_data = {
        "name": "Renosy Premium Properties Q1 2024",
        "description": "Target high-income professionals aged 40-50 for premium real estate investment",
        "target_audience": "high_income_salaryman",
        "min_age": 40,
        "max_age": 50,
        "min_income": 10000000,  # 10M yen
        "target_locations": ["tokyo", "osaka", "nagoya", "yokohama"],
        "daily_budget": 50000.0,  # 50K yen per day
        "total_budget": 1500000.0,  # 1.5M yen total
        "start_date": datetime.now().isoformat(),
        "end_date": (datetime.now() + timedelta(days=30)).isoformat(),
        "ab_test_enabled": True,
        "traffic_split": {"control": 50, "variant_a": 50}
    }
    
    response = requests.post(API_ENDPOINTS['create_campaign'], json=campaign_data)
    
    if response.status_code == 200:
        campaign_result = response.json()
        campaign_id = campaign_result['campaign_id']
        print(f"PASS Campaign created: ID {campaign_id}")
        return campaign_id
    else:
        print(f"FAIL Campaign creation failed: {response.text}")
        return None

def create_ab_test_creatives(campaign_id):
    """Create A/B test creatives for the campaign"""
    
    print("🎨 Creating A/B test creatives...")
    
    creatives = [
        {
            "campaign_id": campaign_id,
            "variant_type": "control",
            "headline": "東京都心の投資用マンション | Renosy",
            "description": "年収1000万円以上の方限定。都心一等地の投資用マンションで資産形成を始めませんか？",
            "call_to_action": "無料相談を予約",
            "image_url": "https://renosy.com/images/premium-tokyo-mansion.jpg",
            "landing_page_url": "https://renosy.com/premium-investment?utm_source=ads&utm_campaign=q1_2024",
            "creative_type": "banner",
            "dimensions": {"width": 728, "height": 90}
        },
        {
            "campaign_id": campaign_id,
            "variant_type": "variant_a", 
            "headline": "40代からの資産運用 | 都心マンション投資",
            "description": "サラリーマンのための不動産投資。月々の家賃収入で将来の安心を手に入れましょう。",
            "call_to_action": "投資シミュレーション",
            "image_url": "https://renosy.com/images/salaryman-investment.jpg",
            "landing_page_url": "https://renosy.com/simulation?utm_source=ads&utm_campaign=q1_2024_var_a",
            "creative_type": "banner",
            "dimensions": {"width": 728, "height": 90}
        }
    ]
    
    creative_ids = []
    for creative_data in creatives:
        response = requests.post(API_ENDPOINTS['create_creative'], json=creative_data)
        
        if response.status_code == 200:
            creative_result = response.json()
            creative_ids.append(creative_result['creative_id'])
            print(f"PASS Creative created: {creative_data['variant_type']} - ID {creative_result['creative_id']}")
        else:
            print(f"FAIL Creative creation failed: {response.text}")
    
    return creative_ids

def activate_campaign(campaign_id):
    """Activate the campaign to start serving ads"""
    
    print("LAUNCH Activating campaign...")
    
    response = requests.put(f"{API_ENDPOINTS['activate_campaign']}/{campaign_id}/status", 
                           params={"status": "active"})
    
    if response.status_code == 200:
        print("PASS Campaign activated and ready to serve ads")
        return True
    else:
        print(f"FAIL Campaign activation failed: {response.text}")
        return False

def simulate_customer_requests(num_requests=10):
    """Simulate customer requests for ad serving"""
    
    print(f"👥 Simulating {num_requests} customer ad requests...")
    
    # Sample customers matching GP MLOps' target demographic
    sample_customers = [
        {"id": 1001, "age": 45, "income": 12000000, "profession": "engineer", "location": "tokyo"},
        {"id": 1002, "age": 42, "income": 15000000, "profession": "manager", "location": "osaka"},
        {"id": 1003, "age": 48, "income": 11000000, "profession": "consultant", "location": "nagoya"},
        {"id": 1004, "age": 44, "income": 13500000, "profession": "executive", "location": "yokohama"},
        {"id": 1005, "age": 46, "income": 10500000, "profession": "doctor", "location": "tokyo"},
    ]
    
    served_ads = []
    
    for i in range(num_requests):
        # Select random customer
        customer = random.choice(sample_customers)
        
        ad_request = {
            "customer_id": customer["id"],
            "page_url": "https://renosy.com/properties",
            "device_type": random.choice(["desktop", "mobile", "tablet"]),
            "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }
        
        # Add customer to database (mock - in real scenario they'd already exist)
        # This would typically be done through the customer creation API
        
        response = requests.post(API_ENDPOINTS['serve_ad'], json=ad_request)
        
        if response.status_code == 200:
            ad_response = response.json()
            served_ads.append({
                'customer_id': customer["id"],
                'ad_id': ad_response['ad_id'],
                'serving_id': ad_response['serving_id'],
                'variant': ad_response['variant_type'],
                'headline': ad_response['content']['headline']
            })
            print(f"📺 Ad served to customer {customer['id']}: {ad_response['variant_type']}")
        else:
            print(f"FAIL Ad serving failed for customer {customer['id']}: {response.text}")
        
        time.sleep(0.1)  # Brief pause between requests
    
    print(f"PASS Served {len(served_ads)} ads successfully")
    return served_ads

def simulate_user_interactions(served_ads):
    """Simulate user interactions with served ads"""
    
    print("👆 Simulating user interactions...")
    
    interactions = []
    
    for ad in served_ads:
        serving_id = ad['serving_id']
        
        # Simulate impression (always happens)
        impression_response = requests.post(
            f"{API_ENDPOINTS['track_interaction']}/{serving_id}",
            json={
                "interaction_type": "impression",
                "page_url": "https://renosy.com/properties",
                "session_id": f"session_{random.randint(10000, 99999)}"
            }
        )
        
        if impression_response.status_code == 200:
            interactions.append({"serving_id": serving_id, "type": "impression"})
        
        # Simulate click (30% probability)
        if random.random() < 0.3:
            click_response = requests.post(
                f"{API_ENDPOINTS['track_interaction']}/{serving_id}",
                json={
                    "interaction_type": "click",
                    "page_url": ad.get('landing_page_url', 'https://renosy.com/premium-investment'),
                    "session_id": f"session_{random.randint(10000, 99999)}"
                }
            )
            
            if click_response.status_code == 200:
                interactions.append({"serving_id": serving_id, "type": "click"})
                print(f"👆 Click tracked for ad {ad['ad_id']}")
        
        # Simulate conversion (5% probability for clicks)
        if len([i for i in interactions if i['serving_id'] == serving_id and i['type'] == 'click']) > 0:
            if random.random() < 0.05:  # 5% conversion rate
                conversion_response = requests.post(
                    API_ENDPOINTS['conversion_track'],
                    json={
                        "event_type": "property_inquiry",
                        "customer_id": ad['customer_id'],
                        "value": 100000,  # 100K yen inquiry value
                        "properties": {
                            "inquiry_type": "premium_consultation",
                            "source_ad_id": ad['ad_id']
                        }
                    }
                )
                
                if conversion_response.status_code == 200:
                    interactions.append({"serving_id": serving_id, "type": "conversion"})
                    print(f"💰 Conversion tracked for customer {ad['customer_id']}")
        
        time.sleep(0.1)
    
    print(f"PASS Tracked {len(interactions)} interactions")
    return interactions

def analyze_campaign_performance(campaign_id):
    """Analyze A/B test results and campaign performance"""
    
    print("📊 Analyzing campaign performance...")
    
    # Wait a bit for data to be processed
    time.sleep(2)
    
    # Get A/B test analysis
    ab_test_response = requests.get(f"{API_ENDPOINTS['ab_test_analysis']}/{campaign_id}/ab-test")
    
    if ab_test_response.status_code == 200:
        ab_results = ab_test_response.json()
        
        print("\n📈 A/B Test Results:")
        print(f"Campaign ID: {ab_results['campaign_id']}")
        print(f"Test Duration: {ab_results['test_duration_days']} days")
        
        for variant, results in ab_results['variant_results'].items():
            print(f"\n{variant.upper()}:")
            print(f"  Impressions: {results['impressions']}")
            print(f"  Clicks: {results['clicks']}")  
            print(f"  Conversions: {results['conversions']}")
            print(f"  CTR: {results['ctr']:.3f}%")
            print(f"  CVR: {results['cvr']:.3f}%")
        
        winner = ab_results['winner']
        print(f"\n🏆 Winner: {winner['variant']}")
        print(f"   CVR: {winner['cvr']:.3f}%")
        print(f"   Improvement: {winner['improvement']:.1f}%")
        print(f"   Statistical Significance: {winner['is_statistically_significant']}")
        
        print(f"\n💡 Recommendation: {ab_results['recommendation']}")
        
    else:
        print(f"FAIL A/B test analysis failed: {ab_test_response.text}")

def main():
    """Run the complete ad serving example"""
    
    print("🏢 GP MLOps Ad Serving Platform Demo")
    print("=" * 50)
    print("Target: High-income salaryman (40-50 years, >10M yen income)")
    print()
    
    # Step 1: Create campaign
    campaign_id = create_sample_campaign()
    if not campaign_id:
        return
    
    # Step 2: Create A/B test creatives
    creative_ids = create_ab_test_creatives(campaign_id)
    if not creative_ids:
        return
    
    # Step 3: Activate campaign
    if not activate_campaign(campaign_id):
        return
    
    # Step 4: Simulate ad serving
    served_ads = simulate_customer_requests(20)
    if not served_ads:
        return
    
    # Step 5: Simulate user interactions
    interactions = simulate_user_interactions(served_ads)
    
    # Step 6: Analyze performance
    analyze_campaign_performance(campaign_id)
    
    print("\n🎉 Demo completed successfully!")
    print(f"Campaign ID {campaign_id} is now running with A/B testing enabled.")
    print("Check the GP MLOps dashboard for real-time performance metrics.")

if __name__ == "__main__":
    main()