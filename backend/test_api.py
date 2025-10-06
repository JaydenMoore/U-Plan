#!/usr/bin/env python3
"""
Test script to check API responses and water risk scaling
"""

import requests
import json

def test_enhanced_assessment():
    """Test the enhanced assessment endpoint"""
    url = "http://localhost:8001/assess-location-enhanced"
    
    # Test with Los Angeles coordinates
    data = {
        "latitude": 34.0522,
        "longitude": -118.2437
    }
    
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        
        result = response.json()
        
        print("=== ENHANCED ASSESSMENT RESPONSE ===")
        print(f"Overall Risk Score: {result.get('overall_risk_score')}")
        
        # Check water risk data
        water_risk = result.get('water_risk')
        if water_risk:
            print("\n=== WATER RISK DATA ===")
            print(f"Overall Water Risk: {water_risk.get('overall_water_risk')}")
            print(f"Overall Category: {water_risk.get('overall_category')}")
            print(f"Baseline Water Stress: {water_risk.get('baseline_water_stress')}")
            print(f"Drought Risk: {water_risk.get('drought_risk')}")
            print(f"Flood Risk Score: {water_risk.get('flood_risk_score')}")
            print(f"Country: {water_risk.get('country')}")
            print(f"Source: {water_risk.get('source')}")
            
            # Check normalized values
            print("\n=== NORMALIZED VALUES (0-1) ===")
            print(f"Water Stress Normalized: {water_risk.get('water_stress_normalized')}")
            print(f"Flood Risk Normalized: {water_risk.get('flood_risk_normalized')}")
            print(f"Overall Risk Normalized: {water_risk.get('overall_risk_normalized')}")
        else:
            print("No water risk data found")
        
        # Check probabilistic risk
        prob_risk = result.get('probabilistic_risk')
        if prob_risk:
            print("\n=== PROBABILISTIC RISK ===")
            print(f"Expected Risk: {prob_risk.get('expected_risk')}")
            print(f"Model Fitted: {prob_risk.get('model_fitted')}")
        
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {e}")
        return None

if __name__ == "__main__":
    test_enhanced_assessment()