#!/usr/bin/env python3
"""
Comprehensive test script to validate all data scales against UI expectations
"""

import requests
import json

def test_all_data_scales():
    """Test all data values against their expected UI scales"""
    url = "http://localhost:8001/assess-location-enhanced"
    
    # Test with Los Angeles coordinates (known to have diverse data)
    data = {
        "latitude": 34.0522,
        "longitude": -118.2437
    }
    
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        result = response.json()
        
        print("=== UI SCALE VALIDATION REPORT ===\n")
        
        # 1. Overall Risk Score (UI expects 0-10)
        overall_score = result.get('overall_risk_score')
        print(f"1. OVERALL RISK SCORE:")
        print(f"   Current Value: {overall_score}")
        print(f"   UI Expects: /10 scale")
        print(f"   ✅ Status: {'GOOD' if 0 <= overall_score <= 10 else 'NEEDS FIXING'}")
        print()
        
        # 2. Water Risk Data (UI expects /5 for all)
        water_risk = result.get('water_risk')
        if water_risk:
            print(f"2. WATER RISK SCORES:")
            water_checks = [
                ('baseline_water_stress', 'Water Stress'),
                ('drought_risk', 'Drought Risk'), 
                ('flood_risk_score', 'Flood Risk'),
                ('overall_water_risk', 'Overall Water Risk')
            ]
            
            for key, label in water_checks:
                value = water_risk.get(key, 0)
                status = 'GOOD' if 0 <= value <= 5 else 'NEEDS FIXING'
                print(f"   {label}: {value}/5 - {status}")
            print()
        
        # 3. Air Quality Index (UI expects 0-500)
        aqi = result.get('air_quality_index')
        print(f"3. AIR QUALITY INDEX:")
        print(f"   Current Value: {aqi}")
        print(f"   UI Expects: 0-500 scale")
        print(f"   ✅ Status: {'GOOD' if 0 <= aqi <= 500 else 'NEEDS FIXING'}")
        print()
        
        # 4. PM2.5 (UI expects μg/m³, reasonable range 0-500)
        pm25 = result.get('pm2_5')
        print(f"4. PM2.5:")
        print(f"   Current Value: {pm25} μg/m³")
        print(f"   UI Expects: μg/m³ (typical range 0-500)")
        print(f"   ✅ Status: {'GOOD' if 0 <= pm25 <= 500 else 'NEEDS FIXING'}")
        print()
        
        # 5. PM10 (UI expects μg/m³, reasonable range 0-600)
        pm10 = result.get('pm10')
        if pm10:
            print(f"5. PM10:")
            print(f"   Current Value: {pm10} μg/m³")
            print(f"   UI Expects: μg/m³ (typical range 0-600)")
            print(f"   ✅ Status: {'GOOD' if 0 <= pm10 <= 600 else 'NEEDS FIXING'}")
            print()
        
        # 6. NO2 (UI expects μg/m³, reasonable range 0-200)
        no2 = result.get('no2')
        if no2:
            print(f"6. NO2:")
            print(f"   Current Value: {no2} μg/m³")
            print(f"   UI Expects: μg/m³ (typical range 0-200)")
            print(f"   ✅ Status: {'GOOD' if 0 <= no2 <= 200 else 'NEEDS FIXING'}")
            print()
        
        # 7. Flood Risk Score (UI shows as percentage 0-100%)
        flood_score = result.get('flood_risk_score')
        print(f"7. FLOOD RISK SCORE:")
        print(f"   Current Value: {flood_score}")
        print(f"   UI Expects: 0-1 scale (shown as percentage)")
        print(f"   UI Display: {(flood_score * 100):.1f}%")
        print(f"   ✅ Status: {'GOOD' if 0 <= flood_score <= 1 else 'NEEDS FIXING'}")
        print()
        
        # 8. Historic Flood Frequency (UI shows as percentage 0-100%)
        historic_flood = result.get('historic_flood_frequency')
        if historic_flood is not None:
            print(f"8. HISTORIC FLOOD FREQUENCY:")
            print(f"   Current Value: {historic_flood}")
            print(f"   UI Expects: 0-1 scale (shown as percentage)")
            print(f"   UI Display: {(historic_flood * 100):.1f}%")
            print(f"   ✅ Status: {'GOOD' if 0 <= historic_flood <= 1 else 'NEEDS FIXING'}")
            print()
        
        # 9. Temperature (UI expects °C, reasonable range -50 to 60)
        temp = result.get('temperature_c')
        print(f"9. TEMPERATURE:")
        print(f"   Current Value: {temp}°C")
        print(f"   UI Expects: °C (reasonable range -50 to 60)")
        print(f"   ✅ Status: {'GOOD' if -50 <= temp <= 60 else 'NEEDS FIXING'}")
        print()
        
        # 10. Rainfall (UI expects mm/month, reasonable range 0-2000)
        rainfall = result.get('rainfall_mm')
        print(f"10. RAINFALL:")
        print(f"   Current Value: {rainfall} mm/month")
        print(f"   UI Expects: mm/month (reasonable range 0-2000)")
        print(f"   ✅ Status: {'GOOD' if 0 <= rainfall <= 2000 else 'NEEDS FIXING'}")
        print()
        
        # 11. Population Density (UI expects people/km², reasonable range 0-50000)
        pop_density = result.get('population_density')
        if pop_density:
            print(f"11. POPULATION DENSITY:")
            print(f"   Current Value: {pop_density}")
            print(f"   UI Expects: people/km² (reasonable range 0-50000)")
            print(f"   ✅ Status: {'GOOD' if 0 <= pop_density <= 50000 else 'NEEDS FIXING'}")
            print()
        
        # 12. Risk Categories (should be text like "High", "Low", etc.)
        risk_categories = [
            ('flood_risk', 'Flood Risk'),
            ('heat_risk', 'Heat Risk'),
            ('air_quality_risk', 'Air Quality Risk')
        ]
        
        print(f"12. RISK CATEGORIES:")
        expected_categories = ['Low', 'Medium', 'High', 'Very Low', 'Very High']
        for key, label in risk_categories:
            value = result.get(key)
            if value:
                status = 'GOOD' if value in expected_categories else 'NEEDS FIXING'
                print(f"   {label}: '{value}' - {status}")
        print()
        
        # 13. Comprehensive Flood Risk (should have combined_score 0-1)
        comp_flood = result.get('comprehensive_flood_risk')
        if comp_flood:
            combined_score = comp_flood.get('combined_score')
            if combined_score is not None:
                print(f"13. COMPREHENSIVE FLOOD RISK:")
                print(f"   Combined Score: {combined_score}")
                print(f"   UI Expects: 0-1 scale (shown as percentage)")
                print(f"   UI Display: {(combined_score * 100):.1f}%")
                print(f"   ✅ Status: {'GOOD' if 0 <= combined_score <= 1 else 'NEEDS FIXING'}")
                print()
        
        # 14. Probabilistic Risk Values
        prob_risk = result.get('probabilistic_risk')
        if prob_risk:
            print(f"14. PROBABILISTIC RISK:")
            expected_risk = prob_risk.get('expected_risk')
            vuln_score = result.get('vulnerability_score')
            
            print(f"   Expected Risk: {expected_risk}")
            print(f"   Vulnerability Score: {vuln_score}")
            print(f"   UI Shows: Vulnerability as percentage")
            if vuln_score:
                print(f"   UI Display: {(vuln_score * 100):.1f}%")
                print(f"   ✅ Vulnerability Status: {'GOOD' if 0 <= vuln_score <= 1 else 'NEEDS FIXING'}")
            print()
        
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {e}")
        return None

if __name__ == "__main__":
    test_all_data_scales()