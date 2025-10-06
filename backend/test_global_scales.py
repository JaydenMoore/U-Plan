#!/usr/bin/env python3
"""
Test multiple global locations to verify consistent scaling
"""

import requests
import json

def test_location(lat, lon, name):
    """Test a specific location"""
    url = "http://localhost:8001/assess-location-enhanced"
    data = {"latitude": lat, "longitude": lon}
    
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        result = response.json()
        
        print(f"\n=== {name.upper()} ({lat}, {lon}) ===")
        
        # Key metrics
        overall = result.get('overall_risk_score', 0)
        water_risk = result.get('water_risk', {})
        aqi = result.get('air_quality_index', 0)
        
        print(f"Overall Risk: {overall:.2f}/10")
        
        if water_risk:
            print(f"Water Stress: {water_risk.get('baseline_water_stress', 0)}/5")
            print(f"Water Category: {water_risk.get('overall_category', 'Unknown')}")
            print(f"Country: {water_risk.get('country', 'Unknown')}")
        
        print(f"AQI: {aqi}")
        print(f"Temperature: {result.get('temperature_c', 0):.1f}°C")
        print(f"Rainfall: {result.get('rainfall_mm', 0):.1f} mm/month")
        
        # Check for scale violations
        violations = []
        if not (0 <= overall <= 10):
            violations.append(f"Overall risk {overall} not in 0-10 range")
        if not (0 <= aqi <= 500):
            violations.append(f"AQI {aqi} not in 0-500 range")
        if water_risk:
            for key, max_val in [('baseline_water_stress', 5), ('drought_risk', 5), ('overall_water_risk', 5)]:
                val = water_risk.get(key, 0)
                if not (0 <= val <= max_val):
                    violations.append(f"Water {key} {val} not in 0-{max_val} range")
        
        if violations:
            print(f"❌ SCALE VIOLATIONS: {', '.join(violations)}")
        else:
            print("✅ All scales OK")
            
        return len(violations) == 0
        
    except Exception as e:
        print(f"❌ Error testing {name}: {e}")
        return False

def test_global_locations():
    """Test multiple global locations"""
    locations = [
        (34.0522, -118.2437, "Los Angeles, CA"),  # California - high water stress
        (40.7128, -74.0060, "New York City"),     # East Coast US
        (51.5074, -0.1278, "London, UK"),         # Europe
        (35.6762, 139.6503, "Tokyo, Japan"),      # Asia
        (-33.8688, 151.2093, "Sydney, Australia"), # Australia
        (19.4326, -99.1332, "Mexico City"),       # Latin America - high altitude
        (30.0444, 31.2357, "Cairo, Egypt"),       # North Africa - arid
        (-1.2921, 36.8219, "Nairobi, Kenya"),     # East Africa
        (55.7558, 37.6173, "Moscow, Russia"),     # Northern region
        (1.3521, 103.8198, "Singapore"),          # Tropical SE Asia
    ]
    
    print("=== GLOBAL LOCATION SCALE VALIDATION ===")
    
    passed = 0
    total = len(locations)
    
    for lat, lon, name in locations:
        success = test_location(lat, lon, name)
        if success:
            passed += 1
    
    print(f"\n=== SUMMARY ===")
    print(f"Locations tested: {total}")
    print(f"Passed validation: {passed}")
    print(f"Failed validation: {total - passed}")
    print(f"Success rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 ALL LOCATIONS PASS SCALE VALIDATION!")
    else:
        print("⚠️  Some locations have scale issues that need fixing.")

if __name__ == "__main__":
    test_global_locations()