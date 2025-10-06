"""
Calibrate heat risk and flood risk thresholds using real-world data from diverse locations
"""

import requests
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import json
import time

# 15 diverse locations with known climate characteristics
TEST_LOCATIONS = [
    # Hot locations with different rainfall
    {"name": "Phoenix, AZ", "lat": 33.4484, "lon": -112.0740, "expected_heat": "High", "expected_flood": "Low"},
    {"name": "Dubai, UAE", "lat": 25.2048, "lon": 55.2708, "expected_heat": "High", "expected_flood": "Low"},
    {"name": "Mumbai, India", "lat": 19.0760, "lon": 72.8777, "expected_heat": "High", "expected_flood": "High"},
    {"name": "Singapore", "lat": 1.3521, "lon": 103.8198, "expected_heat": "High", "expected_flood": "Medium"},
    
    # Moderate temperature locations
    {"name": "London, UK", "lat": 51.5074, "lon": -0.1278, "expected_heat": "Low", "expected_flood": "Medium"},
    {"name": "Tokyo, Japan", "lat": 35.6762, "lon": 139.6503, "expected_heat": "Medium", "expected_flood": "Medium"},
    {"name": "Sydney, Australia", "lat": -33.8688, "lon": 151.2093, "expected_heat": "Medium", "expected_flood": "Low"},
    {"name": "Seattle, WA", "lat": 47.6062, "lon": -122.3321, "expected_heat": "Low", "expected_flood": "High"},
    
    # Cold locations
    {"name": "Oslo, Norway", "lat": 59.9139, "lon": 10.7522, "expected_heat": "Low", "expected_flood": "Low"},
    {"name": "Moscow, Russia", "lat": 55.7558, "lon": 37.6173, "expected_heat": "Low", "expected_flood": "Medium"},
    
    # Tropical/Monsoon
    {"name": "Bangkok, Thailand", "lat": 13.7563, "lon": 100.5018, "expected_heat": "High", "expected_flood": "High"},
    {"name": "Dhaka, Bangladesh", "lat": 23.8103, "lon": 90.4125, "expected_heat": "High", "expected_flood": "High"},
    
    # Dry/Desert
    {"name": "Cairo, Egypt", "lat": 30.0444, "lon": 31.2357, "expected_heat": "High", "expected_flood": "Low"},
    
    # Temperate with high rainfall
    {"name": "Vancouver, Canada", "lat": 49.2827, "lon": -123.1207, "expected_heat": "Low", "expected_flood": "High"},
    {"name": "Wellington, NZ", "lat": -41.2865, "lon": 174.7762, "expected_heat": "Low", "expected_flood": "Medium"},
]

def fetch_nasa_climate_data(lat, lon):
    """Fetch actual climate data from NASA POWER API"""
    # Use climatology endpoint which is more reliable
    url = "https://power.larc.nasa.gov/api/temporal/climatology/point"
    params = {
        "parameters": "PRECTOTCORR,T2M",  # Use PRECTOTCORR instead of PRECTOT
        "community": "AG",  # Agricultural community
        "longitude": lon,
        "latitude": lat,
        "start": "2020",
        "end": "2023",
        "format": "JSON"
    }
    
    try:
        print(f"  Requesting: {url}")
        print(f"  Params: lat={lat}, lon={lon}")
        response = requests.get(url, params=params, timeout=30)
        
        print(f"  Status code: {response.status_code}")
        
        if response.status_code != 200:
            print(f"  Response text: {response.text[:200]}")
            return None
            
        response.raise_for_status()
        data = response.json()
        
        # Check response structure
        if "properties" not in data:
            print(f"  ERROR: No 'properties' in response")
            print(f"  Response keys: {data.keys()}")
            return None
        
        properties = data.get("properties", {})
        parameter_data = properties.get("parameter", {})
        
        rainfall_data = parameter_data.get("PRECTOTCORR", {})
        temp_data = parameter_data.get("T2M", {})
        
        if not rainfall_data or not temp_data:
            print(f"  ERROR: Missing data - rainfall: {bool(rainfall_data)}, temp: {bool(temp_data)}")
            return None
        
        # Filter out invalid values and ensure they're numeric
        valid_rainfall = [v for v in rainfall_data.values() if isinstance(v, (int, float)) and v != -999 and v >= 0]
        valid_temp = [v for v in temp_data.values() if isinstance(v, (int, float)) and v != -999]
        
        if not valid_rainfall or not valid_temp:
            print(f"  ERROR: No valid data - rainfall count: {len(valid_rainfall)}, temp count: {len(valid_temp)}")
            return None
        
        avg_rainfall = sum(valid_rainfall) / len(valid_rainfall)
        avg_temp = sum(valid_temp) / len(valid_temp)
        max_temp = max(valid_temp)
        
        return {
            "avg_rainfall_mm": avg_rainfall,
            "avg_temp_c": avg_temp,
            "max_temp_c": max_temp
        }
    except requests.Timeout:
        print(f"  ERROR: Request timeout")
        return None
    except requests.RequestException as e:
        print(f"  ERROR: Request failed: {e}")
        return None
    except Exception as e:
        print(f"  ERROR: Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return None

def encode_risk_level(risk_level):
    """Convert risk level to numeric value for training"""
    mapping = {"Low": 0, "Medium": 1, "High": 2}
    return mapping.get(risk_level, 1)

def decode_risk_level(numeric_value):
    """Convert numeric value back to risk level"""
    if numeric_value <= 0.5:
        return "Low"
    elif numeric_value <= 1.5:
        return "Medium"
    else:
        return "High"

def analyze_climate_patterns():
    """Fetch data for all locations and analyze patterns"""
    print("Fetching climate data for 15 diverse locations...")
    print("=" * 80)
    
    locations_data = []
    
    for i, location in enumerate(TEST_LOCATIONS, 1):
        print(f"\n[{i}/15] Fetching data for {location['name']}...")
        climate_data = fetch_nasa_climate_data(location['lat'], location['lon'])
        
        if climate_data and climate_data['avg_rainfall_mm'] and climate_data['avg_temp_c']:
            location_info = {
                **location,
                **climate_data
            }
            locations_data.append(location_info)
            
            print(f"  ✓ Temperature: {climate_data['avg_temp_c']:.1f}°C (Max: {climate_data['max_temp_c']:.1f}°C)")
            print(f"  ✓ Rainfall: {climate_data['avg_rainfall_mm']:.1f}mm/month")
            print(f"  ✓ Expected Heat Risk: {location['expected_heat']}")
            print(f"  ✓ Expected Flood Risk: {location['expected_flood']}")
        else:
            print(f"  ✗ Failed to fetch data")
        
        # Rate limiting
        time.sleep(2)
    
    print("\n" + "=" * 80)
    print(f"\nSuccessfully collected data for {len(locations_data)} locations")
    
    return locations_data

def train_heat_risk_model(locations_data):
    """Train a model to determine optimal heat risk thresholds"""
    print("\n" + "=" * 80)
    print("TRAINING HEAT RISK MODEL")
    print("=" * 80)
    
    # Prepare training data
    X_temp = np.array([[loc['avg_temp_c'], loc['max_temp_c']] for loc in locations_data])
    y_heat = np.array([encode_risk_level(loc['expected_heat']) for loc in locations_data])
    
    # Try different threshold combinations
    temps = [loc['avg_temp_c'] for loc in locations_data]
    
    # Sort by temperature
    sorted_temps = sorted(zip(temps, y_heat))
    
    print("\nTemperature Analysis:")
    print("-" * 40)
    for temp, risk in sorted_temps:
        print(f"  {temp:5.1f}°C → {decode_risk_level(risk)}")
    
    # Find optimal thresholds using percentile analysis
    low_temps = [t for t, r in zip(temps, y_heat) if r == 0]  # Low risk
    medium_temps = [t for t, r in zip(temps, y_heat) if r == 1]  # Medium risk
    high_temps = [t for t, r in zip(temps, y_heat) if r == 2]  # High risk
    
    if low_temps and medium_temps:
        threshold_low_medium = (max(low_temps) + min(medium_temps)) / 2
    else:
        threshold_low_medium = 20.0
    
    if medium_temps and high_temps:
        threshold_medium_high = (max(medium_temps) + min(high_temps)) / 2
    else:
        threshold_medium_high = 28.0
    
    print("\n" + "-" * 40)
    print("OPTIMAL HEAT RISK THRESHOLDS:")
    print("-" * 40)
    print(f"  Low → Medium:  {threshold_low_medium:.1f}°C")
    print(f"  Medium → High: {threshold_medium_high:.1f}°C")
    
    # Calculate accuracy
    correct = 0
    for temp, actual_risk in zip(temps, y_heat):
        if temp < threshold_low_medium:
            predicted = 0
        elif temp < threshold_medium_high:
            predicted = 1
        else:
            predicted = 2
        
        if predicted == actual_risk:
            correct += 1
    
    accuracy = (correct / len(temps)) * 100
    print(f"\nModel Accuracy: {accuracy:.1f}%")
    
    return threshold_low_medium, threshold_medium_high

def train_flood_risk_model(locations_data):
    """Train a model to determine optimal flood risk thresholds"""
    print("\n" + "=" * 80)
    print("TRAINING FLOOD RISK MODEL")
    print("=" * 80)
    
    # Prepare training data
    rainfall = [loc['avg_rainfall_mm'] for loc in locations_data]
    y_flood = np.array([encode_risk_level(loc['expected_flood']) for loc in locations_data])
    
    # Sort by rainfall
    sorted_rainfall = sorted(zip(rainfall, y_flood))
    
    print("\nRainfall Analysis:")
    print("-" * 40)
    for rain, risk in sorted_rainfall:
        print(f"  {rain:6.1f}mm/month → {decode_risk_level(risk)}")
    
    # Find optimal thresholds
    low_rainfall = [r for r, risk in zip(rainfall, y_flood) if risk == 0]  # Low risk
    medium_rainfall = [r for r, risk in zip(rainfall, y_flood) if risk == 1]  # Medium risk
    high_rainfall = [r for r, risk in zip(rainfall, y_flood) if risk == 2]  # High risk
    
    if low_rainfall and (medium_rainfall or high_rainfall):
        threshold_low_medium = max(low_rainfall) if not medium_rainfall else (max(low_rainfall) + min(medium_rainfall)) / 2
    else:
        threshold_low_medium = 60.0
    
    if medium_rainfall and high_rainfall:
        threshold_medium_high = (max(medium_rainfall) + min(high_rainfall)) / 2
    elif high_rainfall:
        threshold_medium_high = min(high_rainfall)
    else:
        threshold_medium_high = 120.0
    
    print("\n" + "-" * 40)
    print("OPTIMAL FLOOD RISK THRESHOLDS:")
    print("-" * 40)
    print(f"  Low → Medium:  {threshold_low_medium:.1f}mm/month")
    print(f"  Medium → High: {threshold_medium_high:.1f}mm/month")
    
    # Calculate accuracy
    correct = 0
    for rain, actual_risk in zip(rainfall, y_flood):
        if rain < threshold_low_medium:
            predicted = 0
        elif rain < threshold_medium_high:
            predicted = 1
        else:
            predicted = 2
        
        if predicted == actual_risk:
            correct += 1
    
    accuracy = (correct / len(rainfall)) * 100
    print(f"\nModel Accuracy: {accuracy:.1f}%")
    
    return threshold_low_medium, threshold_medium_high

def generate_code_snippet(heat_low_med, heat_med_high, flood_low_med, flood_med_high):
    """Generate the updated code snippet"""
    print("\n" + "=" * 80)
    print("UPDATED CODE FOR main.py")
    print("=" * 80)
    
    code = f'''
def calculate_risk_levels(rainfall: float, temperature: float) -> Dict[str, str]:
    """
    Calculate flood and heat risk levels based on climate data
    Thresholds calibrated using real-world data from 15 diverse locations
    """
    
    # Heat risk thresholds (calibrated from global climate data)
    if temperature > {heat_med_high:.1f}:
        heat_risk = "High"
    elif temperature > {heat_low_med:.1f}:
        heat_risk = "Medium"
    else:
        heat_risk = "Low"
    
    # Flood risk thresholds (calibrated from global precipitation data)
    if rainfall > {flood_med_high:.1f}:
        flood_risk = "High"
    elif rainfall > {flood_low_med:.1f}:
        flood_risk = "Medium"
    else:
        flood_risk = "Low"
    
    return {{
        "flood_risk": flood_risk,
        "heat_risk": heat_risk
    }}
'''
    
    print(code)
    
    return {
        "heat_thresholds": {
            "low_to_medium": heat_low_med,
            "medium_to_high": heat_med_high
        },
        "flood_thresholds": {
            "low_to_medium": flood_low_med,
            "medium_to_high": flood_med_high
        }
    }

def main():
    """Main calibration workflow"""
    print("\n" + "=" * 80)
    print("CLIMATE RISK THRESHOLD CALIBRATION")
    print("=" * 80)
    print("\nThis script will:")
    print("1. Fetch real climate data from NASA POWER API for 15 diverse locations")
    print("2. Analyze temperature patterns to determine heat risk thresholds")
    print("3. Analyze rainfall patterns to determine flood risk thresholds")
    print("4. Generate optimized code for main.py")
    print("\n" + "=" * 80)
    
    # Fetch data
    locations_data = analyze_climate_patterns()
    
    if len(locations_data) < 10:
        print("\n⚠️  Warning: Only collected data for {len(locations_data)} locations.")
        print("Threshold calibration may be less accurate.")
    
    # Train models
    heat_low_med, heat_med_high = train_heat_risk_model(locations_data)
    flood_low_med, flood_med_high = train_flood_risk_model(locations_data)
    
    # Generate code
    thresholds = generate_code_snippet(heat_low_med, heat_med_high, flood_low_med, flood_med_high)
    
    # Save results
    output_file = "calibration_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "locations_data": locations_data,
            "thresholds": thresholds,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
    
    print("\n" + "=" * 80)
    print(f"✓ Results saved to {output_file}")
    print("=" * 80)
    
    print("\n📊 SUMMARY:")
    print(f"  • Analyzed {len(locations_data)} locations worldwide")
    print(f"  • Heat risk thresholds: {heat_low_med:.1f}°C, {heat_med_high:.1f}°C")
    print(f"  • Flood risk thresholds: {flood_low_med:.1f}mm, {flood_med_high:.1f}mm")
    print("\n✓ Copy the code above and replace calculate_risk_levels() in main.py")

if __name__ == "__main__":
    main()
