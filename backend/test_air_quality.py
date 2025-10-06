import requests
import time
import sys

def test_air_quality(location_name, lat, lon):
    """Test air quality API for a specific location"""
    print(f'Testing air quality in {location_name} (lat: {lat}, lon: {lon})...')
    
    url = 'http://localhost:8001/assess-location'
    data = {'latitude': lat, 'longitude': lon}
    
    try:
        response = requests.post(url, json=data, timeout=15)
        if response.status_code == 200:
            result = response.json()
            print(f'\nAir Quality Results for {location_name}:')
            print(f'AQI: {result.get("air_quality_index", "N/A")}')
            print(f'Air Quality Risk: {result.get("air_quality_risk", "N/A")}')
            print(f'PM2.5: {result.get("pm2_5", "N/A")} μg/m³')
            print(f'PM10: {result.get("pm10", "N/A")} μg/m³')
            print(f'NO2: {result.get("no2", "N/A")} μg/m³')
            print(f'O3: {result.get("o3", "N/A")} μg/m³')
            print(f'CO: {result.get("co", "N/A")} μg/m³')
            print('-' * 50)
            return True
        else:
            print(f'Error: HTTP {response.status_code}')
            print(response.text)
            return False
    except Exception as e:
        print(f'Error: {str(e)}')
        return False

def main():
    # Default location is Bangkok
    location_name = "Bangkok"
    lat = 13.7563
    lon = 100.5018
    
    # Check command-line arguments
    if len(sys.argv) > 3:
        location_name = sys.argv[1]
        lat = float(sys.argv[2])
        lon = float(sys.argv[3])
    
    # Test the specified location
    test_air_quality(location_name, lat, lon)
    
    # Also test New York for comparison
    if location_name != "New York":
        test_air_quality("New York", 40.7128, -74.0060)

if __name__ == "__main__":
    main()