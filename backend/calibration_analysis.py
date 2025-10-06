#!/usr/bin/env python3
"""
Calibration Analysis for PM2.5 values from OpenWeatherMap API
This script collects PM2.5 data from OpenWeatherMap for various cities
and compares it with reference data to develop a calibration formula.
"""

import httpx
import asyncio
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test locations - major cities with known air quality issues and clean air cities
# Format: (name, latitude, longitude)
TEST_LOCATIONS = [
    ("Bangkok, Thailand", 13.7563, 100.5018),
    ("Delhi, India", 28.6139, 77.2090),
    ("Los Angeles, USA", 34.0522, -118.2437),
    ("Beijing, China", 39.9042, 116.4074),
    ("London, UK", 51.5074, -0.1278),
    ("Mexico City, Mexico", 19.4326, -99.1332),
    ("Cairo, Egypt", 30.0444, 31.2357),
    ("Zurich, Switzerland", 47.3769, 8.5417),
    ("Sydney, Australia", -33.8688, 151.2093),
    ("Reykjavik, Iceland", 64.1466, -21.9426),
    ("Santiago, Chile", -33.4489, -70.6693),
    ("Johannesburg, South Africa", -26.2041, 28.0473),
    ("Tokyo, Japan", 35.6762, 139.6503),
    ("São Paulo, Brazil", -23.5505, -46.6333),
    ("New York, USA", 40.7128, -74.0060),
    ("Amsterdam, Netherlands", 52.3676, 4.9041),
    ("Mumbai, India", 19.0760, 72.8777),
    ("Singapore", 1.3521, 103.8198),
    ("Oslo, Norway", 59.9139, 10.7522),
    ("Wellington, New Zealand", -41.2924, 174.7787)
]

# Reference data - approximate PM2.5 values from various sources (µg/m³)
# These are estimated averages, as real-time data would need API access to multiple services
# In a real implementation, you would get this data from authoritative sources via API
REFERENCE_DATA = {
    "Bangkok, Thailand": 25.0,
    "Delhi, India": 110.0,
    "Los Angeles, USA": 12.0,
    "Beijing, China": 40.0,
    "London, UK": 15.0,
    "Mexico City, Mexico": 22.0,
    "Cairo, Egypt": 32.0,
    "Zurich, Switzerland": 8.0,
    "Sydney, Australia": 7.0,
    "Reykjavik, Iceland": 3.0,
    "Santiago, Chile": 21.0,
    "Johannesburg, South Africa": 18.0,
    "Tokyo, Japan": 14.0,
    "São Paulo, Brazil": 17.0,
    "New York, USA": 9.0,
    "Amsterdam, Netherlands": 10.0,
    "Mumbai, India": 45.0,
    "Singapore": 19.0,
    "Oslo, Norway": 6.0,
    "Wellington, New Zealand": 5.0
}

# Load OpenWeatherMap API key
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "")
if not OPENWEATHER_API_KEY:
    try:
        with open(".env", "r") as f:
            for line in f:
                if line.startswith("OPENWEATHER_API_KEY="):
                    OPENWEATHER_API_KEY = line.split("=")[1].strip()
                    break
    except Exception as e:
        logger.error(f"Error loading API key from .env file: {e}")

if not OPENWEATHER_API_KEY:
    raise ValueError("OPENWEATHER_API_KEY not found in environment variables or .env file")

async def fetch_air_pollution_data(lat: float, lng: float) -> Dict[str, Any]:
    """Fetch air pollution data from OpenWeatherMap API"""
    url = "http://api.openweathermap.org/data/2.5/air_pollution"
    
    params = {
        "lat": lat,
        "lon": lng,
        "appid": OPENWEATHER_API_KEY
    }
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if "list" not in data or len(data["list"]) == 0:
                raise ValueError(f"Invalid response format from Air Pollution API for {lat}, {lng}")
                
            air_data = data["list"][0]
            components = air_data["components"]
            
            return {
                "pm2_5": components.get("pm2_5", 0),
                "pm10": components.get("pm10", 0),
                "no2": components.get("no2", 0),
                "o3": components.get("o3", 0),
                "co": components.get("co", 0),
                "openweather_aqi": air_data["main"]["aqi"]
            }
    except Exception as e:
        logger.error(f"Error fetching air pollution data for {lat}, {lng}: {e}")
        return {
            "pm2_5": None,
            "pm10": None,
            "no2": None,
            "o3": None,
            "co": None,
            "openweather_aqi": None
        }

async def collect_test_data() -> pd.DataFrame:
    """Collect test data from all locations"""
    tasks = []
    for name, lat, lng in TEST_LOCATIONS:
        tasks.append(fetch_air_pollution_data(lat, lng))
    
    results = await asyncio.gather(*tasks)
    
    data = []
    for i, result in enumerate(results):
        name, lat, lng = TEST_LOCATIONS[i]
        reference_pm25 = REFERENCE_DATA.get(name)
        
        if result["pm2_5"] is not None:
            data.append({
                "city": name,
                "latitude": lat,
                "longitude": lng,
                "openweathermap_pm25": result["pm2_5"],
                "reference_pm25": reference_pm25,
                "pm10": result["pm10"],
                "no2": result["no2"],
                "o3": result["o3"],
                "co": result["co"],
                "openweather_aqi": result["openweather_aqi"]
            })
    
    return pd.DataFrame(data)

def analyze_calibration_factors(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze the collected data and determine calibration factors"""
    # Calculate ratios between reference and OpenWeatherMap values
    df["ratio"] = df["reference_pm25"] / df["openweathermap_pm25"]
    
    # Calculate basic statistics
    stats = {
        "mean_ratio": df["ratio"].mean(),
        "median_ratio": df["ratio"].median(),
        "std_ratio": df["ratio"].std(),
        "min_ratio": df["ratio"].min(),
        "max_ratio": df["ratio"].max(),
        "q1_ratio": df["ratio"].quantile(0.25),
        "q3_ratio": df["ratio"].quantile(0.75)
    }
    
    # Check for outliers using IQR method
    iqr = stats["q3_ratio"] - stats["q1_ratio"]
    lower_bound = stats["q1_ratio"] - (1.5 * iqr)
    upper_bound = stats["q3_ratio"] + (1.5 * iqr)
    
    # Filter out outliers for regression
    df_filtered = df[(df["ratio"] >= lower_bound) & (df["ratio"] <= upper_bound)]
    
    # Try different regression models
    # 1. Simple multiplier (y = ax)
    simple_multiplier = df_filtered["reference_pm25"].sum() / df_filtered["openweathermap_pm25"].sum()
    
    # 2. Linear regression (y = ax + b)
    from sklearn.linear_model import LinearRegression
    X = df_filtered[["openweathermap_pm25"]].values
    y = df_filtered["reference_pm25"].values
    linear_model = LinearRegression().fit(X, y)
    linear_slope = linear_model.coef_[0]
    linear_intercept = linear_model.intercept_
    
    # 3. Polynomial regression (y = ax² + bx + c)
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline
    
    poly_model = make_pipeline(PolynomialFeatures(2), LinearRegression())
    poly_model.fit(X, y)
    poly_coefs = poly_model.named_steps['linearregression'].coef_
    poly_intercept = poly_model.named_steps['linearregression'].intercept_
    
    # Calculate errors for each model
    df_filtered["simple_pred"] = df_filtered["openweathermap_pm25"] * simple_multiplier
    df_filtered["linear_pred"] = linear_model.predict(df_filtered[["openweathermap_pm25"]])
    df_filtered["poly_pred"] = poly_model.predict(df_filtered[["openweathermap_pm25"]])
    
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    simple_mse = mean_squared_error(df_filtered["reference_pm25"], df_filtered["simple_pred"])
    linear_mse = mean_squared_error(df_filtered["reference_pm25"], df_filtered["linear_pred"])
    poly_mse = mean_squared_error(df_filtered["reference_pm25"], df_filtered["poly_pred"])
    
    simple_mae = mean_absolute_error(df_filtered["reference_pm25"], df_filtered["simple_pred"])
    linear_mae = mean_absolute_error(df_filtered["reference_pm25"], df_filtered["linear_pred"])
    poly_mae = mean_absolute_error(df_filtered["reference_pm25"], df_filtered["poly_pred"])
    
    results = {
        "stats": stats,
        "simple_multiplier": simple_multiplier,
        "linear_model": {
            "slope": linear_slope,
            "intercept": linear_intercept,
            "mse": linear_mse,
            "mae": linear_mae
        },
        "polynomial_model": {
            "coefs": poly_coefs,
            "intercept": poly_intercept,
            "mse": poly_mse,
            "mae": poly_mae
        },
        "simple_model": {
            "multiplier": simple_multiplier,
            "mse": simple_mse,
            "mae": simple_mae
        },
        "data": df,
        "filtered_data": df_filtered,
        "outlier_bounds": (lower_bound, upper_bound)
    }
    
    return results

def visualize_results(analysis_results: Dict[str, Any]) -> None:
    """Visualize the calibration results"""
    df = analysis_results["data"]
    df_filtered = analysis_results["filtered_data"]
    
    # 1. Create a scatter plot with both original and calibrated values
    plt.figure(figsize=(14, 8))
    
    # Original comparison
    plt.subplot(2, 2, 1)
    plt.scatter(df["openweathermap_pm25"], df["reference_pm25"], alpha=0.7)
    plt.plot([0, df["openweathermap_pm25"].max()], [0, df["openweathermap_pm25"].max()], 'r--')
    plt.xlabel('OpenWeatherMap PM2.5 (µg/m³)')
    plt.ylabel('Reference PM2.5 (µg/m³)')
    plt.title('Original Values Comparison')
    plt.grid(True, alpha=0.3)
    
    # Add city labels to the points
    for i, row in df.iterrows():
        plt.annotate(row['city'].split(',')[0], 
                     (row['openweathermap_pm25'], row['reference_pm25']),
                     fontsize=8)
    
    # Model comparison plot
    plt.subplot(2, 2, 2)
    
    x_range = np.linspace(0, df["openweathermap_pm25"].max(), 100)
    
    # Simple multiplier model
    simple_multiplier = analysis_results["simple_multiplier"]
    y_simple = simple_multiplier * x_range
    plt.plot(x_range, y_simple, 'r-', label=f'Simple (y = {simple_multiplier:.2f}x)')
    
    # Linear model
    linear_slope = analysis_results["linear_model"]["slope"]
    linear_intercept = analysis_results["linear_model"]["intercept"]
    y_linear = linear_slope * x_range + linear_intercept
    plt.plot(x_range, y_linear, 'g-', label=f'Linear (y = {linear_slope:.2f}x + {linear_intercept:.2f})')
    
    # Polynomial model
    poly_coefs = analysis_results["polynomial_model"]["coefs"]
    poly_intercept = analysis_results["polynomial_model"]["intercept"]
    if len(poly_coefs) >= 3:
        y_poly = poly_coefs[2] * x_range**2 + poly_coefs[1] * x_range + poly_intercept
        plt.plot(x_range, y_poly, 'b-', 
                 label=f'Polynomial (y = {poly_coefs[2]:.4f}x² + {poly_coefs[1]:.2f}x + {poly_intercept:.2f})')
    
    plt.scatter(df_filtered["openweathermap_pm25"], df_filtered["reference_pm25"], alpha=0.7)
    plt.xlabel('OpenWeatherMap PM2.5 (µg/m³)')
    plt.ylabel('Reference PM2.5 (µg/m³)')
    plt.title('Calibration Models')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Residual plots
    plt.subplot(2, 2, 3)
    plt.scatter(df_filtered["openweathermap_pm25"], 
                df_filtered["reference_pm25"] - df_filtered["simple_pred"], 
                alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('OpenWeatherMap PM2.5 (µg/m³)')
    plt.ylabel('Residual (µg/m³)')
    plt.title('Simple Multiplier Residuals')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.scatter(df_filtered["openweathermap_pm25"], 
                df_filtered["reference_pm25"] - df_filtered["linear_pred"], 
                alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('OpenWeatherMap PM2.5 (µg/m³)')
    plt.ylabel('Residual (µg/m³)')
    plt.title('Linear Model Residuals')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('pm25_calibration_analysis.png')
    logger.info("Saved visualization to pm25_calibration_analysis.png")
    
    # Ratio boxplot
    plt.figure(figsize=(10, 6))
    plt.boxplot(df_filtered["ratio"])
    plt.ylabel('Reference PM2.5 / OpenWeatherMap PM2.5')
    plt.title('Distribution of Calibration Ratios')
    plt.grid(True, alpha=0.3)
    plt.savefig('pm25_ratio_distribution.png')
    logger.info("Saved ratio distribution to pm25_ratio_distribution.png")

def recommend_calibration() -> str:
    """Provide a recommendation based on the analysis"""
    # This will be populated by the analysis
    return "Recommendation will be generated after analysis"

async def main():
    logger.info("Starting PM2.5 calibration analysis...")
    
    # Collect test data
    logger.info(f"Collecting data for {len(TEST_LOCATIONS)} locations...")
    df = await collect_test_data()
    
    if len(df) == 0:
        logger.error("No valid data collected. Check API key and network connection.")
        return
    
    logger.info(f"Successfully collected data for {len(df)} locations")
    
    # Save raw data
    df.to_csv("pm25_calibration_data.csv", index=False)
    logger.info("Saved raw data to pm25_calibration_data.csv")
    
    # Analyze data
    logger.info("Analyzing calibration factors...")
    analysis_results = analyze_calibration_factors(df)
    
    # Display results
    logger.info("\n===== ANALYSIS RESULTS =====")
    logger.info(f"Number of data points: {len(df)}")
    logger.info(f"Outlier bounds (ratio): {analysis_results['outlier_bounds']}")
    logger.info(f"Filtered data points: {len(analysis_results['filtered_data'])}")
    
    # Results table
    logger.info("\nCalibration Statistics:")
    for key, value in analysis_results["stats"].items():
        logger.info(f"  {key}: {value:.4f}")
    
    logger.info("\nCalibration Models:")
    logger.info(f"  Simple Multiplier: {analysis_results['simple_multiplier']:.4f}")
    logger.info(f"    - MAE: {analysis_results['simple_model']['mae']:.4f}")
    logger.info(f"    - MSE: {analysis_results['simple_model']['mse']:.4f}")
    
    logger.info(f"  Linear Model: y = {analysis_results['linear_model']['slope']:.4f}x + {analysis_results['linear_model']['intercept']:.4f}")
    logger.info(f"    - MAE: {analysis_results['linear_model']['mae']:.4f}")
    logger.info(f"    - MSE: {analysis_results['linear_model']['mse']:.4f}")
    
    # Create visualizations
    logger.info("\nGenerating visualizations...")
    visualize_results(analysis_results)
    
    # Determine the best model based on MAE
    models = [
        ("Simple Multiplier", analysis_results['simple_model']['mae']),
        ("Linear Regression", analysis_results['linear_model']['mae']),
        ("Polynomial Regression", analysis_results['polynomial_model']['mae'])
    ]
    
    best_model = min(models, key=lambda x: x[1])
    
    logger.info(f"\nBest model: {best_model[0]} (MAE: {best_model[1]:.4f})")
    
    # Provide recommendation
    if best_model[0] == "Simple Multiplier":
        multiplier = analysis_results['simple_multiplier']
        recommendation = f"""
RECOMMENDATION:
Apply a simple multiplier of {multiplier:.2f} to PM2.5 values from OpenWeatherMap.
Implementation in code would be:
```python
# Apply calibration factor to PM2.5
pm2_5 = pm2_5 * {multiplier:.2f}
```
"""
    elif best_model[0] == "Linear Regression":
        slope = analysis_results['linear_model']['slope']
        intercept = analysis_results['linear_model']['intercept']
        recommendation = f"""
RECOMMENDATION:
Apply a linear transformation to PM2.5 values from OpenWeatherMap.
Implementation in code would be:
```python
# Apply calibration using linear model
pm2_5 = {slope:.4f} * pm2_5 + {intercept:.4f}
```
"""
    else:
        coefs = analysis_results['polynomial_model']['coefs']
        intercept = analysis_results['polynomial_model']['intercept']
        recommendation = f"""
RECOMMENDATION:
Apply a polynomial transformation to PM2.5 values from OpenWeatherMap.
Implementation in code would be:
```python
# Apply calibration using polynomial model
pm2_5 = {coefs[2]:.6f} * (pm2_5**2) + {coefs[1]:.4f} * pm2_5 + {intercept:.4f}
```
"""
    
    logger.info(recommendation)
    
    # Save recommendation to file
    with open("pm25_calibration_recommendation.txt", "w") as f:
        f.write("PM2.5 CALIBRATION ANALYSIS\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Data points analyzed: {len(df)}\n")
        f.write(f"Data points after outlier removal: {len(analysis_results['filtered_data'])}\n\n")
        f.write("MODEL COMPARISON:\n")
        for model_name, mae in models:
            f.write(f"  {model_name}: MAE = {mae:.4f}\n")
        f.write("\n")
        f.write(recommendation)
    
    logger.info("Saved recommendation to pm25_calibration_recommendation.txt")

if __name__ == "__main__":
    asyncio.run(main())