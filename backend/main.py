from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import httpx
import asyncio
from typing import Dict, Any, List
import logging
from functools import lru_cache
import time

# OpenWeatherMap API configuration
OPENWEATHER_API_KEY = "28e7cf23db337a4fcb983071b8d85b2b"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Urban Planner AI", description="NASA-powered environmental risk assessment")

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LocationRequest(BaseModel):
    latitude: float
    longitude: float

class BulkLocationRequest(BaseModel):
    locations: List[LocationRequest]

class CoordinateRequest(BaseModel):
    lat: float
    lng: float
    gridSize: float = None

class BulkCoordinateRequest(BaseModel):
    coordinates: List[CoordinateRequest]

class RiskAssessment(BaseModel):
    latitude: float
    longitude: float
    rainfall_mm: float
    temperature_c: float
    flood_risk: str
    heat_risk: str
    air_quality_index: int  # Calculated US EPA AQI (0-500)
    air_quality_risk: str
    pm2_5: float
    pm10: float
    no2: float
    o3: float
    co: float
    overall_risk_score: float
    summary: str

@lru_cache(maxsize=1000)
def fetch_nasa_power_data_cached(lat_rounded: float, lon_rounded: float) -> Dict[str, Any]:
    """Fetch climate data from NASA POWER API with caching (rounded coordinates)"""
    return fetch_nasa_power_data_direct(lat_rounded, lon_rounded)

def fetch_nasa_power_data(lat: float, lon: float) -> Dict[str, Any]:
    """Fetch climate data from NASA POWER API with coordinate rounding for caching"""
    # Round coordinates to 2 decimal places to enable caching for nearby points
    lat_rounded = round(lat, 2)
    lon_rounded = round(lon, 2)
    return fetch_nasa_power_data_cached(lat_rounded, lon_rounded)

def fetch_nasa_power_data_direct(lat: float, lon: float) -> Dict[str, Any]:
    """Direct fetch from NASA POWER API"""
    try:
        # NASA POWER API endpoint for monthly data (last 5 years)
        url = "https://power.larc.nasa.gov/api/temporal/monthly/point"
        params = {
            "parameters": "PRECTOT,T2M",  # Precipitation total, Temperature at 2m
            "community": "RE",  # Renewable Energy community
            "longitude": lon,
            "latitude": lat,
            "start": "2019",
            "end": "2023",
            "format": "JSON"
        }
        
        logger.info(f"Fetching NASA data for coordinates: {lat}, {lon}")
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        # Extract and average the data
        properties = data.get("properties", {})
        parameter_data = properties.get("parameter", {})
        
        # Calculate averages
        rainfall_data = parameter_data.get("PRECTOT", {})
        temp_data = parameter_data.get("T2M", {})
        
        if not rainfall_data or not temp_data:
            # Provide fallback values if data is missing
            rainfall_data = rainfall_data or {"default": 50.0}
            temp_data = temp_data or {"default": 15.0}
        
        # Filter out invalid values (NASA uses -999 for missing data)
        valid_rainfall = [v for v in rainfall_data.values() if v != -999]
        valid_temp = [v for v in temp_data.values() if v != -999]
        
        avg_rainfall = sum(valid_rainfall) / len(valid_rainfall) if valid_rainfall else 50.0
        avg_temp = sum(valid_temp) / len(valid_temp) if valid_temp else 15.0
        
        return {
            "rainfall_mm": round(avg_rainfall, 2),
            "temperature_c": round(avg_temp, 2)
        }
        
    except requests.RequestException as e:
        logger.error(f"NASA API request failed: {e}")
        raise HTTPException(status_code=503, detail=f"Unable to fetch NASA data: {str(e)}")
    except Exception as e:
        logger.error(f"Data processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing climate data: {str(e)}")

def calculate_risk_levels(rainfall: float, temperature: float) -> Dict[str, str]:
    """Calculate flood and heat risk levels based on climate data"""
    
    # Simple risk assessment logic (can be enhanced)
    if rainfall > 150:
        flood_risk = "High"
    elif rainfall > 80:
        flood_risk = "Medium"
    else:
        flood_risk = "Low"
    
    if temperature > 32:
        heat_risk = "High"
    elif temperature > 25:
        heat_risk = "Medium"
    else:
        heat_risk = "Low"
    
    return {
        "flood_risk": flood_risk,
        "heat_risk": heat_risk
    }

def calculate_air_quality_risk(aqi: int, pm2_5: float, pm10: float) -> str:
    """Calculate air quality risk level based on AQI and particulate matter"""
    
    # US EPA AQI standards: 0-50 Good, 51-100 Moderate, 101-150 Unhealthy for Sensitive, 151-200 Unhealthy, 201-300 Very Unhealthy, 301+ Hazardous
    if aqi >= 201 or pm2_5 > 150 or pm10 > 350:
        return "Very High"
    elif aqi >= 151 or pm2_5 > 55 or pm10 > 250:
        return "High"
    elif aqi >= 101 or pm2_5 > 35 or pm10 > 150:
        return "Medium"
    elif aqi >= 51 or pm2_5 > 12 or pm10 > 50:
        return "Low"
    else:
        return "Very Low"

def calculate_overall_risk_score(flood_risk: str, heat_risk: str, air_quality_risk: str) -> float:
    """Calculate overall risk score (0-10) based on individual risk factors"""
    
    # Risk level to numeric mapping
    risk_values = {
        "Very Low": 1,
        "Low": 2.5,
        "Medium": 5,
        "High": 8,
        "Very High": 10
    }
    
    # Get numeric values for each risk
    flood_val = risk_values.get(flood_risk, 5)
    heat_val = risk_values.get(heat_risk, 5)
    air_val = risk_values.get(air_quality_risk, 5)
    
    # Weighted average: flood 35%, heat 35%, air quality 30%
    overall_score = (flood_val * 0.35) + (heat_val * 0.35) + (air_val * 0.30)
    
    return round(overall_score, 1)

def generate_summary(rainfall: float, temperature: float, flood_risk: str, heat_risk: str, 
                    air_quality_risk: str, aqi: int, pm2_5: float, overall_risk: float) -> str:
    """Generate a comprehensive summary for urban planning"""
    
    summary_parts = []
    
    # Climate conditions
    if temperature > 35:
        summary_parts.append("Extreme heat conditions require robust cooling infrastructure and heat island mitigation.")
    elif temperature > 30:
        summary_parts.append("Hot climate requires cooling infrastructure and energy-efficient building design.")
    elif temperature < 5:
        summary_parts.append("Cold climate requires heating systems and weatherization measures.")
    elif temperature < 10:
        summary_parts.append("Cool climate requires heating considerations and insulation standards.")
    else:
        summary_parts.append("Moderate temperature conditions support diverse development options.")
    
    # Rainfall conditions
    if rainfall > 150:
        summary_parts.append("High rainfall area - implement comprehensive stormwater management, flood-resistant construction, and elevated foundations.")
    elif rainfall > 100:
        summary_parts.append("Above-average rainfall - ensure robust drainage systems and consider permeable surfaces.")
    elif rainfall < 30:
        summary_parts.append("Arid conditions - plan for water conservation, drought-resistant landscaping, and water storage.")
    elif rainfall < 50:
        summary_parts.append("Low rainfall area - consider water supply planning and efficient irrigation systems.")
    else:
        summary_parts.append("Moderate rainfall conditions support standard urban development practices.")
    
    # Air quality conditions
    if air_quality_risk == "High":
        summary_parts.append(f"🏭 HIGH AIR POLLUTION RISK (AQI {aqi}, PM2.5: {pm2_5:.1f}): Air filtration systems, green barriers, and emission controls essential.")
    elif air_quality_risk == "Medium":
        summary_parts.append(f"🌫️ MODERATE AIR POLLUTION (AQI {aqi}): Consider air quality monitoring and green infrastructure.")
    elif air_quality_risk == "Low":
        summary_parts.append(f"🌿 GOOD AIR QUALITY (AQI {aqi}): Favorable conditions for outdoor activities and development.")
    else:
        summary_parts.append(f"✨ EXCELLENT AIR QUALITY (AQI {aqi}): Optimal conditions for all development types.")
    
    # Risk warnings and recommendations
    if flood_risk == "High":
        summary_parts.append("⚠️ HIGH FLOOD RISK: Mandatory flood mitigation measures, restrict development in flood-prone areas.")
    elif flood_risk == "Medium":
        summary_parts.append("⚡ MODERATE FLOOD RISK: Implement flood-resistant design and drainage improvements.")
    
    if heat_risk == "High":
        summary_parts.append("🌡️ HIGH HEAT RISK: Urban heat island mitigation, green infrastructure, and cooling centers recommended.")
    elif heat_risk == "Medium":
        summary_parts.append("☀️ MODERATE HEAT RISK: Consider heat management strategies and green building standards.")
    
    # Overall recommendation based on combined risk score
    if overall_risk >= 7:
        summary_parts.append(f"⚠️ HIGH OVERALL RISK (Score: {overall_risk}/10): Comprehensive mitigation strategies required before development.")
    elif overall_risk >= 5:
        summary_parts.append(f"⚡ MODERATE OVERALL RISK (Score: {overall_risk}/10): Enhanced planning and risk management recommended.")
    elif overall_risk >= 3:
        summary_parts.append(f"✅ LOW OVERALL RISK (Score: {overall_risk}/10): Standard development practices with basic precautions.")
    else:
        summary_parts.append(f"🌟 VERY LOW OVERALL RISK (Score: {overall_risk}/10): Excellent conditions for urban development.")
    
    return " ".join(summary_parts)

@app.get("/")
async def root():
    return {"message": "Urban Planner AI API - Ready to assess environmental risks!"}

@app.post("/assess-location", response_model=RiskAssessment)
async def assess_location(location: LocationRequest):
    """
    Assess environmental risks for a given location using NASA data and air quality data
    """
    try:
        # Validate coordinates
        if not (-90 <= location.latitude <= 90):
            raise HTTPException(status_code=400, detail="Invalid latitude")
        if not (-180 <= location.longitude <= 180):
            raise HTTPException(status_code=400, detail="Invalid longitude")
        
        # Fetch data from multiple sources concurrently
        nasa_data_task = fetch_nasa_power_data_async(location.latitude, location.longitude)
        air_data_task = fetch_air_pollution_data_async(location.latitude, location.longitude)
        
        # Wait for both API calls to complete
        climate_data, air_data = await asyncio.gather(nasa_data_task, air_data_task)
        
        # Calculate risk levels
        risks = calculate_risk_levels(
            climate_data["rainfall_mm"], 
            climate_data["temperature_c"]
        )
        
        # Calculate air quality risk
        air_quality_risk = calculate_air_quality_risk(
            air_data["aqi"],
            air_data["pm2_5"],
            air_data["pm10"]
        )
        
        # Calculate overall risk score
        overall_risk = calculate_overall_risk_score(
            risks["flood_risk"],
            risks["heat_risk"],
            air_quality_risk
        )
        
        # Generate comprehensive summary
        summary = generate_summary(
            climate_data["rainfall_mm"],
            climate_data["temperature_c"],
            risks["flood_risk"],
            risks["heat_risk"],
            air_quality_risk,
            air_data["aqi"],
            air_data["pm2_5"],
            overall_risk
        )
        
        return RiskAssessment(
            latitude=location.latitude,
            longitude=location.longitude,
            rainfall_mm=climate_data["rainfall_mm"],
            temperature_c=climate_data["temperature_c"],
            flood_risk=risks["flood_risk"],
            heat_risk=risks["heat_risk"],
            air_quality_index=air_data["aqi"],
            air_quality_risk=air_quality_risk,
            pm2_5=air_data["pm2_5"],
            pm10=air_data["pm10"],
            no2=air_data["no2"],
            o3=air_data["o3"],
            co=air_data["co"],
            overall_risk_score=overall_risk,
            summary=summary
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/assess-risk", response_model=RiskAssessment)
async def assess_risk(location: LocationRequest):
    """
    Assess environmental risks for a given location using NASA data (alias for assess-location)
    """
    return await assess_location(location)

async def fetch_nasa_power_data_async(lat: float, lng: float) -> Dict[str, float]:
    """Async version of NASA POWER API data fetching using httpx"""
    try:
        # Round coordinates for caching (1 decimal = ~11km precision for better cache hits)
        lat_rounded = round(lat, 1)
        lng_rounded = round(lng, 1)
        cache_key = f"{lat_rounded},{lng_rounded}"
        
        # Check cache first
        if cache_key in fetch_nasa_power_data_async.cache:
            logger.info(f"Cache hit for coordinates {lat_rounded}, {lng_rounded}")
            return fetch_nasa_power_data_async.cache[cache_key]
        
        # NASA POWER API endpoint for climate data
        url = "https://power.larc.nasa.gov/api/temporal/climatology/point"
        
        params = {
            "start": "2020",
            "end": "2023",
            "latitude": lat,
            "longitude": lng,
            "community": "AG",
            "parameters": "PRECTOTCORR,T2M",  # Precipitation, Temperature
            "format": "JSON"
        }
        
        # Try with retries and exponential backoff
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                timeout_duration = 20.0 + (attempt * 5.0)  # 20, 25, 30 seconds
                async with httpx.AsyncClient(timeout=timeout_duration) as client:
                    response = await client.get(url, params=params)
                    response.raise_for_status()
                    data = response.json()
                    break  # Success, exit retry loop
            except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
                if attempt == max_retries:
                    # Final attempt failed, use fallback
                    logger.warning(f"NASA API failed after {max_retries + 1} attempts for {lat}, {lng}: {e}")
                    return get_fallback_climate_data(lat, lng)
                else:
                    # Wait before retry
                    await asyncio.sleep(2.0 * (attempt + 1))
                    logger.info(f"Retrying NASA API for {lat}, {lng} (attempt {attempt + 2})")
        
        # Extract and process the data
        if "properties" not in data or "parameter" not in data["properties"]:
            raise ValueError("Invalid response format from NASA API")
            
        parameters = data["properties"]["parameter"]
        
        # Extract precipitation and temperature data
        prec_data = parameters.get("PRECTOTCORR", {})
        temp_data = parameters.get("T2M", {})
        
        # Calculate averages
        valid_rainfall = [v for v in prec_data.values() if isinstance(v, (int, float)) and v >= 0]
        valid_temp = [v for v in temp_data.values() if isinstance(v, (int, float))]
        
        avg_rainfall = sum(valid_rainfall) / len(valid_rainfall) if valid_rainfall else 50.0
        avg_temp = sum(valid_temp) / len(valid_temp) if valid_temp else 15.0
        
        result = {
            "rainfall_mm": round(avg_rainfall, 2),
            "temperature_c": round(avg_temp, 2)
        }
        
        # Cache the result
        fetch_nasa_power_data_async.cache[cache_key] = result
        logger.info(f"Fetched and cached NASA data for {lat_rounded}, {lng_rounded}")
        
        return result
        
    except httpx.TimeoutException:
        logger.error(f"NASA API timeout for coordinates {lat}, {lng}")
        raise HTTPException(status_code=503, detail=f"NASA API timeout for coordinates {lat}, {lng}")
    except httpx.HTTPStatusError as e:
        logger.error(f"NASA API HTTP error for coordinates {lat}, {lng}: {e}")
        raise HTTPException(status_code=503, detail=f"NASA API error: {e}")
    except Exception as e:
        logger.error(f"NASA API error for coordinates {lat}, {lng}: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing climate data: {str(e)}")

def get_fallback_climate_data(lat: float, lng: float) -> Dict[str, float]:
    """Provide reasonable fallback climate data when NASA API fails"""
    # Use rough climate estimates based on latitude
    if abs(lat) > 60:  # Arctic/Antarctic
        temp = -5.0 + (abs(lat) - 60) * -0.5
        rainfall = 200.0
    elif abs(lat) > 40:  # Temperate
        temp = 15.0 - (abs(lat) - 40) * 0.3
        rainfall = 600.0
    elif abs(lat) > 23.5:  # Subtropical
        temp = 22.0 + (30 - abs(lat)) * 0.3
        rainfall = 800.0
    else:  # Tropical
        temp = 26.0
        rainfall = 1200.0
    
    # Adjust for longitude (rough continental effects)
    if lng < -60 or lng > 120:  # Americas or East Asia - more variable
        rainfall *= 0.8
    
    logger.info(f"Using fallback climate data for {lat}, {lng}: {temp}°C, {rainfall}mm")
    return {
        "rainfall_mm": round(rainfall, 2),
        "temperature_c": round(temp, 2)
    }

# Initialize cache for the async function
fetch_nasa_power_data_async.cache = {}

async def fetch_air_pollution_data_async(lat: float, lng: float) -> Dict[str, float]:
    """Fetch air pollution data from OpenWeatherMap API"""
    try:
        # Round coordinates for caching
        lat_rounded = round(lat, 1)
        lng_rounded = round(lng, 1)
        cache_key = f"air_{lat_rounded},{lng_rounded}"
        
        # Check cache first
        if cache_key in fetch_air_pollution_data_async.cache:
            logger.info(f"Air pollution cache hit for coordinates {lat_rounded}, {lng_rounded}")
            return fetch_air_pollution_data_async.cache[cache_key]
        
        # OpenWeatherMap Air Pollution API endpoint
        url = "http://api.openweathermap.org/data/2.5/air_pollution"
        
        params = {
            "lat": lat,
            "lon": lng,
            "appid": OPENWEATHER_API_KEY
        }
        
        # Try with retries
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                timeout_duration = 10.0 + (attempt * 3.0)  # 10, 13, 16 seconds
                async with httpx.AsyncClient(timeout=timeout_duration) as client:
                    response = await client.get(url, params=params)
                    response.raise_for_status()
                    data = response.json()
                    break
            except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
                if attempt == max_retries:
                    logger.warning(f"Air pollution API failed after {max_retries + 1} attempts for {lat}, {lng}: {e}")
                    return get_fallback_air_quality_data(lat, lng)
                else:
                    await asyncio.sleep(1.0 * (attempt + 1))
                    logger.info(f"Retrying air pollution API for {lat}, {lng} (attempt {attempt + 2})")
        
        # Extract air quality data
        if "list" not in data or len(data["list"]) == 0:
            raise ValueError("Invalid response format from Air Pollution API")
            
        air_data = data["list"][0]
        openweather_aqi = air_data["main"]["aqi"]  # OpenWeatherMap AQI (1-5)
        components = air_data["components"]
        
        # Extract key pollutants
        pm2_5 = components.get("pm2_5", 0)
        pm10 = components.get("pm10", 0)
        no2 = components.get("no2", 0)
        o3 = components.get("o3", 0)
        co = components.get("co", 0)
        
        # Calculate proper AQI based on PM2.5 (US EPA standard)
        calculated_aqi = calculate_aqi_from_pm25(pm2_5)
        
        result = {
            "aqi": calculated_aqi,
            "openweather_aqi": openweather_aqi,
            "pm2_5": pm2_5,
            "pm10": pm10,
            "no2": no2,
            "o3": o3,
            "co": co
        }
        
        # Cache the result
        fetch_air_pollution_data_async.cache[cache_key] = result
        logger.info(f"Fetched and cached air pollution data for {lat_rounded}, {lng_rounded}: AQI={calculated_aqi} (PM2.5: {pm2_5}μg/m³)")
        
        return result
        
    except Exception as e:
        logger.error(f"Air pollution API error for coordinates {lat}, {lng}: {e}")
        return get_fallback_air_quality_data(lat, lng)

# Initialize cache for air pollution function
fetch_air_pollution_data_async.cache = {}

def calculate_aqi_from_pm25(pm25: float) -> int:
    """Calculate AQI based on PM2.5 concentration using US EPA breakpoints"""
    
    # US EPA PM2.5 AQI breakpoints (μg/m³)
    breakpoints = [
        (0.0, 12.0, 0, 50),      # Good
        (12.1, 35.4, 51, 100),   # Moderate  
        (35.5, 55.4, 101, 150),  # Unhealthy for Sensitive Groups
        (55.5, 150.4, 151, 200), # Unhealthy
        (150.5, 250.4, 201, 300), # Very Unhealthy
        (250.5, 350.4, 301, 400), # Hazardous
        (350.5, 500.4, 401, 500)  # Hazardous+
    ]
    
    # Handle extreme values
    if pm25 <= 0:
        return 0
    if pm25 > 500.4:
        return 500
    
    # Find the appropriate breakpoint
    for bp_lo, bp_hi, aqi_lo, aqi_hi in breakpoints:
        if bp_lo <= pm25 <= bp_hi:
            # Linear interpolation formula
            aqi = ((aqi_hi - aqi_lo) / (bp_hi - bp_lo)) * (pm25 - bp_lo) + aqi_lo
            return round(aqi)
    
    return 500  # Default to maximum if no breakpoint found

def get_fallback_air_quality_data(lat: float, lng: float) -> Dict[str, float]:
    """Provide fallback air quality data when API fails"""
    # Estimate air quality based on geographic location
    if abs(lat) > 60:  # Arctic/Antarctic - clean air
        openweather_aqi = 1
        pm2_5, pm10 = 5, 10
    elif abs(lat) > 40:  # Temperate - moderate
        openweather_aqi = 2
        pm2_5, pm10 = 15, 25
    else:  # Tropical/subtropical - variable
        openweather_aqi = 3
        pm2_5, pm10 = 25, 40
    
    # Adjust for longitude (rough urban/industrial effects)
    if -100 < lng < -70 or 100 < lng < 140:  # Major industrial regions
        openweather_aqi = min(5, openweather_aqi + 1)
        pm2_5 *= 1.5
        pm10 *= 1.3
    
    # Calculate proper AQI from PM2.5
    calculated_aqi = calculate_aqi_from_pm25(pm2_5)
    
    logger.info(f"Using fallback air quality data for {lat}, {lng}: AQI={calculated_aqi} (PM2.5: {pm2_5})")
    return {
        "aqi": calculated_aqi,
        "openweather_aqi": openweather_aqi,
        "pm2_5": pm2_5,
        "pm10": pm10,
        "no2": 20.0,
        "o3": 80.0,
        "co": 200.0
    }

async def assess_single_location_async(lat: float, lng: float) -> RiskAssessment:
    """Async version of assess_single_location using httpx with air quality integration"""
    try:
        # Fetch data from multiple sources concurrently
        climate_data, air_data = await asyncio.gather(
            fetch_nasa_power_data_async(lat, lng),
            fetch_air_pollution_data_async(lat, lng)
        )
        
        # Calculate risk levels
        risks = calculate_risk_levels(climate_data["rainfall_mm"], climate_data["temperature_c"])
        
        # Calculate air quality risk
        air_quality_risk = calculate_air_quality_risk(
            air_data["aqi"],
            air_data["pm2_5"],
            air_data["pm10"]
        )
        
        # Calculate overall risk score
        overall_risk = calculate_overall_risk_score(
            risks["flood_risk"],
            risks["heat_risk"],
            air_quality_risk
        )
        
        # Generate comprehensive summary
        summary = generate_summary(
            climate_data["rainfall_mm"], 
            climate_data["temperature_c"], 
            risks["flood_risk"], 
            risks["heat_risk"],
            air_quality_risk,
            air_data["aqi"],
            air_data["pm2_5"],
            overall_risk
        )
        
        return RiskAssessment(
            latitude=lat,
            longitude=lng,
            rainfall_mm=climate_data["rainfall_mm"],
            temperature_c=climate_data["temperature_c"],
            flood_risk=risks["flood_risk"],
            heat_risk=risks["heat_risk"],
            air_quality_index=air_data["aqi"],
            air_quality_risk=air_quality_risk,
            pm2_5=air_data["pm2_5"],
            pm10=air_data["pm10"],
            no2=air_data["no2"],
            o3=air_data["o3"],
            co=air_data["co"],
            overall_risk_score=overall_risk,
            summary=summary
        )
    except Exception as e:
        logger.warning(f"Failed to assess coordinate {lat}, {lng}: {e}")
        raise

@app.post("/assess-risk-bulk", response_model=Dict[str, List[RiskAssessment]])
async def assess_risk_bulk(request: BulkCoordinateRequest):
    """Assess environmental risks for multiple coordinates efficiently (frontend format)"""
    try:
        # Limit the number of coordinates to prevent abuse and timeouts
        max_coords = 50  # Reduced from 200 to prevent timeouts
        if len(request.coordinates) > max_coords:
            raise HTTPException(status_code=400, detail=f"Too many coordinates. Maximum {max_coords} coordinates per request.")
        
        logger.info(f"Processing bulk assessment for {len(request.coordinates)} coordinates")
        start_time = time.time()
        
        # Process coordinates with concurrency limit
        # Process coordinates with async semaphore for concurrency control
        semaphore = asyncio.Semaphore(10)  # Increased to 10 concurrent requests for faster processing
        
        async def process_coordinate(coord):
            async with semaphore:
                try:
                    # Add timeout for individual coordinate processing
                    return await asyncio.wait_for(
                        assess_single_location_async(coord.lat, coord.lng),
                        timeout=30.0  # 30 second timeout per coordinate
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout processing coordinate {coord.lat}, {coord.lng}")
                    return None
                except Exception as e:
                    logger.warning(f"Failed to assess coordinate {coord.lat}, {coord.lng}: {e}")
                    return None
        
        # Process all coordinates concurrently with overall timeout
        try:
            tasks = [process_coordinate(coord) for coord in request.coordinates]
            results_raw = await asyncio.wait_for(
                asyncio.gather(*tasks),
                timeout=60.0  # 60 second overall timeout to allow for retries
            )
            
            # Filter out None results (failed assessments)
            results = [result for result in results_raw if result is not None]
            
        except asyncio.TimeoutError:
            logger.error("Bulk assessment timed out after 60 seconds")
            raise HTTPException(status_code=408, detail="Request timed out. The NASA API may be experiencing issues. Please try again in a moment.")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Successfully processed {len(results)} out of {len(request.coordinates)} coordinates in {elapsed_time:.2f} seconds")
        
        if len(results) == 0:
            raise HTTPException(status_code=500, detail="Unable to process any coordinates. Please try again or check if the NASA API is available.")
        
        return {"results": results}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Bulk assessment error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/assess-locations-bulk", response_model=List[RiskAssessment])
async def assess_locations_bulk(request: BulkLocationRequest):
    """Assess environmental risks for multiple locations efficiently (original format)"""
    try:
        # Limit the number of locations to prevent abuse
        if len(request.locations) > 100:
            raise HTTPException(status_code=400, detail="Too many locations. Maximum 100 locations per request.")
        
        results = []
        for location in request.locations:
            try:
                result = assess_single_location(location.latitude, location.longitude)
                results.append(result)
            except Exception as e:
                # Continue with other locations
                continue
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Bulk assessment error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

def assess_single_location(lat: float, lon: float) -> RiskAssessment:
    """Assess risk for a single location (sync helper)"""
    climate_data = fetch_nasa_power_data(lat, lon)
    risks = calculate_risk_levels(climate_data["rainfall_mm"], climate_data["temperature_c"])
    summary = generate_summary(
        climate_data["rainfall_mm"], 
        climate_data["temperature_c"], 
        risks["flood_risk"], 
        risks["heat_risk"]
    )
    
    return RiskAssessment(
        latitude=lat,
        longitude=lon,
        rainfall_mm=climate_data["rainfall_mm"],
        temperature_c=climate_data["temperature_c"],
        flood_risk=risks["flood_risk"],
        heat_risk=risks["heat_risk"],
        summary=summary
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Urban Planner AI"}

@app.get("/test-nasa")
async def test_nasa_api():
    """Test endpoint to verify NASA API connectivity"""
    try:
        # Test with New York City coordinates
        test_data = fetch_nasa_power_data(40.7128, -74.0060)
        return {
            "status": "NASA API accessible",
            "test_location": "New York City",
            "sample_data": test_data
        }
    except Exception as e:
        return {
            "status": "NASA API error",
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)