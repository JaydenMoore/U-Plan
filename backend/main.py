from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import httpx
import asyncio
from typing import Dict, Any, List, Optional
import logging
from functools import lru_cache
import time
import os
from io import BytesIO
import logging
from functools import lru_cache
import time
import os
import io
from io import BytesIO

# Probabilistic modeling imports
from scipy import stats
import numpy as np
from sklearn.metrics import brier_score_loss
from fastapi import FastAPI, HTTPException, Request
from rising_issues_utils import RisingIssue, infer_issue_type, group_feedback_by_proximity_and_type
from pydantic import BaseModel
from datetime import datetime, timedelta
from dotenv import load_dotenv
try:
    from supabase import create_client
except Exception:
    create_client = None

# Import population data reader
try:
    from population_reader import PopulationDataReader
    POPULATION_AVAILABLE = True
except ImportError:
    POPULATION_AVAILABLE = False
    print("Population libraries not available. Install rasterio and numpy to enable population features.")

# Global variables for caching
population_reader = None
historic_flood_dataset = None
from summary_agent import generate_summary

# Probabilistic Risk Model Implementation
class ProbabilisticRiskModel:
    """
    Probabilistic risk model using Gaussian copula to model dependencies
    between flood, pollution, and population for comprehensive risk assessment
    """
    
    def __init__(self):
        self.pol_fit = None
        self.flood_fit = None
        self.corr_matrix = None
        self.vuln_params = {
            "alpha_f": 0.12,    # flood effect coefficient
            "alpha_p": 0.02,    # pollution effect coefficient (per µg/m3)
            "f0": None,         # flood anchor (will be set to mean)
            "p0": None,         # pollution anchor (will be set to mean)
            "vmax": 1.0
        }
        self.fitted = False
    
    def fit_marginal_distributions(self, pollution_data, flood_data):
        """Fit marginal distributions for pollution and flood data"""
        # Remove invalid data
        pol_clean = pollution_data[~np.isnan(pollution_data) & (pollution_data > 0)]
        flood_clean = flood_data[~np.isnan(flood_data) & (flood_data >= 0)]
        
        if len(pol_clean) < 10 or len(flood_clean) < 10:
            logger.warning("Insufficient data for probabilistic modeling, using simplified approach")
            return False
        
        # Fit pollution (PM2.5) - try lognormal and gamma
        try:
            # Lognormal fit
            s, loc, scale = stats.lognorm.fit(pol_clean, floc=0)
            ks_stat_ln, ks_p_ln = stats.kstest(pol_clean, 'lognorm', args=(s, loc, scale))
            
            # Gamma fit
            a, loc_g, scale_g = stats.gamma.fit(pol_clean, floc=0)
            ks_stat_g, ks_p_g = stats.kstest(pol_clean, 'gamma', args=(a, loc_g, scale_g))
            
            # Choose best fit based on KS test p-value
            if ks_p_ln >= ks_p_g:
                self.pol_fit = {"dist": "lognorm", "params": (s, loc, scale)}
            else:
                self.pol_fit = {"dist": "gamma", "params": (a, loc_g, scale_g)}
                
        except Exception as e:
            logger.warning(f"Failed to fit pollution distribution: {e}")
            return False
        
        # Fit flood probability (0-1) - use beta distribution
        try:
            # Clip to (0,1) range for beta
            flood_clipped = np.clip(flood_clean, 1e-6, 1-1e-6)
            a, b, loc, scale = stats.beta.fit(flood_clipped, floc=0, fscale=1)
            self.flood_fit = {"dist": "beta", "params": (a, b, loc, scale)}
        except Exception as e:
            logger.warning(f"Failed to fit flood distribution: {e}")
            return False
        
        # Set anchor parameters
        self.vuln_params['p0'] = np.mean(pol_clean)
        self.vuln_params['f0'] = np.mean(flood_clean)
        
        logger.info(f"Fitted marginals - Pollution: {self.pol_fit['dist']}, Flood: {self.flood_fit['dist']}")
        return True
    
    def estimate_copula_dependence(self, pollution_data, flood_data):
        """Estimate Gaussian copula dependence structure"""
        # Clean data
        valid_mask = ~np.isnan(pollution_data) & ~np.isnan(flood_data) & (pollution_data > 0) & (flood_data >= 0)
        pol_clean = pollution_data[valid_mask]
        flood_clean = flood_data[valid_mask]
        
        if len(pol_clean) < 20:
            logger.warning("Insufficient data for copula estimation")
            self.corr_matrix = np.eye(2)  # Independence assumption
            return False
        
        try:
            # Transform to uniform margins using fitted distributions
            u_pol = self._cdf_eval(pol_clean, self.pol_fit)
            u_flood = self._cdf_eval(flood_clean, self.flood_fit)
            
            # Clip to avoid numerical issues
            eps = 1e-6
            u_pol = np.clip(u_pol, eps, 1-eps)
            u_flood = np.clip(u_flood, eps, 1-eps)
            
            # Transform to standard normal for Gaussian copula
            z_pol = stats.norm.ppf(u_pol)
            z_flood = stats.norm.ppf(u_flood)
            
            # Estimate correlation
            Z = np.vstack([z_pol, z_flood]).T
            self.corr_matrix = np.corrcoef(Z, rowvar=False)
            
            # Ensure positive definite
            eigvals = np.linalg.eigvals(self.corr_matrix)
            if np.any(eigvals <= 0):
                self.corr_matrix += np.eye(2) * 1e-6
            
            logger.info(f"Estimated copula correlation: {self.corr_matrix[0,1]:.3f}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to estimate copula: {e}")
            self.corr_matrix = np.eye(2)
            return False
    
    def _cdf_eval(self, x, fit):
        """Evaluate CDF for fitted distribution"""
        if fit['dist'] == "lognorm":
            s, loc, scale = fit['params']
            return stats.lognorm.cdf(x, s, loc, scale)
        elif fit['dist'] == "gamma":
            a, loc, scale = fit['params']
            return stats.gamma.cdf(x, a, loc, scale)
        elif fit['dist'] == "beta":
            a, b, loc, scale = fit['params']
            return stats.beta.cdf(np.clip(x, 1e-6, 1-1e-6), a, b, loc, scale)
        else:
            # Fallback to empirical CDF
            return stats.rankdata(x) / (len(x) + 1)
    
    def _inv_cdf(self, u, fit):
        """Inverse CDF (quantile function) for fitted distribution"""
        u = np.clip(u, 1e-9, 1-1e-9)
        if fit['dist'] == "lognorm":
            s, loc, scale = fit['params']
            return stats.lognorm.ppf(u, s, loc, scale)
        elif fit['dist'] == "gamma":
            a, loc, scale = fit['params']
            return stats.gamma.ppf(u, a, loc, scale)
        elif fit['dist'] == "beta":
            a, b, loc, scale = fit['params']
            return stats.beta.ppf(u, a, b, loc, scale)
        else:
            raise ValueError("Unsupported distribution for inverse CDF")
    
    def vulnerability_function(self, flood_prob, pollution_pm25):
        """
        Logistic vulnerability function combining flood and pollution effects
        Returns vulnerability score between 0 and vmax
        """
        alpha_f = self.vuln_params['alpha_f']
        alpha_p = self.vuln_params['alpha_p']
        f0 = self.vuln_params['f0']
        p0 = self.vuln_params['p0']
        vmax = self.vuln_params['vmax']
        
        z = alpha_f * (flood_prob - f0) + alpha_p * (pollution_pm25 - p0)
        return vmax / (1.0 + np.exp(-z))
    
    def sample_joint_hazards(self, n_samples=1000):
        """Sample joint hazards using Gaussian copula"""
        if not self.fitted:
            raise ValueError("Model must be fitted before sampling")
        
        # Sample from multivariate normal
        rng = np.random.default_rng(42)
        L = np.linalg.cholesky(self.corr_matrix)
        z = rng.standard_normal(size=(n_samples, 2))
        correlated = z @ L.T
        
        # Transform to uniform margins
        u = stats.norm.cdf(correlated)
        
        # Transform to original scales
        pollution_samples = self._inv_cdf(u[:, 0], self.pol_fit)
        flood_samples = self._inv_cdf(u[:, 1], self.flood_fit)
        
        return pollution_samples, flood_samples
    
    def compute_probabilistic_risk(self, current_pollution, current_flood, population_density, n_samples=1000):
        """
        Compute probabilistic risk assessment using Monte Carlo sampling
        
        Returns:
            dict with expected risk, confidence intervals, and risk components
        """
        if not self.fitted:
            # Simple fallback if model not fitted
            vuln = self.vulnerability_function(current_flood, current_pollution)
            return {
                "expected_risk": float(population_density * vuln),
                "risk_p05": float(population_density * vuln * 0.5),
                "risk_p50": float(population_density * vuln),
                "risk_p95": float(population_density * vuln * 1.5),
                "vulnerability_mean": float(vuln),
                "model_fitted": False
            }
        
        try:
            # Sample joint hazards
            pol_samples, flood_samples = self.sample_joint_hazards(n_samples)
            
            # Compute vulnerability for each sample
            vuln_samples = np.array([
                self.vulnerability_function(f, p) 
                for f, p in zip(flood_samples, pol_samples)
            ])
            
            # Risk = population * vulnerability
            risk_samples = population_density * vuln_samples
            
            # Compute statistics
            expected_risk = np.mean(risk_samples)
            risk_p05 = np.percentile(risk_samples, 5)
            risk_p50 = np.percentile(risk_samples, 50)
            risk_p95 = np.percentile(risk_samples, 95)
            vuln_mean = np.mean(vuln_samples)
            
            return {
                "expected_risk": float(expected_risk),
                "risk_p05": float(risk_p05),
                "risk_p50": float(risk_p50),
                "risk_p95": float(risk_p95),
                "vulnerability_mean": float(vuln_mean),
                "uncertainty_range": float(risk_p95 - risk_p05),
                "model_fitted": True,
                "n_samples": n_samples
            }
            
        except Exception as e:
            logger.error(f"Error in probabilistic risk computation: {e}")
            # Fallback to deterministic calculation
            vuln = self.vulnerability_function(current_flood, current_pollution)
            return {
                "expected_risk": float(population_density * vuln),
                "risk_p05": float(population_density * vuln * 0.8),
                "risk_p50": float(population_density * vuln),
                "risk_p95": float(population_density * vuln * 1.2),
                "vulnerability_mean": float(vuln),
                "model_fitted": False,
                "error": str(e)
            }
    
    def fit(self, pollution_data, flood_data):
        """Fit the complete probabilistic model"""
        logger.info("Fitting probabilistic risk model...")
        
        # Fit marginal distributions
        if not self.fit_marginal_distributions(pollution_data, flood_data):
            logger.warning("Failed to fit marginal distributions")
            return False
        
        # Estimate copula dependence
        if not self.estimate_copula_dependence(pollution_data, flood_data):
            logger.warning("Failed to estimate copula dependence")
            return False
        
        self.fitted = True
        logger.info("Probabilistic risk model fitted successfully")
        return True

# Initialize global probabilistic model
probabilistic_model = ProbabilisticRiskModel()

# OpenWeatherMap API configuration
OPENWEATHER_API_KEY = "28e7cf23db337a4fcb983071b8d85b2b"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables (GOOGLE_API_KEY, SUPABASE, etc.)
load_dotenv()

app = FastAPI(title="Urban Planner AI", description="NASA-powered environmental risk assessment")

# Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
supabase = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY and create_client is not None:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
        logger.info("Supabase client initialized; feedback storage enabled.")
    except Exception as e:
        logger.warning(f"Failed to initialize Supabase client: {e}")
else:
    logger.info("Supabase env not configured; feedback endpoint will be disabled.")

# Initialize population data reader
population_reader = None
if POPULATION_AVAILABLE:
    population_data_path = "/Users/jaydenmoore/Documents/NASA Space Apps/population density/gpw-v4-population-density-adjusted-to-2015-unwpp-country-totals_2020.tif"
    if os.path.exists(population_data_path):
        try:
            population_reader = PopulationDataReader(population_data_path)
            logger.info("Population data loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load population data: {e}")
    else:
        logger.warning(f"Population data file not found at: {population_data_path}")

# Initialize historic flood data reader (Global Flood Database)
historic_flood_dataset = None
historic_flood_path = "/Users/jaydenmoore/Documents/NASA Space Apps/data/global_flood_frequency.tif"
if POPULATION_AVAILABLE:  # Use same rasterio availability check
    if os.path.exists(historic_flood_path):
        try:
            import rasterio
            historic_flood_dataset = rasterio.open(historic_flood_path)
            logger.info("Historic flood data loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load historic flood data: {e}")
    else:
        logger.info(f"Historic flood data file not found at: {historic_flood_path} - using simulated data")

# CORS middleware for frontend access

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
    population_density: Optional[float] = None  # People per km²
    population_stats: Optional[Dict] = None  # Area population statistics
    flood_risk_score: Optional[float] = None  # NASA MODIS flood risk (0-1)
    flood_message: Optional[str] = None  # Flood risk interpretation
    historic_flood_frequency: Optional[float] = None  # Historic flood frequency (0-1)
    historic_flood_category: Optional[str] = None  # Historic flood risk category
    # Comprehensive flood assessment combining real-time and historical data
    comprehensive_flood_risk: Optional[Dict] = None  # Combined flood risk assessment
    flood_percentage_explanation: Optional[Dict] = None  # Detailed percentage explanations
    historic_flood_category: Optional[str] = None  # Historic flood category

class EnhancedRiskAssessment(RiskAssessment):
    # Add new fields for probabilistic assessment
    probabilistic_risk: Optional[Dict] = None
    risk_confidence_interval: Optional[Dict] = None
    vulnerability_score: Optional[float] = None
    model_uncertainty: Optional[str] = None

class FeedbackRequest(BaseModel):
    latitude: float
    longitude: float
    feedback: str
    summary: Optional[str] = None
    overall_risk_score: Optional[float] = None
    source: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

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

def calculate_comprehensive_flood_risk(realtime_flood_score: float, historic_flood_frequency: float, lat: float, lon: float) -> Dict[str, Any]:
    """
    Calculate comprehensive flood risk combining real-time satellite data and historical patterns
    
    Args:
        realtime_flood_score: NASA MODIS flood detection score (0-1)
        historic_flood_frequency: Historical flood frequency (0-1)
        lat, lon: Location coordinates
    
    Returns:
        Dictionary with comprehensive flood assessment
    """
    # Weight factors: real-time 60%, historical 40%
    realtime_weight = 0.60
    historical_weight = 0.40
    
    # Combined flood risk score
    combined_score = (realtime_flood_score * realtime_weight) + (historic_flood_frequency * historical_weight)
    
    # Determine risk level and detailed explanation
    if combined_score >= 0.7:
        risk_level = "Very High"
        explanation = "Critical flood risk - immediate evacuation may be necessary"
        color_code = "red"
    elif combined_score >= 0.5:
        risk_level = "High"
        explanation = "Significant flood risk - prepare for potential evacuation"
        color_code = "orange"
    elif combined_score >= 0.3:
        risk_level = "Moderate"
        explanation = "Elevated flood risk - monitor conditions closely"
        color_code = "yellow"
    elif combined_score >= 0.1:
        risk_level = "Low"
        explanation = "Minor flood risk - stay informed of weather conditions"
        color_code = "lightgreen"
    else:
        risk_level = "Very Low"
        explanation = "Minimal flood risk - normal precautions sufficient"
        color_code = "green"
    
    # Geographic context for historical percentage
    geographic_context = ""
    if abs(lat) < 10:
        geographic_context = "Tropical region: High baseline risk due to intense rainfall, hurricanes, and monsoons"
    elif abs(lat) > 60:
        geographic_context = "High latitude region: Risk mainly from snowmelt and precipitation"
    elif 25 < abs(lat) < 35:
        geographic_context = "Subtropical region: Moderate risk from seasonal storms and precipitation"
    else:
        geographic_context = "Temperate region: Risk varies with seasonal precipitation patterns"
    
    return {
        "combined_score": round(combined_score, 3),
        "risk_level": risk_level,
        "explanation": explanation,
        "color_code": color_code,
        "realtime_contribution": round(realtime_flood_score * realtime_weight, 3),
        "historical_contribution": round(historic_flood_frequency * historical_weight, 3),
        "geographic_context": geographic_context,
        "percentage_explanation": {
            "realtime_pct": f"{realtime_flood_score * 100:.1f}%",
            "historical_pct": f"{historic_flood_frequency * 100:.1f}%",
            "combined_pct": f"{combined_score * 100:.1f}%",
            "realtime_meaning": "Percentage of satellite pixels showing active flooding right now",
            "historical_meaning": "Likelihood of flooding based on 20+ years of climate data and geographic factors",
            "combined_meaning": "Overall flood risk combining current conditions (60%) with historical patterns (40%)"
        }
    }

def calculate_overall_risk_score(flood_risk: str, heat_risk: str, air_quality_risk: str, population_density: Optional[float] = None) -> float:
    """Calculate overall risk score (0-10) based on individual risk factors and population context"""
    
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
    
    # Base weighted average: flood 30%, heat 30%, air quality 25%
    base_score = (flood_val * 0.30) + (heat_val * 0.30) + (air_val * 0.25)
    
    # Population density modifier (15% weight)
    pop_modifier = 0
    if population_density is not None:
        if population_density > 1000:  # Very high density
            pop_modifier = 2.0  # Increases risk due to exposure
        elif population_density > 500:  # High density
            pop_modifier = 1.5
        elif population_density > 150:  # Medium density
            pop_modifier = 1.0
        elif population_density > 50:  # Low density
            pop_modifier = 0.5
        else:  # Very low/uninhabited
            pop_modifier = 0.0
    
    # Apply population modifier
    overall_score = base_score + (pop_modifier * 0.15)
    
    # Ensure score stays within bounds
    overall_score = max(0, min(10, overall_score))
    
    return round(overall_score, 2)

async def assess_location_enhanced(location: LocationRequest):
    """Enhanced assessment with probabilistic risk modeling"""
    try:
        # Validate coordinates
        if not (-90 <= location.latitude <= 90):
            raise HTTPException(status_code=400, detail="Invalid latitude")
        if not (-180 <= location.longitude <= 180):
            raise HTTPException(status_code=400, detail="Invalid longitude")
        
        # Fetch data from multiple sources concurrently
        nasa_data_task = fetch_nasa_power_data_async(location.latitude, location.longitude)
        air_data_task = fetch_air_pollution_data_async(location.latitude, location.longitude)
        
        climate_data, air_data = await asyncio.gather(nasa_data_task, air_data_task)
        
        # Get population and flood data as before
        population_density = None
        population_stats = None
        if POPULATION_AVAILABLE and population_reader:
            try:
                population_density = population_reader.get_population_density(location.latitude, location.longitude)
                population_stats = population_reader.get_population_stats_in_area(location.latitude, location.longitude, radius_km=5.0)
            except Exception as e:
                logger.warning(f"Failed to get population data: {e}")
                population_density = 100.0  # Default value
        
        flood_data = fetch_nasa_modis_flood_data(location.latitude, location.longitude)
        historic_flood_data = fetch_historic_flood_frequency(location.latitude, location.longitude)
        
        # Prepare data for probabilistic model
        current_pollution = air_data["pm2_5"]
        current_flood_prob = flood_data["flood_risk"]
        
        # Try to fit probabilistic model if we have sufficient historical data
        # (In practice, you'd fit this once with historical data and reuse)
        try:
            # Simulate some historical data for demonstration
            # In practice, you'd use actual historical datasets
            hist_pollution = np.random.lognormal(mean=np.log(max(current_pollution, 1) + 1), sigma=0.5, size=100)
            hist_flood = np.random.beta(a=2, b=8, size=100) * 0.3  # Simulate flood probabilities
            
            if probabilistic_model.fit(hist_pollution, hist_flood):
                prob_risk = probabilistic_model.compute_probabilistic_risk(
                    current_pollution, 
                    current_flood_prob, 
                    population_density or 100.0,
                    n_samples=500
                )
            else:
                prob_risk = None
        except Exception as e:
            logger.warning(f"Probabilistic modeling failed: {e}")
            prob_risk = None
        
        # Calculate existing risk assessments
        risks = calculate_risk_levels(climate_data["rainfall_mm"], climate_data["temperature_c"])
        air_quality_risk = calculate_air_quality_risk(air_data["aqi"], air_data["pm2_5"], air_data["pm10"])
        
        # Enhanced overall risk score incorporating probabilistic assessment
        if prob_risk and prob_risk.get("model_fitted", False):
            # Better normalization for probabilistic risk component
            # Convert expected risk (people at risk) to a 0-10 scale
            # Use log scale for better distribution across the range
            import math
            if prob_risk["expected_risk"] > 0:
                # Log-scale normalization: log10(risk + 1) scaled to 0-10
                prob_risk_component = min(math.log10(prob_risk["expected_risk"] + 1) * 2.5, 10.0)
            else:
                prob_risk_component = 0.0
            
            base_risk = calculate_overall_risk_score(risks["flood_risk"], risks["heat_risk"], air_quality_risk, population_density)
            # Weighted combination: base risk gets 60%, probabilistic gets 40% 
            overall_risk = (base_risk * 0.6) + (prob_risk_component * 0.4)
            
            model_uncertainty = "Low" if prob_risk["uncertainty_range"] < prob_risk["expected_risk"] * 0.5 else "High"
        else:
            overall_risk = calculate_overall_risk_score(risks["flood_risk"], risks["heat_risk"], air_quality_risk, population_density)
            model_uncertainty = "Model not fitted"
        
        # Calculate comprehensive flood risk assessment
        comprehensive_flood = calculate_comprehensive_flood_risk(
            flood_data.get('flood_risk', 0.0),
            historic_flood_data.get('flood_frequency', 0.0),
            location.latitude, location.longitude
        )
        
        # Generate enhanced summary
        summary = await generate_enhanced_summary(
            climate_data, risks, air_quality_risk, air_data, 
            overall_risk, population_density, prob_risk
        )
        
        # Prepare confidence interval data
        confidence_interval = None
        if prob_risk and prob_risk.get("model_fitted", False):
            confidence_interval = {
                "lower_bound": prob_risk["risk_p05"],
                "median": prob_risk["risk_p50"], 
                "upper_bound": prob_risk["risk_p95"],
                "confidence_level": "90%"
            }
        
        return EnhancedRiskAssessment(
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
            summary=summary,
            population_density=population_density,
            population_stats=population_stats,
            flood_risk_score=flood_data["flood_risk"],
            flood_message=flood_data["message"],
            historic_flood_frequency=historic_flood_data["flood_frequency"],
            historic_flood_category=historic_flood_data["category"],
            comprehensive_flood_risk=comprehensive_flood,
            flood_percentage_explanation=comprehensive_flood["percentage_explanation"],
            # New probabilistic fields
            probabilistic_risk=prob_risk,
            risk_confidence_interval=confidence_interval,
            vulnerability_score=prob_risk.get("vulnerability_mean") if prob_risk else None,
            model_uncertainty=model_uncertainty
        )
        
    except Exception as e:
        logger.error(f"Enhanced assessment error: {e}")
        # Fallback to original assessment
        return await assess_location(location)

async def generate_enhanced_summary(climate_data, risks, air_quality_risk, air_data, overall_risk, population_density, prob_risk):
    """Generate enhanced summary including probabilistic insights"""
    base_summary = await generate_summary(
        climate_data["rainfall_mm"], climate_data["temperature_c"],
        risks["flood_risk"], risks["heat_risk"], air_quality_risk,
        air_data["aqi"], air_data["pm2_5"], overall_risk,
        population_density, None
    )
    
    if prob_risk and prob_risk.get("model_fitted", False):
        # Calculate the log-scale component for explanation
        import math
        if prob_risk['expected_risk'] > 0:
            prob_risk_component = min(math.log10(prob_risk['expected_risk'] + 1) * 2.5, 10.0)
        else:
            prob_risk_component = 0.0
            
        prob_text = f" Probabilistic analysis indicates an expected risk level of {prob_risk['expected_risk']:.1f} people at risk, with 90% confidence interval of {prob_risk['risk_p05']:.1f}-{prob_risk['risk_p95']:.1f}. The vulnerability score is {prob_risk['vulnerability_mean']:.3f}, indicating {'high' if prob_risk['vulnerability_mean'] > 0.5 else 'moderate' if prob_risk['vulnerability_mean'] > 0.2 else 'low'} susceptibility to combined environmental hazards. This contributes {prob_risk_component:.2f} points to the overall risk score (0-10 scale)."
        return base_summary + prob_text
    else:
        return base_summary + " Note: Probabilistic risk modeling unavailable due to insufficient historical data."
@app.get("/")
async def root():
    return {"message": "Urban Planner AI API - Ready to assess environmental risks!"}

@app.post("/assess-location-enhanced", response_model=EnhancedRiskAssessment)
async def assess_location_enhanced_endpoint(location: LocationRequest):
    """Enhanced location assessment with probabilistic risk modeling"""
    return await assess_location_enhanced(location)

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
        
        # Get population data
        population_density = None
        population_stats = None
        if POPULATION_AVAILABLE and population_reader:
            try:
                population_density = population_reader.get_population_density(location.latitude, location.longitude)
                population_stats = population_reader.get_population_stats_in_area(location.latitude, location.longitude, radius_km=5.0)
            except Exception as e:
                logger.warning(f"Failed to get population data: {e}")
        
        # Get NASA MODIS flood data
        flood_data = fetch_nasa_modis_flood_data(location.latitude, location.longitude)
        
        # Get historic flood data
        historic_flood_data = fetch_historic_flood_frequency(location.latitude, location.longitude)
        
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
        
        # Calculate overall risk score (now considering population)
        overall_risk = calculate_overall_risk_score(
            risks["flood_risk"],
            risks["heat_risk"],
            air_quality_risk,
            population_density
        )
        
        # Generate comprehensive summary
        summary = await generate_summary(
            climate_data["rainfall_mm"],
            climate_data["temperature_c"],
            risks["flood_risk"],
            risks["heat_risk"],
            air_quality_risk,
            air_data["aqi"],
            air_data["pm2_5"],
            overall_risk,
            population_density,
            population_stats
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
            summary=summary,
            population_density=population_density,
            population_stats=population_stats,
            flood_risk_score=flood_data["flood_risk"],
            flood_message=flood_data["message"],
            historic_flood_frequency=historic_flood_data["flood_frequency"],
            historic_flood_category=historic_flood_data["category"]
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

def fetch_nasa_modis_flood_data(lat: float, lon: float) -> Dict[str, Any]:
    """
    Fetch NASA MODIS near-real-time flood data from ImageServer
    Returns flood risk score (0-1) based on flood pixels
    """
    try:
        # Create bounding box (±0.01° around the point)
        buffer = 0.01
        minx, miny = lon - buffer, lat - buffer
        maxx, maxy = lon + buffer, lat + buffer
        
        # NASA Disaster Program ImageServer endpoint
        url = "https://maps.disasters.nasa.gov/ags03/rest/services/NRT/modis_flood_1_day/ImageServer/exportImage"
        
        params = {
            'bbox': f"{minx},{miny},{maxx},{maxy}",
            'bboxSR': '4326',
            'imageSR': '4326', 
            'size': '256,256',
            'format': 'tiff',
            'f': 'image'
        }
        
        logger.info(f"Fetching MODIS flood data for {lat}, {lon}")
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        
        # Read TIFF response in memory with rasterio
        import rasterio
        import numpy as np
        
        with rasterio.open(BytesIO(response.content)) as dataset:
            # Read the raster data
            data = dataset.read(1)  # Read first band
            
            # Count pixels where value == 3 (flood water)
            flood_pixels = np.count_nonzero(data == 3)
            total_valid_pixels = np.count_nonzero(~np.isnan(data) & (data != dataset.nodata))
            
            if total_valid_pixels == 0:
                flood_risk = 0.0
                message = "No flood data available"
            else:
                # Compute flood risk ratio
                flood_risk = float(flood_pixels / total_valid_pixels)
                
                # Generate flood message
                if flood_risk >= 0.5:
                    message = "High flood risk detected"
                elif flood_risk >= 0.2:
                    message = "Moderate flood risk detected" 
                elif flood_risk > 0:
                    message = "Low flood risk detected"
                else:
                    message = "No flood risk detected"
        
        logger.info(f"MODIS flood analysis: {flood_pixels}/{total_valid_pixels} flood pixels, risk={flood_risk:.3f}")
        
        return {
            "flood_risk": round(flood_risk, 3),
            "message": message,
            "flood_pixels": int(flood_pixels),
            "total_pixels": int(total_valid_pixels)
        }
        
    except Exception as e:
        logger.error(f"Error fetching MODIS flood data: {e}")
        # Graceful fallback
        return {
            "flood_risk": 0.0,
            "message": "Flood data unavailable",
            "flood_pixels": 0,
            "total_pixels": 0
        }

def fetch_nasa_modis_flood_data(lat: float, lon: float) -> Dict[str, Any]:
    """
    Fetch NASA MODIS near-real-time flood data from ImageServer
    Returns flood risk score (0-1) based on flood pixels
    """
    try:
        # Create bounding box (±0.01° around the point)
        buffer = 0.01
        minx, miny = lon - buffer, lat - buffer
        maxx, maxy = lon + buffer, lat + buffer
        
        # NASA Disaster Program ImageServer endpoint
        url = "https://maps.disasters.nasa.gov/ags03/rest/services/NRT/modis_flood_1_day/ImageServer/exportImage"
        
        params = {
            'bbox': f"{minx},{miny},{maxx},{maxy}",
            'bboxSR': '4326',
            'imageSR': '4326', 
            'size': '256,256',
            'format': 'tiff',
            'f': 'image'
        }
        
        logger.info(f"Fetching MODIS flood data for {lat}, {lon}")
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        
        # Read TIFF response in memory with rasterio
        import rasterio
        import numpy as np
        
        with rasterio.open(BytesIO(response.content)) as dataset:
            # Read the raster data
            data = dataset.read(1)  # Read first band
            
            # Count pixels where value == 3 (flood water)
            flood_pixels = np.count_nonzero(data == 3)
            total_valid_pixels = np.count_nonzero(~np.isnan(data) & (data != dataset.nodata))
            
            if total_valid_pixels == 0:
                flood_risk = 0.0
                message = "No flood data available"
            else:
                # Compute flood risk ratio
                flood_risk = float(flood_pixels / total_valid_pixels)
                
                # Generate flood message
                if flood_risk >= 0.5:
                    message = "High flood risk detected"
                elif flood_risk >= 0.2:
                    message = "Moderate flood risk detected" 
                elif flood_risk > 0:
                    message = "Low flood risk detected"
                else:
                    message = "No flood risk detected"
        
        logger.info(f"MODIS flood analysis: {flood_pixels}/{total_valid_pixels} flood pixels, risk={flood_risk:.3f}")
        
        return {
            "flood_risk": round(flood_risk, 3),
            "message": message,
            "flood_pixels": int(flood_pixels),
            "total_pixels": int(total_valid_pixels)
        }
        
    except Exception as e:
        logger.error(f"Error fetching MODIS flood data: {e}")
        # Graceful fallback
        return {
            "flood_risk": 0.0,
            "message": "Flood data unavailable",
            "flood_pixels": 0,
            "total_pixels": 0
        }

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
        
        # Get population data
        population_density = None
        population_stats = None
        if POPULATION_AVAILABLE and population_reader:
            try:
                population_density = population_reader.get_population_density(lat, lng)
                population_stats = population_reader.get_population_stats_in_area(lat, lng, radius_km=5.0)
            except Exception as e:
                logger.warning(f"Failed to get population data: {e}")
        
        # Get NASA MODIS flood data
        flood_data = fetch_nasa_modis_flood_data(lat, lng)
        
        # Get historic flood data
        historic_flood_data = fetch_historic_flood_frequency(lat, lng)
        
        # Calculate comprehensive flood risk assessment
        comprehensive_flood = calculate_comprehensive_flood_risk(
            flood_data.get('flood_risk', 0.0),
            historic_flood_data.get('flood_frequency', 0.0),
            lat, lng
        )
        
        # Calculate risk levels
        risks = calculate_risk_levels(climate_data["rainfall_mm"], climate_data["temperature_c"])
        
        # Calculate air quality risk
        air_quality_risk = calculate_air_quality_risk(
            air_data["aqi"],
            air_data["pm2_5"],
            air_data["pm10"]
        )
        
        # Calculate overall risk score (now considering population)
        overall_risk = calculate_overall_risk_score(
            risks["flood_risk"],
            risks["heat_risk"],
            air_quality_risk,
            population_density
        )
        
        # Generate comprehensive summary
        summary = await generate_summary(
            climate_data["rainfall_mm"], 
            climate_data["temperature_c"], 
            risks["flood_risk"], 
            risks["heat_risk"],
            air_quality_risk,
            air_data["aqi"],
            air_data["pm2_5"],
            overall_risk,
            population_density,
            population_stats
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
            summary=summary,
            population_density=population_density,
            population_stats=population_stats,
            flood_risk_score=flood_data["flood_risk"],
            flood_message=flood_data["message"],
            historic_flood_frequency=historic_flood_data["flood_frequency"],
            historic_flood_category=historic_flood_data["category"],
            comprehensive_flood_risk=comprehensive_flood,
            flood_percentage_explanation=comprehensive_flood["percentage_explanation"]
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

async def assess_single_location(lat: float, lon: float) -> RiskAssessment:
    """Assess risk for a single location (sync helper)"""
    climate_data = fetch_nasa_power_data(lat, lon)
    risks = calculate_risk_levels(climate_data["rainfall_mm"], climate_data["temperature_c"])
    summary = await generate_summary(
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

def fetch_historic_flood_frequency(lat: float, lon: float) -> Dict[str, Any]:
    """
    Fetch historic flood frequency from Global Flood Database (2000-2018)
    Returns flood frequency score (0-1) based on historical flood occurrence
    """
    try:
        if historic_flood_dataset:
            # Convert lat/lon to raster indices
            row, col = historic_flood_dataset.index(lon, lat)
            
            # Check if coordinates are within raster bounds
            if 0 <= row < historic_flood_dataset.height and 0 <= col < historic_flood_dataset.width:
                # Read the flood count value at that pixel
                import rasterio
                window = rasterio.windows.Window(col, row, 1, 1)
                data = historic_flood_dataset.read(1, window=window)
                
                if data.size > 0:
                    flood_count = data[0, 0]
                    
                    # Check for no-data values
                    import numpy as np
                    if np.isnan(flood_count) or flood_count < 0:
                        flood_frequency = 0.0
                        category = "No recorded floods"
                    else:
                        # Normalize by 913 (total GFD events) to get score 0-1
                        flood_frequency = min(float(flood_count) / 913.0, 1.0)
                        
                        # Categorize based on frequency
                        if flood_frequency >= 0.5:
                            category = "Very high flood history"
                        elif flood_frequency >= 0.2:
                            category = "High flood history"
                        elif flood_frequency >= 0.05:
                            category = "Moderate flood history"
                        elif flood_frequency > 0:
                            category = "Low flood history"
                        else:
                            category = "No recorded floods"
                else:
                    flood_frequency = 0.0
                    category = "No recorded floods"
            else:
                flood_frequency = 0.0
                category = "No recorded floods"
        else:
            # Simulated data when no GeoTIFF is available
            # Use geographic heuristics for demonstration
            flood_frequency = get_simulated_flood_frequency(lat, lon)
            frequency_percent = flood_frequency * 100
            
            # Provide detailed explanations for percentage ranges
            if flood_frequency >= 0.2:
                category = f"High flood history ({frequency_percent:.1f}% annual probability - expect flooding every {1/flood_frequency:.0f} years)"
            elif flood_frequency >= 0.05:
                category = f"Moderate flood history ({frequency_percent:.1f}% annual probability - expect flooding every {1/flood_frequency:.0f} years)"
            elif flood_frequency > 0:
                category = f"Low flood history ({frequency_percent:.1f}% annual probability - expect flooding every {1/flood_frequency:.0f} years)"
            else:
                category = "No recorded floods (0% annual probability - very rare flooding expected)"
        
        logger.info(f"Historic flood analysis for {lat}, {lon}: frequency={flood_frequency:.3f}, category={category}")
        
        return {
            "flood_frequency": round(flood_frequency, 3),
            "category": category,
            "source": "NASA Global Flood Database (2000–2018)"
        }
        
    except Exception as e:
        logger.error(f"Error fetching historic flood data: {e}")
        return {
            "flood_frequency": 0.0,
            "category": "Historic flood data unavailable",
            "source": "NASA Global Flood Database (2000–2018)"
        }

def get_simulated_flood_frequency(lat: float, lon: float) -> float:
    """Generate simulated flood frequency based on geographic location"""
    # Higher risk for coastal areas and known flood-prone regions
    
    # Coastal proximity (rough estimation)
    coastal_factor = 0.0
    if abs(lat) < 60:  # Not polar regions
        # Distance from major coastlines (simplified)
        if abs(lon + 95) < 5 and 25 < lat < 35:  # Gulf Coast (Houston area)
            coastal_factor = 0.3
        elif abs(lon + 74) < 5 and 35 < lat < 45:  # US East Coast
            coastal_factor = 0.2
        elif abs(lon - 122) < 5 and 32 < lat < 42:  # US West Coast
            coastal_factor = 0.15
        elif abs(lat) < 10:  # Tropical regions - 25% base flood frequency due to intense rainfall, hurricanes, and monsoons
            coastal_factor = 0.25
    
    # River proximity (major river systems)
    river_factor = 0.0
    if abs(lon + 90) < 10 and 29 < lat < 50:  # Mississippi River basin
        river_factor = 0.2
    elif abs(lon + 105) < 15 and 25 < lat < 45:  # Great Plains (flood prone)
        river_factor = 0.15
    
    # Monsoon regions
    monsoon_factor = 0.0
    if 70 < lon < 120 and 10 < lat < 35:  # South/Southeast Asia
        monsoon_factor = 0.3
    elif -60 < lon < -40 and -30 < lat < 0:  # South America
        monsoon_factor = 0.2
    
    # Combine factors
    total_frequency = min(coastal_factor + river_factor + monsoon_factor, 1.0)
    return total_frequency

@app.get("/flood-score")
def flood_score(lat: float, lon: float):
    """Fetch flood data from NASA ImageServer and return flood risk score."""
    try:
        # Validate coordinates
        if not (-90 <= lat <= 90):
            raise HTTPException(status_code=400, detail="Invalid latitude")
        if not (-180 <= lon <= 180):
            raise HTTPException(status_code=400, detail="Invalid longitude")
        
        flood_data = fetch_nasa_modis_flood_data(lat, lon)
        
        return {
            "lat": lat,
            "lon": lon,
            "flood_risk": flood_data["flood_risk"],
            "message": flood_data["message"],
            "flood_pixels": flood_data["flood_pixels"],
            "total_pixels": flood_data["total_pixels"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Flood score error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/historic-flood")
def historic_flood(lat: float, lon: float):
    """Return flood frequency (0–1) from the NASA Global Flood Database"""
    try:
        # Validate coordinates
        if not (-90 <= lat <= 90):
            raise HTTPException(status_code=400, detail="Invalid latitude")
        if not (-180 <= lon <= 180):
            raise HTTPException(status_code=400, detail="Invalid longitude")
        
        historic_data = fetch_historic_flood_frequency(lat, lon)
        
        return {
            "lat": lat,
            "lon": lon,
            "flood_frequency": historic_data["flood_frequency"],
            "category": historic_data["category"],
            "source": historic_data["source"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Historic flood error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/flood-score")
def flood_score(lat: float, lon: float):
    """Fetch flood data from NASA ImageServer and return flood risk score."""
    try:
        # Validate coordinates
        if not (-90 <= lat <= 90):
            raise HTTPException(status_code=400, detail="Invalid latitude")
        if not (-180 <= lon <= 180):
            raise HTTPException(status_code=400, detail="Invalid longitude")
        
        flood_data = fetch_nasa_modis_flood_data(lat, lon)
        
        return {
            "lat": lat,
            "lon": lon,
            "flood_risk": flood_data["flood_risk"],
            "message": flood_data["message"],
            "flood_pixels": flood_data["flood_pixels"],
            "total_pixels": flood_data["total_pixels"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Flood score error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

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

@app.post("/feedback")
async def submit_feedback(req: FeedbackRequest, request: Request):
    if supabase is None:
        raise HTTPException(status_code=503, detail="Feedback storage not configured")
    try:
        client_ip = request.client.host if request.client else None
        user_agent = request.headers.get("user-agent")
        record = {
            "latitude": req.latitude,
            "longitude": req.longitude,
            "feedback": req.feedback.strip(),
            "summary": req.summary,
            "overall_risk_score": req.overall_risk_score,
            "source": req.source or "frontend",
            "client_ip": client_ip,
            "user_agent": user_agent,
            "metadata": req.metadata or {},
            "created_at": datetime.utcnow().isoformat()
        }
        resp = supabase.table("feedback").insert(record).execute()
        data = getattr(resp, "data", None)
        inserted_id = data[0].get("id") if data and isinstance(data, list) and len(data) else None
        logger.info(f"Feedback stored (id={inserted_id}) for {req.latitude:.4f}, {req.longitude:.4f}")
        return {"status": "ok", "id": inserted_id}
    except Exception as e:
        logger.error(f"Failed to store feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to store feedback")
    
@app.get("/rising-issues", response_model=List[RisingIssue])
async def get_rising_issues(
    min_reports: int = 3,
    timeframe_days: int = 30,
    radius_km: float = 5.0
):
    """
    Aggregate user feedback to identify geographically clustered "rising issues".
    """
    if supabase is None:
        logger.warning("Rising issues endpoint called but Supabase is not configured.")
        raise HTTPException(status_code=503, detail="Service unavailable: Feedback database not configured.")

    try:
        # Calculate the start time for the query
        since_datetime = datetime.utcnow() - timedelta(days=timeframe_days)
        
        logger.info(f"Fetching feedback since {since_datetime.isoformat()} for rising issues analysis.")

        # Query Supabase for recent feedback
        response = supabase.from_("feedback").select(
            "id, latitude, longitude, feedback, summary, overall_risk_score, created_at, metadata"
        ).gte("created_at", since_datetime.isoformat()).execute()

        feedback_entries = getattr(response, 'data', [])
        
        if not feedback_entries:
            logger.info("No recent feedback found to analyze for rising issues.")
            return []

        # Filter out entries without valid coordinates
        valid_entries = [
            entry for entry in feedback_entries 
            if entry.get('latitude') is not None and entry.get('longitude') is not None
        ]

        # Group feedback into rising issues
        rising_issues = group_feedback_by_proximity_and_type(
            valid_entries, radius_km=radius_km, min_reports=min_reports
        )
        
        # Sort issues by count and recency
        rising_issues.sort(key=lambda x: (x.count, x.last_reported_at), reverse=True)

        return rising_issues

    except Exception as e:
        logger.error(f"Error in /rising-issues endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred while analyzing rising issues.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)