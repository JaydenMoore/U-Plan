"""
Aqueduct Water Risk Atlas Data Reader
Integrates WRI Aqueduct 4.0 Water Risk data for comprehensive water risk assessment
"""

import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, Optional, Tuple
from math import radians, cos, sin, asin, sqrt

logger = logging.getLogger(__name__)

class AqueductWaterRiskReader:
    """
    Reader for World Resources Institute Aqueduct 4.0 Water Risk Atlas data
    Provides water stress, drought risk, and flood risk indicators
    """
    
    def __init__(self, csv_path: str = None):
        """
        Initialize with Aqueduct CSV data
        
        Args:
            csv_path: Path to Aqueduct baseline annual CSV file
        """
        self.data = None
        self.csv_path = csv_path or "../data/Aqueduct_WaterRisk/CVS/Aqueduct40_baseline_annual_y2023m07d05.csv"
        
        if os.path.exists(self.csv_path):
            try:
                self.load_data()
                logger.info(f"Loaded Aqueduct water risk data: {len(self.data)} records")
            except Exception as e:
                logger.error(f"Failed to load Aqueduct data: {e}")
                self.data = None
        else:
            logger.warning(f"Aqueduct data file not found: {self.csv_path}")
    
    def load_data(self):
        """Load and preprocess Aqueduct CSV data"""
        try:
            # Load the CSV data
            self.data = pd.read_csv(self.csv_path)
            
            # Clean and prepare the data
            if 'string_id' in self.data.columns:
                # Remove any rows with missing essential data
                essential_cols = ['string_id', 'gid_0', 'name_0']
                self.data = self.data.dropna(subset=essential_cols)
                
            logger.info(f"Aqueduct data shape: {self.data.shape}")
            logger.info(f"Countries in dataset: {len(self.data['name_0'].unique()) if 'name_0' in self.data.columns else 'Unknown'}")
                
        except Exception as e:
            logger.error(f"Error loading Aqueduct data: {e}")
            raise
    
    def get_water_risk_by_country(self, lat: float, lon: float, country_name: str = None) -> Dict:
        """
        Get water risk indicators by country/region matching
        
        Args:
            lat: Latitude
            lon: Longitude  
            country_name: Optional country name for direct matching
            
        Returns:
            Dictionary with water risk indicators
        """
        if self.data is None:
            logger.warning("Aqueduct data not available, using fallback")
            return self._get_fallback_water_risk(lat, lon)
        
        try:
            # Get country from coordinates using a simple geocoding approach
            if not country_name:
                country_name = self._geocode_to_country(lat, lon)
            
            logger.info(f"Looking for water risk data for country: {country_name}")
            
            # Filter data by country
            country_data = self.data[self.data['name_0'].str.contains(country_name, case=False, na=False)]
            
            logger.info(f"Found {len(country_data)} records for {country_name}")
            
            if len(country_data) == 0:
                # Try ISO code matching if available
                iso_code = self._get_iso_code_from_coords(lat, lon)
                logger.info(f"Trying ISO code matching: {iso_code}")
                if iso_code:
                    country_data = self.data[self.data['gid_0'] == iso_code]
                    logger.info(f"Found {len(country_data)} records for ISO {iso_code}")
            
            if len(country_data) > 0:
                # Try to find a more specific regional match first
                region_name = self._get_region_from_coords(lat, lon)
                if region_name:
                    regional_data = country_data[country_data['name_1'].str.contains(region_name, case=False, na=False)]
                    if len(regional_data) > 0:
                        representative_record = regional_data.iloc[0]
                        logger.info(f"Using regional record: {representative_record.get('name_1', 'Unknown region')}")
                    else:
                        # Fall back to country-level data but try to pick a central/populated region
                        # Avoid Alaska for US coordinates that are not in Alaska
                        if country_name == "United States" and not (lat > 60 or lat < 25):
                            non_alaska = country_data[~country_data['name_1'].str.contains('Alaska', case=False, na=False)]
                            if len(non_alaska) > 0:
                                representative_record = non_alaska.iloc[0]
                                logger.info(f"Using non-Alaska US record: {representative_record.get('name_1', 'Unknown region')}")
                            else:
                                representative_record = country_data.iloc[0]
                                logger.info(f"Using first record: {representative_record.get('name_1', 'Unknown region')}")
                        else:
                            representative_record = country_data.iloc[0]
                            logger.info(f"Using first record: {representative_record.get('name_1', 'Unknown region')}")
                else:
                    # No region matching, use first available record
                    representative_record = country_data.iloc[0]
                    logger.info(f"Using first record: {representative_record.get('name_1', 'Unknown region')}")
                
                return self._extract_water_risk_indicators(representative_record)
            else:
                logger.warning(f"No water risk data found for {country_name}, using fallback")
                return self._get_fallback_water_risk(lat, lon)
            
        except Exception as e:
            logger.error(f"Error getting water risk data: {e}")
            return self._get_fallback_water_risk(lat, lon)
    
    def _extract_water_risk_indicators(self, record) -> Dict:
        """Extract and normalize water risk indicators from Aqueduct record"""
        
        # Extract key indicators (scores are normalized 0-5)
        baseline_water_stress = self._safe_float(record.get('bws_score', 0))
        baseline_water_depletion = self._safe_float(record.get('bwd_score', 0))
        drought_risk = self._safe_float(record.get('drr_score', 0))
        riverine_flood_risk = self._safe_float(record.get('rfr_score', 0))
        coastal_flood_risk = self._safe_float(record.get('cfr_score', 0))
        groundwater_decline = self._safe_float(record.get('gtd_score', 0))
        
        # Overall risk scores
        overall_water_risk = self._safe_float(record.get('w_awr_def_tot_score', 0))
        physical_quantity_risk = self._safe_float(record.get('w_awr_def_qan_score', 0))
        physical_quality_risk = self._safe_float(record.get('w_awr_def_qal_score', 0))
        
        # Calculate composite scores
        water_stress_indicators = [baseline_water_stress, baseline_water_depletion, drought_risk]
        flood_indicators = [riverine_flood_risk, coastal_flood_risk]
        
        water_stress_score = np.mean([x for x in water_stress_indicators if x > 0])
        flood_risk_score = np.mean([x for x in flood_indicators if x > 0])
        
        # Normalize to 0-1 scale for integration with our model
        water_stress_normalized = water_stress_score / 5.0 if water_stress_score > 0 else 0.0
        flood_risk_normalized = flood_risk_score / 5.0 if flood_risk_score > 0 else 0.0
        overall_risk_normalized = overall_water_risk / 5.0 if overall_water_risk > 0 else 0.0
        
        return {
            'baseline_water_stress': round(baseline_water_stress, 2),
            'water_stress_normalized': round(water_stress_normalized, 3),
            'baseline_water_depletion': round(baseline_water_depletion, 2),
            'drought_risk': round(drought_risk, 2),
            'riverine_flood_risk': round(riverine_flood_risk, 2),
            'coastal_flood_risk': round(coastal_flood_risk, 2),
            'groundwater_decline': round(groundwater_decline, 2),
            'flood_risk_score': round(flood_risk_score, 2),
            'flood_risk_normalized': round(flood_risk_normalized, 3),
            'overall_water_risk': round(overall_water_risk, 2),
            'overall_risk_normalized': round(overall_risk_normalized, 3),
            'physical_quantity_risk': round(physical_quantity_risk, 2),
            'physical_quality_risk': round(physical_quality_risk, 2),
            'water_stress_category': self._categorize_risk(baseline_water_stress),
            'drought_category': self._categorize_risk(drought_risk),
            'overall_category': self._categorize_risk(overall_water_risk),
            'country': record.get('name_0', 'Unknown'),
            'region': record.get('name_1', 'Unknown'),
            'source': 'WRI Aqueduct 4.0 Water Risk Atlas',
            'aqueduct_id': record.get('string_id', 'Unknown')
        }
    
    def _safe_float(self, value, default=0.0):
        """Safely convert value to float, handling -9999 (NoData) values"""
        try:
            val = float(value)
            return default if val == -9999.0 or pd.isna(val) else val
        except (ValueError, TypeError):
            return default
    
    def _categorize_risk(self, score: float) -> str:
        """Categorize risk level based on Aqueduct 0-5 scale"""
        if score >= 4.0:
            return "Extremely High"
        elif score >= 3.0:
            return "High"
        elif score >= 2.0:
            return "Medium-High"
        elif score >= 1.0:
            return "Low-Medium"
        elif score > 0:
            return "Low"
        else:
            return "No Data"
    
    def _get_region_from_coords(self, lat: float, lon: float) -> str:
        """
        Get region/state name from coordinates for more specific matching
        """
        # Simple regional mapping for major areas
        if -130 <= lon <= -65 and 25 <= lat <= 50:  # Continental US
            if -125 <= lon <= -114 and 32 <= lat <= 42:  # California region
                return "California"
            elif -106 <= lon <= -93 and 25 <= lat <= 36:  # Texas region
                return "Texas"
            elif -88 <= lon <= -68 and 40 <= lat <= 47:  # Great Lakes/Northeast
                return "New York"
            elif -84 <= lon <= -75 and 32 <= lat <= 40:  # Southeast
                return "Florida"
        elif 45 <= lat <= 85 and -180 <= lon <= -65:  # Alaska
            return "Alaska"
        
        # Add more regional mappings as needed
        return None
    
    def _geocode_to_country(self, lat: float, lon: float) -> str:
        """
        Simple geocoding to country based on lat/lon
        In production, you'd use a proper geocoding service
        """
        # More precise country mapping with better boundaries
        country_mapping = [
            # Asia (most specific first to avoid overlaps)
            ((30, 46, 129, 146), "Japan"),  # Japan islands
            ((33, 38, 126, 132), "South Korea"), # South Korea
            ((1, 7, 103, 105), "Singapore"),
            ((10, 35, 95, 141), "Southeast Asia"),
            ((18, 54, 73, 135), "China"),  # China proper
            ((8, 37, 68, 97), "India"),
            ((55, 82, 20, 180), "Russia"),  # Russia
            
            # North America (more specific ranges)
            ((25, 50, -130, -65), "United States"),
            ((45, 85, -180, -65), "Canada"),
            ((14, 33, -118, -86), "Mexico"),
            
            # Europe (more specific)
            ((50, 72, -10, 40), "Northern Europe"),
            ((35, 55, -10, 40), "Southern Europe"),
            
            # Africa (regional)
            ((22, 38, 25, 55), "North Africa"),
            ((-5, 20, 25, 55), "East Africa"),
            ((-35, 5, -20, 25), "Southern Africa"),
            
            # South America
            ((-56, 13, -82, -35), "South America"),
            
            # Australia
            ((-45, -10, 110, 160), "Australia")
        ]
        
        # Process in order of specificity (smaller regions first)
        for (min_lat, max_lat, min_lon, max_lon), country in country_mapping:
            if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
                return country
        
        return "Unknown"
    
    def _get_iso_code_from_coords(self, lat: float, lon: float) -> str:
        """Get ISO country code from coordinates"""
        # Enhanced mapping with more countries
        country_iso_mapping = {
            "United States": "USA",
            "Canada": "CAN", 
            "Mexico": "MEX",
            "China": "CHN",
            "Japan": "JPN",
            "South Korea": "KOR",
            "India": "IND",
            "Singapore": "SGP",
            "Southeast Asia": "THA",  # Default to Thailand for SE Asia
            "Russia": "RUS",
            "Northern Europe": "GBR",  # Default to UK
            "Southern Europe": "ITA",  # Default to Italy
            "North Africa": "EGY",     # Default to Egypt
            "East Africa": "KEN",      # Default to Kenya
            "Southern Africa": "ZAF",  # Default to South Africa
            "South America": "BRA",    # Default to Brazil
            "Australia": "AUS"
        }
        
        country = self._geocode_to_country(lat, lon)
        return country_iso_mapping.get(country, "")
    
    def _get_fallback_water_risk(self, lat: float, lon: float) -> Dict:
        """Provide fallback water risk data when Aqueduct data is unavailable"""
        # Basic risk estimation based on geographic location
        baseline_risk = 1.0  # Default low risk
        
        # Higher risk for arid regions
        if abs(lat) < 35:  # Tropical/subtropical
            if -30 < lon < 50:  # Africa/Middle East
                baseline_risk = 3.0
            elif 60 < lon < 100:  # South Asia
                baseline_risk = 2.5
            elif -120 < lon < -80:  # SW United States/Mexico
                baseline_risk = 2.0
        
        return {
            'baseline_water_stress': baseline_risk,
            'water_stress_normalized': baseline_risk / 5.0,
            'baseline_water_depletion': baseline_risk * 0.8,
            'drought_risk': baseline_risk * 0.7,
            'riverine_flood_risk': 1.0,
            'coastal_flood_risk': 1.0 if abs(lat) < 45 else 0.5,
            'groundwater_decline': baseline_risk * 0.6,
            'flood_risk_score': 1.0,
            'flood_risk_normalized': 0.2,
            'overall_water_risk': baseline_risk,
            'overall_risk_normalized': baseline_risk / 5.0,
            'physical_quantity_risk': baseline_risk,
            'physical_quality_risk': baseline_risk * 0.8,
            'water_stress_category': self._categorize_risk(baseline_risk),
            'drought_category': self._categorize_risk(baseline_risk * 0.7),
            'overall_category': self._categorize_risk(baseline_risk),
            'country': 'Unknown',
            'region': 'Unknown',
            'source': 'Fallback estimation',
            'aqueduct_id': 'Unknown'
        }