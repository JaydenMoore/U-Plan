"""
GPW-v4 Population Data Reader
Reads Gridded Population of the World (GPW) version 4 raster data for Urban Planner AI
"""

import rasterio
import numpy as np
from typing import Optional, Dict, Tuple
import logging
import os

logger = logging.getLogger(__name__)

class PopulationDataReader:
    def __init__(self, tif_path: str):
        """
        Initialize the population data reader with the GPW TIF file
        
        Args:
            tif_path: Path to the GPW population density TIF file
        """
        self.tif_path = tif_path
        self.dataset = None
        self.transform = None
        self.crs = None
        self.bounds = None
        
        if os.path.exists(tif_path):
            try:
                self.dataset = rasterio.open(tif_path)
                self.transform = self.dataset.transform
                self.crs = self.dataset.crs
                self.bounds = self.dataset.bounds
                logger.info(f"Loaded GPW population data: {self.dataset.shape} pixels, CRS: {self.crs}")
                logger.info(f"Data bounds: {self.bounds}")
            except Exception as e:
                logger.error(f"Failed to load population data: {e}")
                self.dataset = None
        else:
            logger.warning(f"Population data file not found: {tif_path}")
    
    def get_population_density(self, lat: float, lon: float) -> Optional[float]:
        """
        Get population density at a specific coordinate
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Population density in people per km² or None if data unavailable
        """
        if not self.dataset:
            return None
            
        try:
            # Check if coordinates are within bounds
            if not (self.bounds.left <= lon <= self.bounds.right and 
                   self.bounds.bottom <= lat <= self.bounds.top):
                return 0.0  # Outside data bounds
            
            # Convert lat/lon to pixel coordinates
            row, col = rasterio.transform.rowcol(self.transform, lon, lat)
            
            # Check if coordinates are within raster bounds
            if 0 <= row < self.dataset.height and 0 <= col < self.dataset.width:
                # Read the pixel value
                window = rasterio.windows.Window(col, row, 1, 1)
                data = self.dataset.read(1, window=window)
                
                if data.size > 0:
                    population_density = data[0, 0]
                    
                    # Check for no-data values (GPW uses -9999 for no data)
                    if np.isnan(population_density) or population_density < 0:
                        return 0.0
                        
                    return float(population_density)
                else:
                    return 0.0
            else:
                return 0.0  # Outside raster bounds
                
        except Exception as e:
            logger.error(f"Error reading population data at {lat}, {lon}: {e}")
            return None
    
    def get_population_stats_in_area(self, lat: float, lon: float, radius_km: float = 5.0) -> Dict:
        """
        Get population statistics within a radius of a point
        
        Args:
            lat: Center latitude
            lon: Center longitude  
            radius_km: Radius in kilometers (default: 5km)
            
        Returns:
            Dictionary with population statistics
        """
        if not self.dataset:
            return {"error": "No population data available"}
            
        try:
            # Convert radius to degrees (rough approximation)
            # 1 degree ≈ 111 km at equator, adjust for latitude
            lat_factor = 111.0
            lon_factor = 111.0 * np.cos(np.radians(lat))
            
            radius_lat = radius_km / lat_factor
            radius_lon = radius_km / lon_factor
            
            # Define bounding box
            min_lon, max_lon = lon - radius_lon, lon + radius_lon
            min_lat, max_lat = lat - radius_lat, lat + radius_lat
            
            # Check bounds
            if not (self.bounds.left <= max_lon and min_lon <= self.bounds.right and 
                   self.bounds.bottom <= max_lat and min_lat <= self.bounds.top):
                return {"error": "Area outside data bounds"}
            
            # Convert to pixel coordinates
            top, left = rasterio.transform.rowcol(self.transform, min_lon, max_lat)
            bottom, right = rasterio.transform.rowcol(self.transform, max_lon, min_lat)
            
            # Ensure bounds are within dataset
            left = max(0, min(left, self.dataset.width - 1))
            right = max(0, min(right, self.dataset.width - 1))
            top = max(0, min(top, self.dataset.height - 1))
            bottom = max(0, min(bottom, self.dataset.height - 1))
            
            # Ensure we have a valid window
            if left >= right or top >= bottom:
                return {"error": "Invalid area bounds"}
            
            # Read the data window
            width = right - left + 1
            height = bottom - top + 1
            window = rasterio.windows.Window(left, top, width, height)
            data = self.dataset.read(1, window=window)
            
            # Filter out no-data values
            valid_mask = (~np.isnan(data)) & (data >= 0)
            valid_data = data[valid_mask]
            
            if len(valid_data) == 0:
                return {
                    "mean_density": 0.0,
                    "max_density": 0.0,
                    "min_density": 0.0,
                    "total_pixels": int(data.size),
                    "populated_pixels": 0,
                    "area_km2": np.pi * radius_km * radius_km
                }
            
            return {
                "mean_density": float(np.mean(valid_data)),
                "max_density": float(np.max(valid_data)),
                "min_density": float(np.min(valid_data)),
                "median_density": float(np.median(valid_data)),
                "total_pixels": int(data.size),
                "populated_pixels": int(len(valid_data)),
                "area_km2": np.pi * radius_km * radius_km,
                "population_category": self._categorize_density(np.mean(valid_data))
            }
            
        except Exception as e:
            logger.error(f"Error calculating population stats: {e}")
            return {"error": str(e)}
    
    def _categorize_density(self, density: float) -> str:
        """Categorize population density"""
        if density < 1:
            return "Uninhabited"
        elif density < 10:
            return "Very Low Density"
        elif density < 50:
            return "Low Density"
        elif density < 150:
            return "Medium Density"
        elif density < 500:
            return "High Density"
        elif density < 1000:
            return "Very High Density"
        else:
            return "Extremely High Density"
    
    def get_data_info(self) -> Dict:
        """Get basic information about the dataset"""
        if not self.dataset:
            return {"error": "No dataset loaded"}
        
        return {
            "width": self.dataset.width,
            "height": self.dataset.height,
            "crs": str(self.crs),
            "bounds": {
                "left": self.bounds.left,
                "bottom": self.bounds.bottom,
                "right": self.bounds.right,
                "top": self.bounds.top
            },
            "transform": list(self.transform),
            "data_type": str(self.dataset.dtypes[0]),
            "no_data_value": self.dataset.nodata
        }
    
    def close(self):
        """Close the dataset"""
        if self.dataset:
            self.dataset.close()
            self.dataset = None

def main():
    """Test the population data reader"""
    tif_path = "/Users/jaydenmoore/Documents/NASA Space Apps/population density/gpw-v4-population-density-adjusted-to-2015-unwpp-country-totals_2020.tif"
    
    try:
        # Initialize reader
        reader = PopulationDataReader(tif_path)
        
        # Get basic info
        info = reader.get_data_info()
        print("Population Data Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Test with a location (e.g., New York City)
        lat, lon = 40.7128, -74.0060
        density = reader.get_population_density(lat, lon)
        print(f"\nPopulation density at NYC ({lat}, {lon}): {density} people/km²")
        
        # Get area statistics
        stats = reader.get_population_stats_in_area(lat, lon, radius_km=10)
        print(f"\nPopulation stats within 10km of NYC:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        reader.close()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()