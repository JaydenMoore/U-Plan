"""
Data Validation and Anomaly Detection Module
Implements automated data quality checks and anomaly detection for all data sources
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    anomalies: List[str]
    quality_score: float
    metadata: Dict[str, Any]
    timestamp: datetime

@dataclass
class DataSourceMetadata:
    """Metadata for tracking data sources and versions"""
    source_name: str
    version: str
    last_updated: datetime
    schema_version: str
    data_lineage: List[str]
    quality_metrics: Dict[str, float]
    known_limitations: List[str]

class DataValidator:
    """Comprehensive data validation and quality assessment"""
    
    def __init__(self):
        self.validation_rules = {
            'climate_data': {
                'temperature_range': (-60, 60),  # °C
                'rainfall_range': (0, 5000),     # mm/month  
                'required_fields': ['temperature_c', 'rainfall_mm']
            },
            'air_quality': {
                'aqi_range': (0, 500),
                'pm25_range': (0, 1000),         # μg/m³
                'pm10_range': (0, 1500),         # μg/m³
                'required_fields': ['air_quality_index', 'pm2_5']
            },
            'water_risk': {
                'stress_range': (0, 5),
                'drought_range': (0, 5),
                'required_fields': ['baseline_water_stress', 'drought_risk']
            },
            'population': {
                'density_range': (0, 50000),     # people/km²
                'required_fields': ['population_density']
            }
        }
        
        self.anomaly_thresholds = {
            'z_score_threshold': 3.0,
            'iqr_multiplier': 1.5,
            'missing_data_threshold': 0.05  # 5% missing data triggers warning
        }
    
    def validate_data_source(self, data: Dict[str, Any], source_type: str) -> ValidationResult:
        """Validate a data source against defined rules"""
        anomalies = []
        quality_score = 1.0
        
        if source_type not in self.validation_rules:
            return ValidationResult(
                is_valid=False,
                anomalies=[f"Unknown data source type: {source_type}"],
                quality_score=0.0,
                metadata={"source_type": source_type},
                timestamp=datetime.now()
            )
        
        rules = self.validation_rules[source_type]
        
        # Check required fields
        missing_fields = []
        for field in rules['required_fields']:
            if field not in data or data[field] is None:
                missing_fields.append(field)
        
        if missing_fields:
            anomalies.append(f"Missing required fields: {missing_fields}")
            quality_score *= 0.7
        
        # Range validation
        for field, value in data.items():
            if value is None:
                continue
                
            range_key = f"{field}_range"
            if range_key in rules:
                min_val, max_val = rules[range_key]
                if not (min_val <= value <= max_val):
                    anomalies.append(f"{field} value {value} outside valid range [{min_val}, {max_val}]")
                    quality_score *= 0.8
        
        # Statistical anomaly detection
        numeric_fields = {k: v for k, v in data.items() if isinstance(v, (int, float)) and v is not None}
        statistical_anomalies = self._detect_statistical_anomalies(numeric_fields, source_type)
        anomalies.extend(statistical_anomalies)
        
        if statistical_anomalies:
            quality_score *= 0.9
        
        is_valid = quality_score > 0.5 and len(anomalies) < 3
        
        return ValidationResult(
            is_valid=is_valid,
            anomalies=anomalies,
            quality_score=quality_score,
            metadata={
                "source_type": source_type,
                "field_count": len(data),
                "numeric_field_count": len(numeric_fields)
            },
            timestamp=datetime.now()
        )
    
    def _detect_statistical_anomalies(self, data: Dict[str, float], source_type: str) -> List[str]:
        """Detect statistical anomalies in numeric data"""
        anomalies = []
        
        if not data:
            return anomalies
        
        values = list(data.values())
        if len(values) < 2:
            return anomalies
        
        # Z-score based detection (for single values, compare against typical ranges)
        for field, value in data.items():
            if source_type == 'climate_data':
                if field == 'temperature_c':
                    # Global temperature typically -20 to 40°C
                    if abs(value - 10) > 30:  # 30°C deviation from global mean
                        anomalies.append(f"Temperature {value}°C appears unusual for global average")
                elif field == 'rainfall_mm':
                    # Most locations: 0-200mm/month
                    if value > 500:
                        anomalies.append(f"Rainfall {value}mm/month is extremely high")
            
            elif source_type == 'air_quality':
                if field == 'pm2_5' and value > 150:
                    anomalies.append(f"PM2.5 level {value}μg/m³ indicates severe pollution")
                elif field == 'air_quality_index' and value > 300:
                    anomalies.append(f"AQI {value} indicates hazardous air quality")
        
        return anomalies
    
    def generate_quality_report(self, validation_results: List[ValidationResult]) -> Dict[str, Any]:
        """Generate comprehensive data quality report"""
        total_sources = len(validation_results)
        valid_sources = sum(1 for r in validation_results if r.is_valid)
        
        avg_quality = np.mean([r.quality_score for r in validation_results])
        all_anomalies = []
        for r in validation_results:
            all_anomalies.extend(r.anomalies)
        
        return {
            "report_timestamp": datetime.now().isoformat(),
            "summary": {
                "total_data_sources": total_sources,
                "valid_sources": valid_sources,
                "validation_rate": valid_sources / total_sources if total_sources > 0 else 0,
                "average_quality_score": round(avg_quality, 3)
            },
            "anomalies": {
                "total_count": len(all_anomalies),
                "unique_anomalies": list(set(all_anomalies))
            },
            "recommendations": self._generate_recommendations(validation_results)
        }
    
    def _generate_recommendations(self, results: List[ValidationResult]) -> List[str]:
        """Generate actionable recommendations based on validation results"""
        recommendations = []
        
        low_quality_count = sum(1 for r in results if r.quality_score < 0.7)
        if low_quality_count > len(results) * 0.3:
            recommendations.append("Consider implementing additional data preprocessing steps")
        
        missing_data_sources = [r for r in results if not r.is_valid]
        if missing_data_sources:
            recommendations.append("Review data collection processes for failing sources")
        
        common_anomalies = {}
        for result in results:
            for anomaly in result.anomalies:
                common_anomalies[anomaly] = common_anomalies.get(anomaly, 0) + 1
        
        frequent_anomalies = [a for a, count in common_anomalies.items() if count > 1]
        if frequent_anomalies:
            recommendations.append(f"Address recurring data issues: {frequent_anomalies}")
        
        return recommendations

class MetadataTracker:
    """Tracks metadata and lineage for all data sources"""
    
    def __init__(self):
        self.metadata_store: Dict[str, DataSourceMetadata] = {}
        
    def register_data_source(self, 
                           source_name: str,
                           version: str,
                           schema_version: str = "1.0",
                           data_lineage: List[str] = None,
                           known_limitations: List[str] = None) -> DataSourceMetadata:
        """Register a new data source with metadata"""
        
        metadata = DataSourceMetadata(
            source_name=source_name,
            version=version,
            last_updated=datetime.now(),
            schema_version=schema_version,
            data_lineage=data_lineage or [],
            quality_metrics={},
            known_limitations=known_limitations or []
        )
        
        self.metadata_store[source_name] = metadata
        logger.info(f"Registered data source: {source_name} v{version}")
        
        return metadata
    
    def update_quality_metrics(self, source_name: str, metrics: Dict[str, float]):
        """Update quality metrics for a data source"""
        if source_name in self.metadata_store:
            self.metadata_store[source_name].quality_metrics.update(metrics)
            self.metadata_store[source_name].last_updated = datetime.now()
    
    def get_data_lineage_report(self) -> Dict[str, Any]:
        """Generate comprehensive data lineage report"""
        return {
            "report_timestamp": datetime.now().isoformat(),
            "data_sources": {
                name: {
                    "version": meta.version,
                    "last_updated": meta.last_updated.isoformat(),
                    "schema_version": meta.schema_version,
                    "lineage": meta.data_lineage,
                    "quality_metrics": meta.quality_metrics,
                    "limitations": meta.known_limitations
                }
                for name, meta in self.metadata_store.items()
            }
        }
    
    def export_metadata(self, filepath: str):
        """Export metadata to JSON file"""
        report = self.get_data_lineage_report()
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Metadata exported to {filepath}")

# Global instances
data_validator = DataValidator()
metadata_tracker = MetadataTracker()

# Initialize core data sources
metadata_tracker.register_data_source(
    "NASA_POWER", 
    "2.2.0",
    data_lineage=["NASA Goddard Space Flight Center", "POWER API"],
    known_limitations=["5-year averages only", "Limited temporal resolution"]
)

metadata_tracker.register_data_source(
    "OpenWeatherMap_AirPollution", 
    "3.0",
    data_lineage=["OpenWeatherMap", "Global monitoring stations"],
    known_limitations=["Real-time data gaps in remote areas", "Limited historical data"]
)

metadata_tracker.register_data_source(
    "WRI_Aqueduct", 
    "4.0",
    data_lineage=["World Resources Institute", "Global basin data"],
    known_limitations=["Regional aggregation", "Annual updates only"]
)

metadata_tracker.register_data_source(
    "GPW_PopulationDensity",
    "4.11",
    data_lineage=["NASA SEDAC", "National statistical offices"],
    known_limitations=["5-year estimates", "Administrative boundary dependencies"]
)