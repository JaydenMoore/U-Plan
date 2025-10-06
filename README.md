# 🌍 U-Plan — NASA-Powered Environmental Risk Assessment

A comprehensive web application that helps **urban planners** assess environmental risks and construction conditions in any location using **NASA's open APIs** and advanced data analytics.

Built for **NASA Space Apps Challenge 2025**.

---

## 🚀 Overview

U-Plan lets you **click on any location on a map** and instantly see:
- 🌧️ Average rainfall and potential **flood risk** based on NASA POWER API and MODIS flood detection
- 🌡️ Average temperature and **heat exposure** analysis
- � Air quality with calibrated PM2.5 values and EPA AQI categorization
- 💧 Water risk assessment using WRI Aqueduct 4.0 data
- 👥 Population density analysis from NASA SEDAC GPW-v4 dataset
- � AI-powered risk interpretation with feature importance analysis
- 📊 Comprehensive probabilistic risk modeling with confidence intervals

All data is fetched in real-time from **NASA APIs** and other authoritative sources, with advanced processing and validation.

---

## 🧠 Tech Stack

| Layer | Tech |
|-------|------|
| **Frontend** | React (Vite) + Tailwind CSS + Mapbox GL JS |
| **Backend** | FastAPI (Python) + Probabilistic Risk Modeling |
| **Data Source** | NASA POWER API (climate), NASA MODIS (flood), OpenWeatherMap API (air quality), WRI Aqueduct (water risk), NASA GPW-v4 (population) |
| **AI Components** | SHAP for model interpretability, Gaussian copula for probabilistic risk modeling |
| **Monitoring** | Real-time dashboard with Plotly Dash |

---

## 🧩 Architecture

```
Frontend (React + Mapbox)
↓
Backend (FastAPI)
↓
Data Sources:
- NASA POWER API (climate)
- NASA MODIS (flood detection)
- OpenWeatherMap API (air quality)
- WRI Aqueduct 4.0 (water risk)
- NASA GPW-v4 (population density)
↓
Processing:
- Probabilistic Risk Modeling
- Data Validation
- AI Interpretability
↓
Real-time Dashboard
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/JaydenMoore/U-Plan.git
cd U-Plan
```

---

### 2. Set up the **backend**

#### 🐍 Install dependencies

```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### 📦 API Dependencies in `requirements.txt`

```text
fastapi==0.95.1
uvicorn==0.22.0
python-dotenv==1.0.0
httpx==0.24.0
requests==2.29.0
supabase==0.0.2
numpy==1.24.3
pandas==2.0.1
scikit-learn==1.2.2
scipy==1.10.1
rasterio==1.3.6
plotly==5.14.1
dash==2.9.3
shap==0.41.0
```

#### 🔑 Environment variables

Create `.env` file in `/backend` with your API keys:

```bash
# Required API Keys
OPEN_WEATHER_API=your_openweather_api_key_here

# Optional API Keys (for enhanced features)
GOOGLE_API_KEY=your_google_api_key_here
SUPABASE_URL=your_supabase_url_here
SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key_here
```

#### ▶️ Run FastAPI

```bash
cd backend
./launch.sh
# Or alternatively:
# uvicorn main:app --reload --port 8001
```

Backend runs at `http://localhost:8001` with documentation at `http://localhost:8001/docs`

If you want to run the monitoring dashboard:
```bash
cd backend
python dashboard.py
```
Dashboard runs at `http://localhost:8050`

---

### 3. Set up the **frontend**

#### 💻 Install dependencies

```bash
cd frontend
npm install
```

#### 🔑 Environment variables

Create `.env` file in `/frontend` with your Mapbox token:

```bash
VITE_MAPBOX_TOKEN=your_mapbox_token_here
VITE_API_BASE_URL=http://localhost:8001
```

#### ▶️ Start the frontend

```bash
npm run dev
```

Frontend runs at `http://localhost:5173`

---

## 📊 Enhanced Technical Features

### Real-time Data Reporting Dashboard

- **Live System Monitoring**: Real-time metrics including CPU usage, memory consumption, and API response times
- **Global Risk Assessment Map**: Interactive world map showing recent risk assessments with color-coded risk levels
- **Data Quality Tracking**: Validation status monitoring for all data sources
- Access at `http://localhost:8050` when running dashboard.py

### AI Model Interpretability (SHAP Integration)

- **Feature Importance Analysis**: Identifies which factors contribute most to risk predictions
- **Natural Language Explanations**: Human-readable explanations of model decisions
- **Confidence Scoring**: Prediction reliability assessment

### Automated Data Validation & Anomaly Detection

- **Multi-source Validation**: Comprehensive checks for all data sources
- **Statistical Anomaly Detection**: Z-score analysis to identify unusual data patterns
- **Quality Scoring**: Automated assessment of data reliability

### Probabilistic Risk Modeling

- **Gaussian Copula**: Advanced statistical modeling for risk dependencies
- **Monte Carlo Simulation**: Uncertainty quantification with confidence intervals
- **Multi-dimensional Analysis**: Integration of flood, pollution, water stress, and population factors

---

## 🧰 API References

| Data                       | Source                                                           | Example Endpoint                                                                                                                                             |
| -------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Rainfall & Temperature** | [NASA POWER API](https://power.larc.nasa.gov/docs/services/api/) | `https://power.larc.nasa.gov/api/temporal/monthly/point?parameters=PRECTOT,T2M&community=RE&longitude=-95.36&latitude=29.76&start=2020&end=2024&format=JSON` |
| **Air Quality** | [OpenWeatherMap API](https://openweathermap.org/api/air-pollution) | `http://api.openweathermap.org/data/2.5/air_pollution?lat=50&lon=50&appid={API_key}` |
| **Flood Detection** | [NASA MODIS](https://maps.disasters.nasa.gov/) | `https://maps.disasters.nasa.gov/ags03/rest/services/NRT/modis_flood_1_day/ImageServer/exportImage` |
| **Population Density** | [NASA SEDAC GPW-v4](https://sedac.ciesin.columbia.edu/data/collection/gpw-v4) | Local GeoTIFF processing |
| **Water Risk** | [WRI Aqueduct 4.0](https://www.wri.org/data/aqueduct-global-maps-40-data) | Local geodatabase processing |

---

## 📱 Frontend Features

### Enhanced UI Components
- Interactive map with clickable locations
- Comprehensive risk assessment cards
- Standard EPA AQI visualization with color-coding
- Detailed probabilistic risk displays
- Feature importance visualizations
- Population density analysis views
- Historic and real-time flood risk assessment

### Air Quality Display
- US EPA AQI standards implementation
- Color-coded risk levels (Good, Moderate, Unhealthy for Sensitive Groups, etc.)
- PM2.5, PM10, NO2 measurements with proper units
- Calibrated readings for improved accuracy

---

## 🔍 Key API Endpoints

```bash
# Basic location assessment
POST /assess-location
{
  "latitude": 40.7128,
  "longitude": -74.0060
}

# Enhanced assessment with probabilistic modeling
POST /assess-location-enhanced
{
  "latitude": 40.7128,
  "longitude": -74.0060
}

# Get AI model explanation for specific location
POST /explain-prediction
{
  "latitude": 40.7128,
  "longitude": -74.0060
}

# Check system health
GET /health

# View emerging community-reported issues
GET /rising-issues
```

---

## 🔧 Configuration

### Dashboard Settings
```python
DASHBOARD_CONFIG = {
    "update_interval": 30,      # Data refresh rate (seconds)
    "max_data_points": 100,     # Historical data limit
    "api_base_url": "http://localhost:8001",
    "refresh_rate": 5000,       # UI refresh rate (milliseconds)
}
```

### Data Validation Settings
```python
# Validation thresholds
VALIDATION_THRESHOLDS = {
    "temperature": (-50, 60),       # Celsius
    "precipitation": (0, 1000),     # mm/month
    "pm25": (0, 500),              # μg/m³
    "water_stress": (0, 5),        # WRI scale
}
```

---

## 📊 Advanced Risk Assessment

### Probabilistic Risk Model
The system uses a Gaussian copula-based probabilistic risk model to assess environmental hazards:

```python
# Vulnerability function combining multiple risk factors
def vulnerability_function(self, flood_prob, pollution_pm25, water_risk=0.0):
    """
    Logistic vulnerability function combining flood, pollution, and water risk effects
    Returns vulnerability score between 0 and vmax
    """
    alpha_f = self.vuln_params['alpha_f']
    alpha_p = self.vuln_params['alpha_p']
    alpha_w = self.vuln_params['alpha_w']
    f0 = self.vuln_params['f0']
    p0 = self.vuln_params['p0']
    w0 = self.vuln_params['w0'] if self.vuln_params['w0'] is not None else 0.2
    vmax = self.vuln_params['vmax']
    
    z = alpha_f * (flood_prob - f0) + alpha_p * (pollution_pm25 - p0) + alpha_w * (water_risk - w0)
    return vmax / (1.0 + np.exp(-z))
```

### Air Quality Assessment
The system uses the US EPA AQI standard:

```python
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
    
    # Find the appropriate breakpoint and calculate AQI
    for bp_lo, bp_hi, aqi_lo, aqi_hi in breakpoints:
        if bp_lo <= pm25 <= bp_hi:
            # Linear interpolation formula
            aqi = ((aqi_hi - aqi_lo) / (bp_hi - bp_lo)) * (pm25 - bp_lo) + aqi_lo
            return round(aqi)
```

---

## 🧪 Test Locally

1. Run backend with `./launch.sh` or `uvicorn main:app --reload --port 8001`
2. Run frontend with `npm run dev`
3. Open browser and navigate to http://localhost:5173
4. Click anywhere on the map to see comprehensive risk assessment
5. View dashboard at http://localhost:8050 (if running dashboard.py)

---

## 📊 Data Sources & Attribution

- **NASA POWER**: Climate data (CC BY 4.0)
- **NASA MODIS**: Real-time flood detection
- **OpenWeatherMap**: Air quality data (Commercial license)
- **WRI Aqueduct 4.0**: Water risk data (CC BY 4.0)
- **NASA SEDAC GPW-v4**: Population data (CC BY 4.0)

### Software Libraries
- **SHAP**: Model interpretability (MIT License)
- **Plotly Dash**: Dashboard framework (MIT License)
- **FastAPI**: Web framework (MIT License)
- **React**: Frontend framework (MIT License)
- **Tailwind CSS**: Utility-first CSS framework (MIT License)

---

## 🔮 Future Enhancements

### Planned Features
1. **Temporal Fusion Transformers**: Advanced time series forecasting
2. **Mobile Web Interface**: Lightweight responsive design
3. **SMS Notifications**: Alert system for high-risk events
4. **Multilingual Support**: International accessibility
5. **Community Feedback Module**: Crowdsourced validation system

---

**Built with ❤️ for NASA Space Apps Challenge 2025**
