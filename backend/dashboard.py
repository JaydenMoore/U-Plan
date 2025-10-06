"""
Real-time Data Reporting Dashboard using Plotly Dash
Provides comprehensive monitoring and visualization of risk assessment data
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import asyncio
import logging
from typing import Dict, List, Any, Optional
import threading
import time
import json
from dataclasses import dataclass

# Dashboard configuration
DASHBOARD_CONFIG = {
    "update_interval": 30,  # seconds
    "max_data_points": 100,
    "api_base_url": "http://localhost:8001",
    "refresh_rate": 5000,  # milliseconds
}

# Global data store for dashboard
dashboard_data = {
    "assessments": [],
    "system_metrics": [],
    "data_validation_logs": [],
    "model_explanations": [],
    "last_update": None
}

logger = logging.getLogger(__name__)

@dataclass
class DashboardMetrics:
    """Dashboard metrics container"""
    total_assessments: int
    avg_risk_score: float
    high_risk_locations: int
    data_freshness: str
    system_health: str
    validation_success_rate: float

def create_dashboard_app():
    """Create and configure the Dash application"""
    
    app = dash.Dash(__name__, external_stylesheets=[
        "https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.1.3/css/bootstrap.min.css"
    ])
    
    app.title = "Urban Planner AI - Real-time Dashboard"
    
    # Define layout
    app.layout = html.Div([
        # Header
        html.Div([
            html.H1("🌍 Urban Planner AI Dashboard", className="text-center mb-4"),
            html.P("Real-time monitoring of environmental risk assessments", 
                   className="text-center text-muted"),
            html.Hr()
        ], className="container-fluid bg-light py-3"),
        
        # Auto-refresh interval
        dcc.Interval(
            id='interval-component',
            interval=DASHBOARD_CONFIG["refresh_rate"],
            n_intervals=0
        ),
        
        # Main content
        html.Div([
            # Metrics row
            html.Div([
                html.H3("📊 System Metrics", className="mb-3"),
                html.Div(id="metrics-cards", className="row mb-4")
            ]),
            
            # Charts row
            html.Div([
                html.Div([
                    html.H4("🗺️ Global Risk Assessment Map"),
                    dcc.Graph(id="risk-map")
                ], className="col-lg-6 mb-4"),
                
                html.Div([
                    html.H4("📈 Risk Score Trends"),
                    dcc.Graph(id="risk-trends")
                ], className="col-lg-6 mb-4")
            ], className="row"),
            
            # Data quality and model performance
            html.Div([
                html.Div([
                    html.H4("🔍 Data Quality Status"),
                    dcc.Graph(id="data-quality-chart")
                ], className="col-lg-6 mb-4"),
                
                html.Div([
                    html.H4("🤖 Model Performance"),
                    dcc.Graph(id="model-performance")
                ], className="col-lg-6 mb-4")
            ], className="row"),
            
            # Recent assessments table
            html.Div([
                html.H4("📋 Recent Assessments"),
                html.Div(id="recent-assessments-table", className="table-responsive")
            ], className="mb-4"),
            
            # System logs
            html.Div([
                html.H4("📝 System Activity Log"),
                html.Div(id="system-logs", className="bg-dark text-light p-3 rounded",
                        style={"height": "200px", "overflow-y": "scroll"})
            ])
            
        ], className="container-fluid px-4")
    ])
    
    return app

def create_metrics_cards(metrics: DashboardMetrics) -> html.Div:
    """Create bootstrap cards for key metrics"""
    
    cards = [
        {
            "title": "Total Assessments",
            "value": f"{metrics.total_assessments:,}",
            "icon": "📊",
            "color": "primary"
        },
        {
            "title": "Average Risk Score",
            "value": f"{metrics.avg_risk_score:.2f}/10",
            "icon": "⚠️",
            "color": "warning" if metrics.avg_risk_score > 6 else "success"
        },
        {
            "title": "High Risk Locations",
            "value": f"{metrics.high_risk_locations}",
            "icon": "🚨",
            "color": "danger" if metrics.high_risk_locations > 0 else "success"
        },
        {
            "title": "Data Freshness",
            "value": metrics.data_freshness,
            "icon": "🕒",
            "color": "info"
        },
        {
            "title": "System Health",
            "value": metrics.system_health,
            "icon": "💚" if metrics.system_health == "Healthy" else "💛",
            "color": "success" if metrics.system_health == "Healthy" else "warning"
        },
        {
            "title": "Validation Success",
            "value": f"{metrics.validation_success_rate:.1%}",
            "icon": "✅",
            "color": "success" if metrics.validation_success_rate > 0.95 else "warning"
        }
    ]
    
    card_elements = []
    for card in cards:
        card_element = html.Div([
            html.Div([
                html.Div([
                    html.H2(f"{card['icon']} {card['value']}", className="card-title"),
                    html.P(card["title"], className="card-text text-muted")
                ], className="card-body text-center")
            ], className=f"card border-{card['color']} h-100")
        ], className="col-lg-2 col-md-4 col-sm-6 mb-3")
        card_elements.append(card_element)
    
    return html.Div(card_elements, className="row")

def fetch_dashboard_data():
    """Fetch data for dashboard from API endpoints"""
    try:
        # Simulate fetching data from various endpoints
        base_url = DASHBOARD_CONFIG["api_base_url"]
        
        # Generate sample data for demonstration
        current_time = datetime.now()
        
        # Simulate recent assessments
        sample_locations = [
            {"lat": 40.7128, "lon": -74.0060, "name": "New York"},
            {"lat": 51.5074, "lon": -0.1278, "name": "London"},
            {"lat": 35.6762, "lon": 139.6503, "name": "Tokyo"},
            {"lat": -33.8688, "lon": 151.2093, "name": "Sydney"},
            {"lat": 55.7558, "lon": 37.6176, "name": "Moscow"}
        ]
        
        new_assessments = []
        for i, loc in enumerate(sample_locations):
            risk_score = np.random.uniform(3.0, 8.0)
            assessment = {
                "id": f"assess_{current_time.timestamp()}_{i}",
                "timestamp": current_time - timedelta(minutes=np.random.randint(1, 60)),
                "latitude": loc["lat"],
                "longitude": loc["lon"],
                "location_name": loc["name"],
                "overall_risk_score": risk_score,
                "flood_risk": np.random.choice(["Low", "Medium", "High"]),
                "air_quality": np.random.randint(20, 150),
                "water_stress": np.random.uniform(0, 5),
                "population_density": np.random.uniform(100, 5000)
            }
            new_assessments.append(assessment)
        
        # Update global data store
        dashboard_data["assessments"].extend(new_assessments)
        
        # Keep only recent data
        cutoff_time = current_time - timedelta(hours=24)
        dashboard_data["assessments"] = [
            a for a in dashboard_data["assessments"] 
            if a["timestamp"] > cutoff_time
        ]
        
        # Generate system metrics
        dashboard_data["system_metrics"].append({
            "timestamp": current_time,
            "cpu_usage": np.random.uniform(20, 80),
            "memory_usage": np.random.uniform(30, 70),
            "api_response_time": np.random.uniform(100, 500),
            "active_connections": np.random.randint(5, 50)
        })
        
        # Generate validation logs
        dashboard_data["data_validation_logs"].append({
            "timestamp": current_time,
            "source": np.random.choice(["NASA_POWER", "OpenWeatherMap", "WRI_Aqueduct"]),
            "status": np.random.choice(["SUCCESS", "WARNING", "ERROR"], p=[0.8, 0.15, 0.05]),
            "message": "Data validation completed",
            "processing_time": np.random.uniform(0.5, 3.0)
        })
        
        dashboard_data["last_update"] = current_time
        
        logger.info(f"Dashboard data updated: {len(new_assessments)} new assessments")
        
    except Exception as e:
        logger.error(f"Failed to fetch dashboard data: {e}")

def calculate_dashboard_metrics() -> DashboardMetrics:
    """Calculate key metrics for dashboard"""
    
    assessments = dashboard_data["assessments"]
    
    if not assessments:
        return DashboardMetrics(
            total_assessments=0,
            avg_risk_score=0.0,
            high_risk_locations=0,
            data_freshness="No data",
            system_health="Unknown",
            validation_success_rate=0.0
        )
    
    total_assessments = len(assessments)
    avg_risk_score = np.mean([a["overall_risk_score"] for a in assessments])
    high_risk_locations = len([a for a in assessments if a["overall_risk_score"] > 7.0])
    
    # Data freshness
    if dashboard_data["last_update"]:
        minutes_ago = (datetime.now() - dashboard_data["last_update"]).total_seconds() / 60
        if minutes_ago < 5:
            data_freshness = "Just now"
        elif minutes_ago < 60:
            data_freshness = f"{int(minutes_ago)}m ago"
        else:
            data_freshness = f"{int(minutes_ago/60)}h ago"
    else:
        data_freshness = "Unknown"
    
    # System health (based on recent validation logs)
    recent_logs = [
        log for log in dashboard_data["data_validation_logs"]
        if (datetime.now() - log["timestamp"]).total_seconds() < 3600  # Last hour
    ]
    
    if recent_logs:
        success_rate = len([log for log in recent_logs if log["status"] == "SUCCESS"]) / len(recent_logs)
        system_health = "Healthy" if success_rate > 0.9 else "Warning"
    else:
        success_rate = 0.0
        system_health = "Unknown"
    
    return DashboardMetrics(
        total_assessments=total_assessments,
        avg_risk_score=avg_risk_score,
        high_risk_locations=high_risk_locations,
        data_freshness=data_freshness,
        system_health=system_health,
        validation_success_rate=success_rate
    )

def create_risk_map():
    """Create global risk assessment map"""
    
    assessments = dashboard_data["assessments"]
    if not assessments:
        return go.Figure().add_annotation(
            text="No assessment data available",
            x=0.5, y=0.5, showarrow=False
        )
    
    df = pd.DataFrame(assessments)
    
    fig = px.scatter_mapbox(
        df,
        lat="latitude",
        lon="longitude",
        color="overall_risk_score",
        size="population_density",
        hover_name="location_name",
        hover_data=["flood_risk", "air_quality", "water_stress"],
        color_continuous_scale="Viridis",
        range_color=[0, 10],
        mapbox_style="open-street-map",
        zoom=1,
        height=400
    )
    
    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        coloraxis_colorbar=dict(title="Risk Score")
    )
    
    return fig

def create_risk_trends_chart():
    """Create risk score trends over time"""
    
    assessments = dashboard_data["assessments"]
    if not assessments:
        return go.Figure().add_annotation(
            text="No trend data available",
            x=0.5, y=0.5, showarrow=False
        )
    
    df = pd.DataFrame(assessments)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Group by hour and calculate average risk
    hourly_avg = df.groupby(df["timestamp"].dt.floor("H"))["overall_risk_score"].mean().reset_index()
    
    fig = px.line(
        hourly_avg,
        x="timestamp",
        y="overall_risk_score",
        title="Average Risk Score Over Time",
        markers=True
    )
    
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Average Risk Score",
        height=400
    )
    
    return fig

def create_data_quality_chart():
    """Create data quality status chart"""
    
    logs = dashboard_data["data_validation_logs"]
    if not logs:
        return go.Figure().add_annotation(
            text="No validation data available",
            x=0.5, y=0.5, showarrow=False
        )
    
    # Count status by source
    df = pd.DataFrame(logs)
    status_counts = df.groupby(["source", "status"]).size().reset_index(name="count")
    
    fig = px.bar(
        status_counts,
        x="source",
        y="count",
        color="status",
        title="Data Validation Status by Source",
        color_discrete_map={
            "SUCCESS": "green",
            "WARNING": "orange", 
            "ERROR": "red"
        }
    )
    
    fig.update_layout(height=400)
    
    return fig

def create_model_performance_chart():
    """Create model performance metrics chart"""
    
    metrics = dashboard_data["system_metrics"]
    if not metrics:
        return go.Figure().add_annotation(
            text="No performance data available",
            x=0.5, y=0.5, showarrow=False
        )
    
    df = pd.DataFrame(metrics)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["api_response_time"],
        mode="lines+markers",
        name="API Response Time (ms)",
        yaxis="y"
    ))
    
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["cpu_usage"],
        mode="lines+markers",
        name="CPU Usage (%)",
        yaxis="y2"
    ))
    
    fig.update_layout(
        title="System Performance Metrics",
        xaxis_title="Time",
        yaxis=dict(title="Response Time (ms)", side="left"),
        yaxis2=dict(title="CPU Usage (%)", side="right", overlaying="y"),
        height=400
    )
    
    return fig

def create_recent_assessments_table():
    """Create table of recent assessments"""
    
    assessments = dashboard_data["assessments"]
    if not assessments:
        return html.P("No recent assessments available", className="text-muted")
    
    # Get most recent 10 assessments
    recent = sorted(assessments, key=lambda x: x["timestamp"], reverse=True)[:10]
    
    table_rows = []
    for assessment in recent:
        row = html.Tr([
            html.Td(assessment["timestamp"].strftime("%H:%M:%S")),
            html.Td(assessment["location_name"]),
            html.Td(f"{assessment['overall_risk_score']:.2f}"),
            html.Td(assessment["flood_risk"]),
            html.Td(f"{assessment['air_quality']}"),
            html.Td(f"{assessment['water_stress']:.1f}"),
            html.Td(
                html.Span("🔴", title="High Risk") if assessment["overall_risk_score"] > 7
                else html.Span("🟡", title="Medium Risk") if assessment["overall_risk_score"] > 4
                else html.Span("🟢", title="Low Risk")
            )
        ])
        table_rows.append(row)
    
    table = html.Table([
        html.Thead([
            html.Tr([
                html.Th("Time"),
                html.Th("Location"),
                html.Th("Risk Score"),
                html.Th("Flood Risk"),
                html.Th("Air Quality"),
                html.Th("Water Stress"),
                html.Th("Status")
            ])
        ]),
        html.Tbody(table_rows)
    ], className="table table-striped table-hover")
    
    return table

# Callback functions for Dash app
def register_callbacks(app):
    """Register all dashboard callbacks"""
    
    @app.callback(
        [Output("metrics-cards", "children"),
         Output("risk-map", "figure"),
         Output("risk-trends", "figure"),
         Output("data-quality-chart", "figure"),
         Output("model-performance", "figure"),
         Output("recent-assessments-table", "children"),
         Output("system-logs", "children")],
        [Input("interval-component", "n_intervals")]
    )
    def update_dashboard(n):
        """Update all dashboard components"""
        
        # Fetch fresh data
        fetch_dashboard_data()
        
        # Calculate metrics
        metrics = calculate_dashboard_metrics()
        
        # Create components
        metrics_cards = create_metrics_cards(metrics)
        risk_map = create_risk_map()
        risk_trends = create_risk_trends_chart()
        data_quality = create_data_quality_chart()
        model_performance = create_model_performance_chart()
        recent_table = create_recent_assessments_table()
        
        # Create system logs
        recent_logs = dashboard_data["data_validation_logs"][-10:]
        log_entries = []
        for log in reversed(recent_logs):
            status_icon = "✅" if log["status"] == "SUCCESS" else "⚠️" if log["status"] == "WARNING" else "❌"
            log_text = f"{log['timestamp'].strftime('%H:%M:%S')} {status_icon} [{log['source']}] {log['message']}"
            log_entries.append(html.Div(log_text))
        
        return metrics_cards, risk_map, risk_trends, data_quality, model_performance, recent_table, log_entries

def run_dashboard():
    """Run the dashboard application"""
    app = create_dashboard_app()
    register_callbacks(app)
    
    # Start background data fetching
    def background_data_fetcher():
        while True:
            try:
                fetch_dashboard_data()
                time.sleep(DASHBOARD_CONFIG["update_interval"])
            except Exception as e:
                logger.error(f"Background data fetch error: {e}")
                time.sleep(60)  # Wait longer on error
    
    # Start background thread
    data_thread = threading.Thread(target=background_data_fetcher, daemon=True)
    data_thread.start()
    
    logger.info("Starting Urban Planner AI Dashboard on http://localhost:8050")
    app.run(debug=False, host="0.0.0.0", port=8050)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_dashboard()