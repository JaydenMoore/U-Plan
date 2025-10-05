from dotenv import load_dotenv
import google.generativeai as genai
import os
import asyncio
import logging

load_dotenv()
logger = logging.getLogger(__name__)


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    logger.info("GOOGLE_API_KEY detected; LLM summaries enabled.")
else:
    logger.info("GOOGLE_API_KEY not set; using rule-based summaries.")

def _build_summary_prompt(rainfall: float, temperature: float, flood_risk: str, heat_risk: str, 
                          air_quality_risk: str, aqi: int, pm2_5: float, overall_risk: float,
                          population_density: float = None, population_stats: dict = None) -> str:
    return f"""
You are an urban planning assistant. Create a concise, actionable planning summary (max 120 words).
Use clear, neutral language with a few relevant emojis similar to existing UI.

Inputs:
- Rainfall (mm/month avg): {rainfall}
- Temperature (°C avg): {temperature}
- Flood Risk: {flood_risk}
- Heat Risk: {heat_risk}
- Air Quality Risk: {air_quality_risk}
- AQI: {aqi}
- PM2.5 (μg/m³): {pm2_5:.1f}
- Overall Risk Score (0-10): {overall_risk}
- Population Density: {population_density}
- Population Statistics: {population_stats}

Output guidance:
List down in point form overall risk score, environmental risk (air quality, flood and heat risk) and if population density insight if data is provided. (with suggestions associated with the value)
Call out any "high" risks with clear mitigations
Bold "high" or "low"
Extended suggestions for residents (house buyer) and urban planner
Include appropriate emojis
Keep tone practical for city planning
"""

def _gemini_generate(prompt: str) -> str:
    model = genai.GenerativeModel('gemini-2.5-flash-lite')
    resp = model.generate_content(prompt)
    txt = getattr(resp, "text", None)
    return txt.strip() if txt else "Summary unavailable."

async def generate_summary(rainfall: float, temperature: float, flood_risk: str, heat_risk: str, 
                           air_quality_risk: str, aqi: int, pm2_5: float, overall_risk: float,
                           population_density: float = None, population_stats: dict = None) -> str:
    if GOOGLE_API_KEY:
        try:
            logger.info("Generating summary with Google LLM (Gemini).")
            prompt = _build_summary_prompt(rainfall, temperature, flood_risk, heat_risk, air_quality_risk, aqi, pm2_5, overall_risk, population_density, population_stats)
            logger.info(f"Prompt: {prompt}")
            text = await asyncio.to_thread(_gemini_generate, prompt)
            return text
        except Exception as e:
            logging.getLogger(__name__).warning(f"LLM summary failed, using rule-based. Error: {e}")
    logger.info("Generating summary with rule-based logic.")
    return generate_summary_rule_based(rainfall, temperature, flood_risk, heat_risk, air_quality_risk, aqi, pm2_5, overall_risk, population_density, population_stats)

def generate_summary_rule_based(rainfall: float, temperature: float, flood_risk: str, heat_risk: str, 
                    air_quality_risk: str, aqi: int, pm2_5: float, overall_risk: float,
                    population_density: float = None, population_stats: dict = None) -> str:
    """Generate a comprehensive summary for urban planning including population context"""
    
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
    
    # Population context
    if population_density is not None:
        if population_density > 1000:
            summary_parts.append(f"🏙️ VERY HIGH POPULATION DENSITY ({population_density:.1f} people/km²): Dense urban area requiring careful infrastructure planning and emergency services.")
        elif population_density > 500:
            summary_parts.append(f"🏘️ HIGH POPULATION DENSITY ({population_density:.1f} people/km²): Urban area with significant infrastructure demands.")
        elif population_density > 150:
            summary_parts.append(f"🏡 MEDIUM POPULATION DENSITY ({population_density:.1f} people/km²): Suburban area with moderate development pressure.")
        elif population_density > 50:
            summary_parts.append(f"🌲 LOW POPULATION DENSITY ({population_density:.1f} people/km²): Rural area with development potential.")
        elif population_density > 1:
            summary_parts.append(f"🌿 VERY LOW POPULATION DENSITY ({population_density:.1f} people/km²): Sparsely populated area.")
        else:
            summary_parts.append("🌿 UNINHABITED AREA: No permanent population detected.")
    
    # Population area statistics
    if population_stats and "error" not in population_stats:
        if population_stats.get("populated_pixels", 0) > 0:
            mean_density = population_stats.get("mean_density", 0)
            max_density = population_stats.get("max_density", 0)
            if max_density > mean_density * 2:
                summary_parts.append(f"📊 VARIABLE POPULATION DENSITY in area: peaks at {max_density:.1f} people/km² with clusters requiring targeted infrastructure.")
    
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