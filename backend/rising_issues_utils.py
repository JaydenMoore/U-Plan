import math
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

# Configure logger for this module
logger = logging.getLogger(__name__)



# --- Pydantic Model for a Rising Issue ---

class RisingIssue(BaseModel):
    id: str
    issue_type: str
    count: int
    avg_latitude: float
    avg_longitude: float
    overall_risk_score_avg: Optional[float]
    last_reported_at: datetime
    recent_reports: List[Dict[str, Any]]
    call_to_action_link: Optional[str] = None
    partner_ngo_name: Optional[str] = None

# --- Helper Functions for Grouping Logic ---

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the distance in kilometers between two geographic points."""
    R = 6371  # Radius of Earth in kilometers
    
    # Handle potential None values
    if any(v is None for v in [lat1, lon1, lat2, lon2]):
        return float('inf')

    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance

def infer_issue_type(feedback_entry: Dict[str, Any]) -> str:
    """Analyzes a single feedback entry to determine its primary issue type."""
    # 1. Prioritize metadata
    metadata = feedback_entry.get("metadata")
    if metadata and isinstance(metadata, dict):
        issue_category = metadata.get("issue_category")
        if issue_category and isinstance(issue_category, str):
            valid_categories = {"Flooding", "Heat Island", "Air Quality", "Infrastructure Damage", "Noise Pollution", "Biodiversity Loss", "Waste Management", "General Observation"}
            if issue_category in valid_categories:
                return issue_category

    # 2. Keyword Inference (Fallback)
    feedback_text = (feedback_entry.get("feedback") or "").lower()
    if not feedback_text:
        return "General Observation"

    keyword_map = {
        "Flooding": ["flood", "waterlogging", "drainage", "overflow", "wet", "puddle", "rising water"],
        "Heat Island": ["heat", "hot", "temperatures", "sweltering", "scorching", "warm", "sun"],
        "Air Quality": ["smog", "air quality", "pollution", "haze", "particulate", "dust", "smoke"],
        "Infrastructure Damage": ["collapse", "damaged", "unsafe", "sinkhole", "crack", "pothole", "broken"],
        "Noise Pollution": ["noise", "loud", "sound", "disturbance"],
        "Biodiversity Loss": ["wildlife", "animal", "plant", "trees", "habitat", "nature"],
        "Waste Management": ["trash", "garbage", "waste", "litter"],
    }

    for issue_type, keywords in keyword_map.items():
        if any(keyword in feedback_text for keyword in keywords):
            return issue_type

    return "General Observation"

def group_feedback_by_proximity_and_type(feedback_entries: List[Dict], radius_km: float, min_reports: int) -> List[RisingIssue]:
    """Groups feedback entries by geographic proximity and inferred issue type."""
    issue_groups: List[RisingIssue] = []
    processed_feedback_ids = set()

    # Sort entries by creation time to process older ones first
    feedback_entries.sort(key=lambda x: x.get('created_at', ''))

    for i, entry1 in enumerate(feedback_entries):
        if entry1.get('id') in processed_feedback_ids:
            continue

        if 'id' not in entry1:
            logger.warning(f"Skipping feedback entry due to missing 'id': {entry1}")
            continue

        issue_type1 = infer_issue_type(entry1)
        group_members = [entry1]
        processed_feedback_ids.add(entry1['id'])

        for j in range(i + 1, len(feedback_entries)):
            entry2 = feedback_entries[j]
            if entry2.get('id') in processed_feedback_ids:
                continue

            if 'id' not in entry2:
                logger.warning(f"Skipping feedback sub-entry due to missing 'id': {entry2}")
                continue

            issue_type2 = infer_issue_type(entry2)
            
            dist = haversine_distance(
                entry1.get('latitude'), entry1.get('longitude'),
                entry2.get('latitude'), entry2.get('longitude')
            )

            if issue_type1 == issue_type2 and dist <= radius_km:
                group_members.append(entry2)
                processed_feedback_ids.add(entry2['id'])
        
        if len(group_members) >= min_reports:
            # Calculate group aggregates
            avg_latitude = sum(m['latitude'] for m in group_members) / len(group_members)
            avg_longitude = sum(m['longitude'] for m in group_members) / len(group_members)
            
            scores = [m.get('overall_risk_score') for m in group_members if m.get('overall_risk_score') is not None]
            overall_risk_score_avg = round(sum(scores) / len(scores), 2) if scores else None

            # Sort by created_at to find the most recent
            group_members.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            last_reported_at = datetime.fromisoformat(group_members[0]['created_at'].replace('Z', '+00:00'))
            
            recent_reports = [
                {
                    "id": m['id'],
                    "feedback": m.get('feedback'),
                    "created_at": m.get('created_at')
                } for m in group_members[:3]
            ]

            # Placeholder for NGO links
            call_to_action_link = None
            partner_ngo_name = None
            if issue_type1 == "Flooding":
                call_to_action_link = "https://www.redcross.org/donate/disaster-relief.html"
                partner_ngo_name = "Red Cross Disaster Relief"
            elif issue_type1 == "Air Quality":
                call_to_action_link = "https://www.lung.org/donate"
                partner_ngo_name = "American Lung Association"

            rising_issue = RisingIssue(
                id=f"{issue_type1.replace(' ', '_').lower()}-{avg_latitude:.4f}-{avg_longitude:.4f}",
                issue_type=issue_type1,
                count=len(group_members),
                avg_latitude=avg_latitude,
                avg_longitude=avg_longitude,
                overall_risk_score_avg=overall_risk_score_avg,
                last_reported_at=last_reported_at,
                recent_reports=recent_reports,
                call_to_action_link=call_to_action_link,
                partner_ngo_name=partner_ngo_name,
            )
            issue_groups.append(rising_issue)
            
    logger.info(f"Found {len(issue_groups)} rising issue clusters from {len(feedback_entries)} feedback entries.")
    return issue_groups