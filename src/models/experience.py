from pydantic import BaseModel, Field
from typing import Optional


class Experience(BaseModel):
    id: str
    name: str
    location: tuple[float, float]  # lat, lng
    neighborhood: str = ""
    duration_minutes: int = 60
    experience_type: str  # "active" | "passive" | "food" | "cultural" | "serendipity" | "scenic"
    best_time_of_day: str  # "early_morning" | "morning" | "afternoon" | "golden_hour" | "evening" | "night" | "any"
    energy_required: float = Field(5.0, ge=0, le=10)
    crowd_level: float = Field(5.0, ge=0, le=10)
    local_score: float = Field(5.0, ge=0, le=10)
    cost_euros: float = 0.0
    weather_dependent: bool = False
    physical_intensity: str = "low"  # "low" | "medium" | "high"
    romantic_score: float = Field(5.0, ge=0, le=10)
    family_friendly: bool = True
    description: str = ""
    why_this_matters: str = ""
    practical_tips: str = ""
    city: str = ""

    # Scoring result (set by scoring engine)
    persona_score: Optional[float] = None
    highlight_score: Optional[float] = None
    highlight_tier: Optional[str] = None  # "must_do" | "should_do" | "nice_to_have"


class ScheduledExperience(BaseModel):
    experience: Experience
    start_time: str  # "HH:MM"
    end_time: str
    travel_time_from_prev: int = 0  # minutes
    travel_mode: str = "walk"  # "walk" | "transit" | "bike"
    notes: str = ""
