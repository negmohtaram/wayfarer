from pydantic import BaseModel, Field
from typing import Literal, Optional
from .experience import ScheduledExperience
from .persona import TravelerPersona


class WeatherForecast(BaseModel):
    date: str
    condition: str  # "sunny" | "cloudy" | "rainy" | "partly_cloudy" | "unknown"
    temp_high: float
    temp_low: float
    rain_probability: float = 0.0
    available: bool = True  # False when weather fetch failed — do not fabricate


class DayPlan(BaseModel):
    day_number: int
    date: Optional[str] = None
    arc_label: str = ""  # "Orientation Day", "Peak Day", "Deep Dive Day", "Final Day"
    arc_description: str = ""
    experiences: list[ScheduledExperience] = Field(default_factory=list)
    total_cost: float = 0.0
    total_walking_km: float = 0.0
    weather: Optional[WeatherForecast] = None
    day_summary: str = ""


class CritiqueResult(BaseModel):
    critic_type: str
    score: float  # 0-10
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    verdict: Literal["approve", "revise", "reject_day", "error"]
    affected_days: list[int] = Field(default_factory=list)


class RegretItem(BaseModel):
    experience_name: str
    experience_id: str
    day_number: int
    regret_score: float  # 0-10
    tier: str  # "must_do" | "should_do" | "nice_to_have"
    reason: str


class Itinerary(BaseModel):
    session_id: str
    persona: TravelerPersona
    destination: str
    days: list[DayPlan] = Field(default_factory=list)
    critiques: list[CritiqueResult] = Field(default_factory=list)
    regret_rankings: list[RegretItem] = Field(default_factory=list)
    revision_count: int = 0
    total_cost: float = 0.0
    status: str = "planning"  # "planning" | "complete" | "replanning"
    generation_log: list[str] = Field(default_factory=list)
