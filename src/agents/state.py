from typing import TypedDict, Optional, Any
from src.models.persona import TravelerPersona
from src.models.experience import Experience
from src.models.itinerary import Itinerary, CritiqueResult


class BaseGraphState(TypedDict, total=False):
    session_id: str
    persona: Optional[TravelerPersona]
    itinerary: Optional[Itinerary]
    log: list[str]
    error: Optional[str]


class PlanningState(BaseGraphState, total=False):
    city_context: str
    experience_pool: list[Experience]
    optimized_days: list[list[Experience]]
    critiques: list[CritiqueResult]
    revision_count: int
    weather_data: dict[str, Any]


class ReplanningState(BaseGraphState, total=False):
    disruption: str
    completed_day_indices: list[int]
    disruption_analysis: dict[str, Any]
