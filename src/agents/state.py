from typing import TypedDict, Optional, Any
from src.models.persona import TravelerPersona
from src.models.experience import Experience
from src.models.itinerary import Itinerary, CritiqueResult


class PlanningState(TypedDict, total=False):
    session_id: str
    persona: Optional[TravelerPersona]
    city_context: str
    experience_pool: list[Experience]
    scored_experiences: list[Experience]
    itinerary: Optional[Itinerary]
    critiques: list[CritiqueResult]
    revision_count: int
    log: list[str]
    error: Optional[str]
    weather_data: dict[str, Any]


class ReplanningState(TypedDict, total=False):
    session_id: str
    persona: Optional[TravelerPersona]
    itinerary: Optional[Itinerary]
    disruption: str
    completed_day_indices: list[int]
    log: list[str]
    error: Optional[str]
