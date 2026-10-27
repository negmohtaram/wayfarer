from typing import TypedDict, Optional, Any
from src.models.persona import TravelerPersona
from src.models.experience import Experience
from src.models.constraints import ConstraintViolation
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
    feasible: list[Experience]              # every stop that passed the hard rules, scored, before the variety cut
    planner: str                            # "baseline" | "v2a"; overrides config.PLANNER for this run
    planner_trace: dict[str, Any]
    rejected: list[ConstraintViolation]     # stops removed by hard constraints, with reasons
    optimized_days: list[list[Experience]]
    critiques: list[CritiqueResult]
    revision_count: int
    weather_data: dict[str, Any]


class ReplanningState(BaseGraphState, total=False):
    disruption: str
    completed_day_indices: list[int]
    disruption_analysis: dict[str, Any]
