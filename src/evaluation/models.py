from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from src.models.persona import TravelerPersona


class DisruptionSpec(BaseModel):
    message: str
    completed_day_indices: list[int]
    expected_stable_day_numbers: list[int] = []  # day_number (1-based) that must not change


class EvaluationScenario(BaseModel):
    id: str
    description: str
    tags: list[str] = []
    persona: TravelerPersona
    must_do_ids: list[str] = []
    must_avoid_ids: list[str] = []
    injected_weather: Optional[list[dict | None]] = None  # one entry per day; None = unavailable
    disruption: Optional[DisruptionSpec] = None
    # Codes we expect the baseline to fail — documents known planner limitations
    expected_violation_codes: list[str] = []


class ValidationResult(BaseModel):
    code: str
    passed: bool
    severity: Literal["error", "warning"]
    day: Optional[int] = None
    affected_experience_ids: list[str] = []
    observed: Any = None
    expected: Any = None
    message: str


class SoftMetrics(BaseModel):
    avg_persona_score: Optional[float]         # mean experience.persona_score; None if scores missing
    total_travel_time_minutes: int             # sum of all travel_time_from_prev
    total_walking_km: float                    # sum of day.total_walking_km
    energy_mismatch_count: int                 # experiences where demand - traveler_energy > threshold
    experience_type_diversity: int             # distinct experience_type values scheduled (max 6)
    localness_delta: Optional[float]           # |mean(local_score) − local_vs_tourist|; None if no exps
    budget_utilization_pct: float              # total_cost / (budget_per_day × trip_days) × 100
    avg_experiences_per_day: float


class OperationalMetrics(BaseModel):
    total_runtime_seconds: float
    node_runtimes: dict[str, float] = {}       # node_name → elapsed seconds
    llm_call_count: int                        # estimated: 1 per critic_panel pass
    revision_count: int
    estimated_cost_usd: Optional[float] = None
    errors: list[str] = []


class EvaluationResult(BaseModel):
    scenario_id: str
    planner_version: str
    run: int = 0                               # repeat index when a scenario runs several times
    run_at: str                                # ISO timestamp
    scenario: EvaluationScenario
    output_itinerary: Optional[dict] = None    # full itinerary dict; None if planning failed
    initial_itinerary: Optional[dict] = None   # for disruption scenarios: itinerary before replanning
    input_snapshot: dict = Field(default_factory=dict)  # persona + injected_weather for replay
    validation_results: list[ValidationResult] = []
    soft_metrics: Optional[SoftMetrics] = None
    operational_metrics: OperationalMetrics
    passed_all_hard_validators: bool
    planning_error: Optional[str] = None
    recorded_calls: Optional[int] = None       # model + weather calls recorded or replayed
    eval_error: Optional[str] = None           # set when a call failed or had no recording; run is invalid
