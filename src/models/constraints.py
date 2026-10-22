from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, Union, get_args

from pydantic import BaseModel, ConfigDict, Field

from src.models.experience import Experience

if TYPE_CHECKING:
    from src.models.persona import TravelerPersona


Mobility = Literal["wheelchair", "bad_knee", "elderly"]
KNOWN_MOBILITY: tuple[str, ...] = get_args(Mobility)


class HardConstraints(BaseModel):
    """Rules a valid plan must satisfy. A plan that breaks one is rejected, never traded for points."""

    model_config = ConfigDict(frozen=True)

    destination: str
    trip_days: int
    budget_per_day: int
    day_budget_tolerance: float = 1.2
    trip_budget_tolerance: float = 1.3
    mobility: list[Mobility] = Field(default_factory=list)
    unrecognised_mobility: list[str] = Field(
        default_factory=list, description="values the persona builder wrote that no rule checks yet"
    )
    dietary: list[str] = Field(default_factory=list)

    @property
    def day_budget_ceiling(self) -> float:
        return self.budget_per_day * self.day_budget_tolerance

    @property
    def trip_budget_ceiling(self) -> float:
        return self.budget_per_day * self.trip_days * self.trip_budget_tolerance


class SoftPreferences(BaseModel):
    """What the traveler would like. Ranks plans that already pass every hard constraint."""

    pace: float
    food_importance: float
    nature_importance: float
    culture_importance: float
    nightlife_importance: float
    walking_tolerance: float
    crowd_tolerance: float
    local_vs_tourist: float
    energy_profile: str
    photography_interest: float


class ConstraintViolation(BaseModel):
    """One broken hard constraint, with enough detail to explain it and to repair it."""

    code: str
    severity: Literal["error", "warning"] = "error"
    day: Optional[int] = None
    affected_ids: list[str] = Field(default_factory=list)
    observed: Union[float, str, None] = None
    required: Union[float, str, None] = None
    message: str
    repair_hint: Optional[str] = None


def split_mobility(values: list[str]) -> tuple[list[str], list[str]]:
    known = [v for v in values if v in KNOWN_MOBILITY]
    unknown = [v for v in values if v not in KNOWN_MOBILITY]
    return known, unknown


def wheelchair_ok(experience: Experience, persona: TravelerPersona) -> bool:
    """One wheelchair rule for the planner, the replanner and the validator."""
    return "wheelchair" not in persona.hard.mobility or experience.physical_intensity == "low"


def check_experience(experience: Experience, persona: TravelerPersona) -> list[ConstraintViolation]:
    """Hard rules one stop must pass before it may be scored. Empty list = feasible."""
    hard = persona.hard
    found = []
    if not wheelchair_ok(experience, persona):
        found.append(ConstraintViolation(
            code="ACCESSIBILITY", affected_ids=[experience.id],
            observed=experience.physical_intensity, required="low",
            message=f"{experience.name} is not wheelchair accessible",
        ))
    if experience.cost_euros > hard.day_budget_ceiling:
        found.append(ConstraintViolation(
            code="OVER_DAY_BUDGET", affected_ids=[experience.id],
            observed=experience.cost_euros, required=hard.day_budget_ceiling,
            message=f"{experience.name} alone costs more than a whole day's budget",
        ))
    return found
