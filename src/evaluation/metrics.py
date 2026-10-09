"""
Soft metric calculations for Wayfarer itinerary evaluation.

All metrics here are interpretable observations — not claims about quality.
avg_persona_score measures self-consistency (the planner selected by this formula).
It does not claim the itinerary is "good"; it claims it is internally consistent
with the planner's own scoring function.
"""

from src.evaluation.models import EvaluationScenario, SoftMetrics
from src.models.itinerary import Itinerary
from src.optimization.energy import build_energy_curve, get_energy_at_time

ENERGY_MISMATCH_THRESHOLD = 3.0  # mirrors src/optimization/energy.py convention


def compute_soft_metrics(itinerary: Itinerary, scenario: EvaluationScenario) -> SoftMetrics:
    persona = scenario.persona
    all_scheduled = [se for day in itinerary.days for se in day.experiences]

    if not all_scheduled:
        return SoftMetrics(
            avg_persona_score=None,
            total_travel_time_minutes=0,
            total_walking_km=0.0,
            energy_mismatch_count=0,
            experience_type_diversity=0,
            localness_delta=None,
            budget_utilization_pct=0.0,
            avg_experiences_per_day=0.0,
        )

    # avg_persona_score — mean of persona_score set by experience_generator
    scores = [se.experience.persona_score for se in all_scheduled if se.experience.persona_score is not None]
    avg_persona_score = sum(scores) / len(scores) if scores else None

    # total_travel_time_minutes
    total_travel = sum(se.travel_time_from_prev for se in all_scheduled)

    # total_walking_km
    total_walking = sum(day.total_walking_km for day in itinerary.days)

    # energy_mismatch_count
    curve = build_energy_curve(persona.energy_profile)
    mismatch_count = 0
    for se in all_scheduled:
        traveler_energy = get_energy_at_time(se.start_time, curve)
        if se.experience.energy_required - traveler_energy > ENERGY_MISMATCH_THRESHOLD:
            mismatch_count += 1

    # experience_type_diversity (count of distinct types)
    types = {se.experience.experience_type for se in all_scheduled}
    diversity = len(types)

    # localness_delta — |mean(local_score) - local_vs_tourist|
    local_scores = [se.experience.local_score for se in all_scheduled]
    if local_scores:
        mean_local = sum(local_scores) / len(local_scores)
        localness_delta = abs(mean_local - persona.local_vs_tourist)
    else:
        localness_delta = None

    # budget_utilization_pct
    budget_cap = persona.budget_per_day * persona.trip_days
    utilization = (itinerary.total_cost / budget_cap * 100) if budget_cap > 0 else 0.0

    # avg_experiences_per_day
    avg_per_day = len(all_scheduled) / persona.trip_days if persona.trip_days > 0 else 0.0

    return SoftMetrics(
        avg_persona_score=round(avg_persona_score, 3) if avg_persona_score is not None else None,
        total_travel_time_minutes=total_travel,
        total_walking_km=round(total_walking, 2),
        energy_mismatch_count=mismatch_count,
        experience_type_diversity=diversity,
        localness_delta=round(localness_delta, 2) if localness_delta is not None else None,
        budget_utilization_pct=round(utilization, 1),
        avg_experiences_per_day=round(avg_per_day, 2),
    )
