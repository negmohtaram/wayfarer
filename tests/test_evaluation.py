"""
Unit tests for the evaluation framework.

Tests only pure functions (validators, metrics) using crafted fixtures.
Does NOT run the planning graph — that would require LLM calls.
"""

import pytest

from src.evaluation.metrics import compute_soft_metrics
from src.evaluation.models import DisruptionSpec, EvaluationScenario, ValidationResult
from src.evaluation.validators import (
    passed_all_hard,
    validate_accessibility,
    validate_completed_days_preserved,
    validate_daily_budget,
    validate_day_count,
    validate_must_avoid,
    validate_must_do,
    validate_no_day_overflow,
    validate_no_duplicate_experiences,
    validate_no_impossible_values,
    validate_no_schedule_overlap,
    validate_trip_budget,
)
from src.models.experience import Experience, ScheduledExperience
from src.models.itinerary import DayPlan, Itinerary, WeatherForecast
from src.models.persona import TravelerPersona


# ── Fixtures ───────────────────────────────────────────────────────────────────

def make_persona(**kw) -> TravelerPersona:
    defaults = dict(
        pace=5, budget_per_day=100, food_importance=5, nature_importance=5,
        culture_importance=5, nightlife_importance=3, walking_tolerance=5,
        crowd_tolerance=5, local_vs_tourist=5, energy_profile="consistent",
        social_context="solo", physical_constraints=[], dietary_requirements=[],
        photography_interest=5, trip_days=3, destination="amsterdam",
    )
    defaults.update(kw)
    return TravelerPersona(**defaults)


def make_experience(**kw) -> Experience:
    defaults = dict(
        id="exp1", name="Test Exp", location=(52.374, 4.880),
        duration_minutes=60, experience_type="cultural",
        best_time_of_day="morning", energy_required=5.0,
        crowd_level=5.0, local_score=5.0, cost_euros=20.0,
        weather_dependent=False, physical_intensity="low",
        romantic_score=5.0, family_friendly=True, persona_score=0.7,
    )
    defaults.update(kw)
    return Experience(**defaults)


def make_scheduled(exp: Experience, start: str, end: str, travel: int = 0) -> ScheduledExperience:
    return ScheduledExperience(
        experience=exp, start_time=start, end_time=end,
        travel_time_from_prev=travel, travel_mode="walk",
    )


def make_day(number: int, scheduled: list[ScheduledExperience], cost: float = 40.0) -> DayPlan:
    return DayPlan(
        day_number=number,
        experiences=scheduled,
        total_cost=cost,
        total_walking_km=1.0,
        weather=WeatherForecast(
            date="2026-09-10", condition="sunny",
            temp_high=20.0, temp_low=12.0, rain_probability=5.0,
        ),
    )


def make_itinerary(days: list[DayPlan], persona: TravelerPersona) -> Itinerary:
    return Itinerary(
        session_id="test",
        persona=persona,
        destination=persona.destination,
        days=days,
        total_cost=sum(d.total_cost for d in days),
        status="complete",
    )


def make_scenario(**kw) -> EvaluationScenario:
    defaults = dict(id="test_scenario", description="test", persona=make_persona())
    defaults.update(kw)
    return EvaluationScenario(**defaults)


# ── validate_day_count ─────────────────────────────────────────────────────────

def test_day_count_matches():
    persona = make_persona(trip_days=2)
    itin = make_itinerary([make_day(1, []), make_day(2, [])], persona)
    results = validate_day_count(itin, make_scenario(persona=persona))
    assert all(r.passed for r in results)


def test_day_count_mismatch():
    persona = make_persona(trip_days=3)
    itin = make_itinerary([make_day(1, [])], persona)  # only 1 day
    results = validate_day_count(itin, make_scenario(persona=persona))
    assert any(not r.passed and r.code == "DAY_COUNT_MATCHES" for r in results)


# ── validate_no_duplicate_experiences ────────────────────────────────────────

def test_no_duplicates_clean():
    exp1 = make_experience(id="exp1")
    exp2 = make_experience(id="exp2")
    itin = make_itinerary([
        make_day(1, [make_scheduled(exp1, "09:00", "10:00")]),
        make_day(2, [make_scheduled(exp2, "09:00", "10:00")]),
    ], make_persona())
    results = validate_no_duplicate_experiences(itin, make_scenario())
    assert all(r.passed for r in results)


def test_duplicates_detected():
    exp = make_experience(id="same_id")
    itin = make_itinerary([
        make_day(1, [make_scheduled(exp, "09:00", "10:00")]),
        make_day(2, [make_scheduled(exp, "09:00", "10:00")]),
    ], make_persona())
    results = validate_no_duplicate_experiences(itin, make_scenario())
    assert any(not r.passed and r.code == "NO_DUPLICATE_EXPERIENCES" for r in results)


# ── validate_no_schedule_overlap ──────────────────────────────────────────────

def test_no_overlap_sequential():
    exp1 = make_experience(id="a")
    exp2 = make_experience(id="b")
    sched = [
        make_scheduled(exp1, "09:00", "10:00"),
        make_scheduled(exp2, "10:00", "11:00"),
    ]
    itin = make_itinerary([make_day(1, sched)], make_persona())
    results = validate_no_schedule_overlap(itin, make_scenario())
    assert all(r.passed for r in results)


def test_overlap_detected():
    exp1 = make_experience(id="a")
    exp2 = make_experience(id="b")
    sched = [
        make_scheduled(exp1, "09:00", "10:30"),  # ends 10:30
        make_scheduled(exp2, "10:00", "11:00"),  # starts 10:00 — overlaps
    ]
    itin = make_itinerary([make_day(1, sched)], make_persona())
    results = validate_no_schedule_overlap(itin, make_scenario())
    assert any(not r.passed and r.code == "NO_SCHEDULE_OVERLAP" for r in results)


# ── validate_no_day_overflow ──────────────────────────────────────────────────

def test_no_overflow_normal():
    exp = make_experience()
    itin = make_itinerary([make_day(1, [make_scheduled(exp, "21:00", "22:00")])], make_persona())
    results = validate_no_day_overflow(itin, make_scenario())
    assert all(r.passed for r in results)


def test_overflow_detected():
    exp = make_experience()
    itin = make_itinerary([make_day(1, [make_scheduled(exp, "23:00", "25:00")])], make_persona())
    results = validate_no_day_overflow(itin, make_scenario())
    assert any(not r.passed and r.code == "NO_DAY_OVERFLOW" for r in results)


# ── validate_daily_budget ─────────────────────────────────────────────────────

def test_daily_budget_ok():
    persona = make_persona(budget_per_day=100)  # ceiling = 120
    itin = make_itinerary([make_day(1, [], cost=100)], persona)
    results = validate_daily_budget(itin, make_scenario(persona=persona))
    assert all(r.passed for r in results)


def test_daily_budget_exceeded():
    persona = make_persona(budget_per_day=100)  # ceiling = 120
    itin = make_itinerary([make_day(1, [], cost=150)], persona)
    results = validate_daily_budget(itin, make_scenario(persona=persona))
    assert any(not r.passed and r.code == "DAILY_BUDGET_RESPECTED" for r in results)


# ── validate_trip_budget ──────────────────────────────────────────────────────

def test_trip_budget_ok():
    persona = make_persona(budget_per_day=100, trip_days=3)  # ceiling = 390
    days = [make_day(i, [], cost=100) for i in range(1, 4)]
    itin = make_itinerary(days, persona)
    itin.total_cost = 300
    results = validate_trip_budget(itin, make_scenario(persona=persona))
    assert all(r.passed for r in results)


def test_trip_budget_exceeded():
    persona = make_persona(budget_per_day=100, trip_days=3)  # ceiling = 390
    days = [make_day(i, [], cost=150) for i in range(1, 4)]
    itin = make_itinerary(days, persona)
    itin.total_cost = 450
    results = validate_trip_budget(itin, make_scenario(persona=persona))
    assert any(not r.passed and r.code == "TRIP_BUDGET_RESPECTED" for r in results)


# ── validate_accessibility ────────────────────────────────────────────────────

def test_accessibility_wheelchair_high_intensity_fails():
    persona = make_persona(physical_constraints=["wheelchair"])
    exp = make_experience(physical_intensity="high")
    itin = make_itinerary([make_day(1, [make_scheduled(exp, "09:00", "10:00")])], persona)
    scenario = make_scenario(persona=persona)
    results = validate_accessibility(itin, scenario)
    assert any(not r.passed and r.code == "ACCESSIBILITY_RESPECTED" for r in results)


def test_accessibility_wheelchair_medium_intensity_fails():
    persona = make_persona(physical_constraints=["wheelchair"])
    exp = make_experience(physical_intensity="medium")
    itin = make_itinerary([make_day(1, [make_scheduled(exp, "09:00", "10:00")])], persona)
    scenario = make_scenario(persona=persona)
    results = validate_accessibility(itin, scenario)
    assert any(not r.passed and r.code == "ACCESSIBILITY_RESPECTED" for r in results)

def test_accessibility_wheelchair_low_intensity_ok():
    persona = make_persona(physical_constraints=["wheelchair"])
    exp = make_experience(physical_intensity="low")
    itin = make_itinerary([make_day(1, [make_scheduled(exp, "09:00", "10:00")])], persona)
    scenario = make_scenario(persona=persona)
    results = validate_accessibility(itin, scenario)
    assert all(r.passed for r in results)


def test_accessibility_no_constraints():
    persona = make_persona(physical_constraints=[])
    exp = make_experience(physical_intensity="high")
    itin = make_itinerary([make_day(1, [make_scheduled(exp, "09:00", "10:00")])], persona)
    results = validate_accessibility(itin, make_scenario(persona=persona))
    assert all(r.passed for r in results)


# ── validate_no_impossible_values ─────────────────────────────────────────────

def test_impossible_values_clean():
    exp = make_experience(duration_minutes=60, cost_euros=10.0)
    sched = make_scheduled(exp, "09:00", "10:00", travel=15)
    itin = make_itinerary([make_day(1, [sched], cost=10.0)], make_persona())
    results = validate_no_impossible_values(itin, make_scenario())
    assert all(r.passed for r in results)


def test_negative_cost_detected():
    exp = make_experience(cost_euros=-5.0)
    sched = make_scheduled(exp, "09:00", "10:00")
    itin = make_itinerary([make_day(1, [sched], cost=-5.0)], make_persona())
    results = validate_no_impossible_values(itin, make_scenario())
    assert any(not r.passed and r.code == "NO_IMPOSSIBLE_VALUES" for r in results)


# ── validate_must_do / must_avoid ─────────────────────────────────────────────

def test_must_do_satisfied():
    exp = make_experience(id="ams_001")
    itin = make_itinerary([make_day(1, [make_scheduled(exp, "07:00", "08:15")])], make_persona())
    scenario = make_scenario(must_do_ids=["ams_001"])
    results = validate_must_do(itin, scenario)
    assert all(r.passed for r in results)


def test_must_do_not_satisfied():
    exp = make_experience(id="ams_002")
    itin = make_itinerary([make_day(1, [make_scheduled(exp, "09:00", "10:00")])], make_persona())
    scenario = make_scenario(must_do_ids=["ams_001"])
    results = validate_must_do(itin, scenario)
    assert any(not r.passed and r.code == "MUST_DO_SATISFIED" for r in results)


def test_must_do_empty_ids_skipped():
    itin = make_itinerary([make_day(1, [])], make_persona())
    results = validate_must_do(itin, make_scenario(must_do_ids=[]))
    assert results == []


def test_must_avoid_respected():
    exp = make_experience(id="ams_002")
    itin = make_itinerary([make_day(1, [make_scheduled(exp, "09:00", "10:00")])], make_persona())
    scenario = make_scenario(must_avoid_ids=["ams_001"])  # different ID
    results = validate_must_avoid(itin, scenario)
    assert all(r.passed for r in results)


def test_must_avoid_violated():
    exp = make_experience(id="forbidden_exp")
    itin = make_itinerary([make_day(1, [make_scheduled(exp, "09:00", "10:00")])], make_persona())
    scenario = make_scenario(must_avoid_ids=["forbidden_exp"])
    results = validate_must_avoid(itin, scenario)
    assert any(not r.passed and r.code == "MUST_AVOID_RESPECTED" for r in results)


# ── validate_completed_days_preserved ────────────────────────────────────────

def test_completed_days_preserved_when_unchanged():
    exp = make_experience(id="original_exp")
    day1 = make_day(1, [make_scheduled(exp, "09:00", "10:00")])
    itin_before = make_itinerary([day1], make_persona(trip_days=1))
    itin_after = make_itinerary([day1], make_persona(trip_days=1))

    scenario = make_scenario(
        disruption=DisruptionSpec(
            message="test disruption",
            completed_day_indices=[0],
            expected_stable_day_numbers=[1],
        )
    )
    results = validate_completed_days_preserved(itin_after, scenario, itin_before)
    assert all(r.passed for r in results)


def test_completed_days_changed_detected():
    exp_before = make_experience(id="exp_before")
    exp_after = make_experience(id="exp_after")
    day_before = make_day(1, [make_scheduled(exp_before, "09:00", "10:00")])
    day_after = make_day(1, [make_scheduled(exp_after, "09:00", "10:00")])
    itin_before = make_itinerary([day_before], make_persona(trip_days=1))
    itin_after = make_itinerary([day_after], make_persona(trip_days=1))

    scenario = make_scenario(
        disruption=DisruptionSpec(
            message="test",
            completed_day_indices=[0],
            expected_stable_day_numbers=[1],
        )
    )
    results = validate_completed_days_preserved(itin_after, scenario, itin_before)
    assert any(not r.passed and r.code == "COMPLETED_DAYS_PRESERVED" for r in results)


# ── passed_all_hard ───────────────────────────────────────────────────────────

def test_passed_all_hard_true_when_only_warnings():
    results = [
        ValidationResult(code="X", passed=False, severity="warning", message="warning only"),
        ValidationResult(code="Y", passed=True, severity="error", message="ok"),
    ]
    assert passed_all_hard(results) is True


def test_passed_all_hard_false_when_error_fails():
    results = [
        ValidationResult(code="X", passed=False, severity="error", message="hard fail"),
    ]
    assert passed_all_hard(results) is False


# ── compute_soft_metrics ──────────────────────────────────────────────────────

def test_soft_metrics_empty_itinerary():
    persona = make_persona(trip_days=3, budget_per_day=100)
    days = [make_day(i, [], cost=0.0) for i in range(1, 4)]
    itin = make_itinerary(days, persona)
    itin.total_cost = 0.0
    metrics = compute_soft_metrics(itin, make_scenario(persona=persona))
    assert metrics.avg_persona_score is None
    assert metrics.total_travel_time_minutes == 0
    assert metrics.experience_type_diversity == 0


def test_soft_metrics_computed():
    persona = make_persona(trip_days=1, budget_per_day=100, local_vs_tourist=5)
    exp = make_experience(
        id="e1", persona_score=0.8, local_score=7.0,
        experience_type="cultural", energy_required=5.0,
    )
    sched = make_scheduled(exp, "09:00", "10:00", travel=10)
    day = make_day(1, [sched], cost=20.0)
    itin = make_itinerary([day], persona)
    itin.total_cost = 20.0

    scenario = make_scenario(persona=persona)
    metrics = compute_soft_metrics(itin, scenario)

    assert metrics.avg_persona_score == pytest.approx(0.8, abs=0.001)
    assert metrics.total_travel_time_minutes == 10
    assert metrics.experience_type_diversity == 1
    assert metrics.budget_utilization_pct == pytest.approx(20.0, abs=0.1)
    assert metrics.localness_delta == pytest.approx(abs(7.0 - 5.0), abs=0.01)


# ── Scenario model validation ─────────────────────────────────────────────────

def test_scenario_must_do_and_avoid_coexist():
    scenario = EvaluationScenario(
        id="test",
        description="test",
        persona=make_persona(),
        must_do_ids=["a", "b"],
        must_avoid_ids=["c"],
        expected_violation_codes=["MUST_DO_SATISFIED"],
    )
    assert scenario.must_do_ids == ["a", "b"]
    assert scenario.expected_violation_codes == ["MUST_DO_SATISFIED"]


def test_all_baseline_scenarios_are_valid():
    from src.evaluation.scenarios.baseline_scenarios import ALL_SCENARIOS
    assert len(ALL_SCENARIOS) >= 50
    ids = [s.id for s in ALL_SCENARIOS]
    assert len(ids) == len(set(ids)), "Duplicate scenario IDs found"
