from src.models.constraints import ConstraintViolation, HardConstraints, SoftPreferences
from src.models.persona import TravelerPersona


def test_hard_view_holds_the_rules():
    p = TravelerPersona(budget_per_day=100, trip_days=3, destination="paris",
                        physical_constraints=["wheelchair"], dietary_requirements=["halal"])
    hard = p.hard
    assert (hard.destination, hard.trip_days, hard.budget_per_day) == ("paris", 3, 100)
    assert hard.mobility == ["wheelchair"]
    assert hard.dietary == ["halal"]


def test_budget_ceilings_match_the_old_hard_coded_numbers():
    hard = TravelerPersona(budget_per_day=100, trip_days=3).hard
    assert hard.day_budget_ceiling == 100 * 1.2
    assert hard.trip_budget_ceiling == 100 * 3 * 1.3


def test_unknown_mobility_value_is_kept_not_dropped():
    hard = TravelerPersona(physical_constraints=["bad_knee", "crutches"]).hard
    assert hard.mobility == ["bad_knee"]
    assert hard.unrecognised_mobility == ["crutches"]


def test_soft_view_holds_only_preferences():
    soft = TravelerPersona(pace=8, food_importance=9).soft
    assert isinstance(soft, SoftPreferences)
    assert (soft.pace, soft.food_importance) == (8, 9)
    assert not set(SoftPreferences.model_fields) & set(HardConstraints.model_fields)


def test_views_do_not_change_the_persona_json():
    p = TravelerPersona()
    assert "hard" not in p.model_dump() and "soft" not in p.model_dump()


def test_violation_carries_what_repair_needs():
    v = ConstraintViolation(code="DAY_OVER_BUDGET", day=2, affected_ids=["ams_rijks"],
                            observed=180.0, required=120.0, message="Day 2 over budget",
                            repair_hint="swap one paid stop for a free one")
    assert v.severity == "error" and v.day == 2


def test_hard_view_is_read_only():
    import pytest
    from pydantic import ValidationError
    hard = TravelerPersona(budget_per_day=100).hard
    with pytest.raises(ValidationError):
        hard.budget_per_day = 500


def test_critic_and_validator_use_the_same_day_ceiling():
    import asyncio
    from src.agents.nodes.critics.budget import budget_critic
    from src.evaluation.validators import validate_daily_budget
    from src.models.itinerary import DayPlan, Itinerary
    persona = TravelerPersona(budget_per_day=100, trip_days=1)
    day = DayPlan(day_number=1, date="2026-10-19", experiences=[], total_cost=121.0)
    itinerary = Itinerary(session_id="t", persona=persona, days=[day], destination="amsterdam", total_cost=121.0)
    critic = asyncio.run(budget_critic(itinerary))
    scenario = type("S", (), {"persona": persona})()
    validator = validate_daily_budget(itinerary, scenario)
    assert critic.affected_days == [1]
    assert validator[0].passed is False
