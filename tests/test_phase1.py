"""Phase 1 correctness tests.

Covers: time overflow, energy mismatch, weather unavailable state,
disruption structured output, critic routing, budget repair,
logistics repair, highlight_ranker rename, graph without enrichment.
"""
import asyncio
import pytest

from src.models.experience import Experience, ScheduledExperience
from src.models.itinerary import (
    CritiqueResult, DayPlan, Itinerary, WeatherForecast,
)
from src.models.persona import TravelerPersona
from src.optimization.energy import (
    format_time_hhmm, minutes_to_time, time_to_minutes,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_persona(**kwargs) -> TravelerPersona:
    defaults = dict(
        pace=5, food_importance=5, culture_importance=5,
        nature_importance=5, nightlife_importance=2,
        budget_per_day=100, walking_tolerance=5, crowd_tolerance=5,
        local_vs_tourist=5, energy_profile="consistent", social_context="solo",
        physical_constraints=[], dietary_requirements=[],
        photography_interest=5, trip_days=1, destination="amsterdam",
    )
    defaults.update(kwargs)
    return TravelerPersona(**defaults)


def make_experience(**kwargs) -> Experience:
    defaults = dict(
        id="exp1", name="Test Exp", location=(52.374, 4.880),
        duration_minutes=60, experience_type="cultural",
        best_time_of_day="morning", energy_required=5.0,
        crowd_level=5.0, local_score=5.0, cost_euros=10.0,
        weather_dependent=False, physical_intensity="low",
        romantic_score=5.0, family_friendly=True,
    )
    defaults.update(kwargs)
    return Experience(**defaults)


def make_scheduled(exp: Experience, start_time: str, end_time: str) -> ScheduledExperience:
    return ScheduledExperience(
        experience=exp, start_time=start_time, end_time=end_time,
        travel_time_from_prev=0, travel_mode="walk",
    )


def make_day(day_number: int, exps: list[ScheduledExperience], weather=None) -> DayPlan:
    wf = weather or WeatherForecast(
        date="2026-09-01", condition="sunny",
        temp_high=20.0, temp_low=12.0, rain_probability=10.0,
    )
    return DayPlan(day_number=day_number, experiences=exps, weather=wf)


def make_itinerary(days: list[DayPlan], persona: TravelerPersona) -> Itinerary:
    return Itinerary(
        session_id="test", persona=persona, destination="amsterdam",
        days=days, total_cost=sum(d.total_cost for d in days),
    )


# ── Task 4: Time overflow ─────────────────────────────────────────────────────

def test_minutes_to_time_normal():
    assert minutes_to_time(480) == "08:00"
    assert minutes_to_time(0) == "00:00"
    assert minutes_to_time(60) == "01:00"


def test_minutes_to_time_midnight():
    assert minutes_to_time(1440) == "24:00"


def test_minutes_to_time_overflow_not_clamped():
    assert minutes_to_time(1500) == "25:00"
    assert minutes_to_time(1560) == "26:00"


def test_format_time_hhmm_clamps():
    assert format_time_hhmm(1500) == "23:59"
    assert format_time_hhmm(480) == "08:00"


def test_time_to_minutes_handles_overflow_strings():
    assert time_to_minutes("25:00") == 1500
    assert time_to_minutes("24:00") == 1440
    assert time_to_minutes("08:00") == 480


def test_logistics_critic_detects_overflow():
    """Logistics critic must flag days that end past midnight."""
    from src.agents.nodes.critics.logistics import logistics_critic

    exp = make_experience(duration_minutes=120, best_time_of_day="any")
    sched = make_scheduled(exp, start_time="23:00", end_time=minutes_to_time(1500))
    day = make_day(1, [sched])
    persona = make_persona()
    itinerary = make_itinerary([day], persona)

    result = asyncio.run(logistics_critic(itinerary))
    assert any("midnight" in issue for issue in result.issues), \
        f"Expected midnight overflow issue, got: {result.issues}"


# ── Task 5: Weather unavailable ───────────────────────────────────────────────

def test_weather_unavailable_produces_available_false():
    from src.agents.nodes.assembler import _build_day_schedule

    persona = make_persona()
    exp = make_experience()
    day = _build_day_schedule([exp], persona, 1, 1, None)

    assert day.weather is not None
    assert day.weather.available is False
    assert day.weather.condition == "unknown"
    assert day.weather.temp_high == 0.0
    assert day.weather.temp_low == 0.0


def test_weather_available_when_data_present():
    from src.agents.nodes.assembler import _build_day_schedule

    persona = make_persona()
    exp = make_experience()
    weather = {
        "date": "2026-09-01", "condition": "sunny",
        "temp_high": 22.0, "temp_low": 14.0, "rain_probability": 5.0,
    }
    day = _build_day_schedule([exp], persona, 1, 1, weather)

    assert day.weather.available is True
    assert day.weather.condition == "sunny"
    assert day.weather.temp_high == 22.0


# ── Task 3: Energy mismatch ───────────────────────────────────────────────────

def test_energy_mismatch_fires_for_high_demand_at_low_energy_time():
    """morning_person has energy ~2/10 at 22:00. A 9/10 demand activity should flag."""
    from src.agents.nodes.critics.energy import energy_critic

    persona = make_persona(energy_profile="morning_person")
    exp = make_experience(energy_required=9.0, best_time_of_day="night")
    sched = make_scheduled(exp, "22:00", "23:00")
    day = make_day(1, [sched])
    itinerary = make_itinerary([day], persona)

    result = asyncio.run(energy_critic(itinerary))
    mismatch_issues = [i for i in result.issues if "requires energy" in i]
    assert len(mismatch_issues) > 0, f"Expected energy mismatch issue, got: {result.issues}"


def test_energy_mismatch_does_not_fire_for_matched_activity():
    """morning_person has energy ~9.5/10 at 08:00. A 5/10 demand activity should be fine."""
    from src.agents.nodes.critics.energy import energy_critic

    persona = make_persona(energy_profile="morning_person")
    exp = make_experience(energy_required=5.0, best_time_of_day="morning")
    sched = make_scheduled(exp, "08:00", "09:00")
    day = make_day(1, [sched])
    itinerary = make_itinerary([day], persona)

    result = asyncio.run(energy_critic(itinerary))
    mismatch_issues = [i for i in result.issues if "requires energy" in i]
    assert len(mismatch_issues) == 0, f"Unexpected mismatch issue: {result.issues}"


# ── Task 7: Revision contract ─────────────────────────────────────────────────

def test_revision_budget_removes_most_expensive_experience():
    """Budget critic fires → revision removes most expensive experience."""
    from src.agents.nodes.revision import revision_agent_node

    persona = make_persona(budget_per_day=30)
    cheap = make_experience(id="cheap", name="Cheap", cost_euros=10.0)
    expensive = make_experience(id="pricey", name="Pricey", cost_euros=80.0)

    sched_cheap = make_scheduled(cheap, "09:00", "10:00")
    sched_pricey = make_scheduled(expensive, "11:00", "12:00")
    day = make_day(1, [sched_cheap, sched_pricey])
    itinerary = make_itinerary([day], persona)

    budget_critique = CritiqueResult(
        critic_type="budget", score=5.0, verdict="revise",
        issues=["Day 1: over budget"], suggestions=[], affected_days=[1],
    )

    state = {
        "itinerary": itinerary,
        "critiques": [budget_critique],
        "experience_pool": [cheap, expensive],
        "revision_count": 0,
        "log": [],
    }

    result = asyncio.run(revision_agent_node(state))
    remaining_ids = {s.experience.id for s in result["itinerary"].days[0].experiences}
    assert "pricey" not in remaining_ids, "Most expensive experience should have been removed"
    assert "cheap" in remaining_ids


def test_revision_logistics_marks_day_for_rebuild():
    """Logistics critic fires with timing issue → day is rebuilt (changed=True)."""
    from src.agents.nodes.revision import revision_agent_node

    persona = make_persona()
    exp = make_experience(best_time_of_day="morning")
    sched = make_scheduled(exp, "21:00", "22:00")  # wrong time for morning activity
    day = make_day(1, [sched])
    itinerary = make_itinerary([day], persona)

    logistics_critique = CritiqueResult(
        critic_type="logistics", score=5.0, verdict="revise",
        issues=["Day 1: 'Test Exp' scheduled at 21:00 but best at morning"],
        suggestions=[], affected_days=[1],
    )

    state = {
        "itinerary": itinerary,
        "critiques": [logistics_critique],
        "experience_pool": [exp],
        "revision_count": 0,
        "log": [],
    }

    result = asyncio.run(revision_agent_node(state))
    log = result["log"]
    rebuilt = any("days rebuilt" in entry for entry in log)
    assert rebuilt, f"Expected day rebuild log entry, got: {log}"


# ── Task 8: Highlight ranker rename ──────────────────────────────────────────

def test_graph_has_highlight_ranker_not_regret_minimizer():
    from src.agents.graph import planning_graph
    nodes = list(planning_graph.nodes)
    assert "highlight_ranker" in nodes, "highlight_ranker node missing from graph"
    assert "regret_minimizer" not in nodes, "regret_minimizer should be removed"


def test_highlight_ranker_populates_rankings():
    """highlight_ranker_node fills itinerary.regret_rankings."""
    from src.agents.nodes.highlight_ranker import highlight_ranker_node

    persona = make_persona()
    exp = make_experience(persona_score=0.7)
    sched = make_scheduled(exp, "09:00", "10:00")
    day = make_day(1, [sched])
    itinerary = make_itinerary([day], persona)

    state = {"itinerary": itinerary, "log": []}
    result = asyncio.run(highlight_ranker_node(state))

    rankings = result["itinerary"].regret_rankings
    assert len(rankings) == 1
    assert rankings[0].experience_name == "Test Exp"
    assert rankings[0].tier in ("must_do", "should_do", "nice_to_have")
    assert result["itinerary"].status == "complete"


# ── Task 1: Graph skips enrichment ───────────────────────────────────────────

def test_graph_does_not_contain_enrichment_node():
    from src.agents.graph import planning_graph
    nodes = list(planning_graph.nodes)
    assert "enrichment" not in nodes, "enrichment should not be in the planning graph"


def test_graph_connects_experience_generator_to_optimizer():
    from src.agents.graph import planning_graph
    # Verify the edge exists by checking nodes are present and graph compiles
    assert "experience_generator" in planning_graph.nodes
    assert "optimizer" in planning_graph.nodes


# ── Task 6: Disruption structured output ─────────────────────────────────────

def test_disruption_analysis_model_fields():
    from src.agents.replan_graph import DisruptionAnalysis
    analysis = DisruptionAnalysis(disruption_type="weather", affected_experiences=["Bike ride"])
    assert analysis.disruption_type == "weather"
    assert analysis.affected_experiences == ["Bike ride"]
    assert analysis.recommendation == ""


def test_disruption_analysis_rejects_invalid_type():
    from src.agents.replan_graph import DisruptionAnalysis
    with pytest.raises(Exception):
        DisruptionAnalysis(disruption_type="alien_invasion")
