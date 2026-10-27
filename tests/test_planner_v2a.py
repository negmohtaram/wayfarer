"""Planner v2a on every evaluation scenario: hard rules hold, the search ends at a local optimum."""
import pytest

from src.agents.nodes.experience_generator import score_experience
from src.agents.nodes.optimizer import _experiences_per_day
from src.evaluation.scenarios.baseline_scenarios import ALL_SCENARIOS, SCENARIO_BY_ID
from src.knowledge.rag import get_city_experiences
from src.models.constraints import check_experience
from src.planning import planner_v2a
from src.planning.planner_v2a import DAY_END_LIMIT, MIN_STOPS_PER_DAY, plan_v2a
from src.planning.timing import time_day


def candidates(persona):
    feasible = [e for e in get_city_experiences(persona.destination.lower()) if not check_experience(e, persona)]
    for e in feasible:
        e.persona_score = score_experience(e, persona)
    return feasible


@pytest.mark.parametrize("scenario", ALL_SCENARIOS, ids=lambda s: s.id)
def test_every_scenario_gets_a_plan_that_passes_the_hard_rules(scenario):
    persona = scenario.persona
    per_day = _experiences_per_day(persona)
    pool = candidates(persona)
    result = plan_v2a(pool, persona, per_day)

    assert result.status == "feasible", result.reason
    assert result.stopped_by == "local_optimum"
    assert len(result.days) == persona.trip_days
    ids = [e.id for day in result.days for e in day]
    assert len(ids) == len(set(ids))                                   # no stop twice
    assert set(ids) <= {e.id for e in pool}                            # only stops that passed the per-stop rules
    for day in result.days:
        assert MIN_STOPS_PER_DAY <= len(day) <= per_day
        assert sum(e.cost_euros for e in day) <= persona.hard.day_budget_ceiling
        assert time_day(day, persona)[-1].end <= DAY_END_LIMIT                   # the day fits in waking hours
    assert sum(e.cost_euros for day in result.days for e in day) <= persona.hard.trip_budget_ceiling


def test_no_single_move_improves_the_final_plan():
    persona = SCENARIO_BY_ID["balanced_paris_reference"].persona
    per_day = _experiences_per_day(persona)
    pool = candidates(persona)
    result = plan_v2a(pool, persona, per_day)

    scorer = planner_v2a._Scorer(persona, per_day)
    used = {e.id for day in result.days for e in day}
    unused = [e for e in pool if e.id not in used]
    better = [p for p in planner_v2a._neighbours(result.days, unused)
              if scorer.allowed(p) and scorer.score(p) > result.score + planner_v2a.EPS]
    assert better == []


def test_search_improves_on_the_greedy_start():
    persona = SCENARIO_BY_ID["balanced_paris_reference"].persona
    per_day = _experiences_per_day(persona)
    pool = candidates(persona)
    scorer = planner_v2a._Scorer(persona, per_day)
    greedy = scorer.score(planner_v2a._greedy_start(pool, persona.trip_days, scorer))

    result = plan_v2a(pool, persona, per_day)
    assert result.score >= greedy


def test_same_input_gives_the_same_plan():
    persona = SCENARIO_BY_ID["balanced_paris_reference"].persona
    per_day = _experiences_per_day(persona)
    first = plan_v2a(candidates(persona), persona, per_day)
    second = plan_v2a(candidates(persona), persona, per_day)
    assert [[e.id for e in d] for d in first.days] == [[e.id for e in d] for d in second.days]


def test_too_few_stops_is_infeasible_with_a_reason():
    persona = SCENARIO_BY_ID["balanced_paris_reference"].persona
    result = plan_v2a(candidates(persona)[:3], persona, 4)
    assert result.status == "infeasible"
    assert "at least" in result.reason


def test_a_short_catalogue_is_spread_evenly_over_the_days():
    persona = SCENARIO_BY_ID["length_fiveday_tokyo"].persona          # 15 stops for 5 days at 4 a day
    result = plan_v2a(candidates(persona), persona, _experiences_per_day(persona))
    assert sorted(len(day) for day in result.days) == [3, 3, 3, 3, 3]
