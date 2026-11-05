"""Day timing: walk or transit per leg, and the wait at the stop on every transit leg."""
from src.evaluation.scenarios.baseline_scenarios import SCENARIO_BY_ID
from src.knowledge.rag import get_city_experiences
from src.optimization.route import travel_time_minutes
from src.planning import timing
from src.planning.timing import TRANSIT_WAIT_MIN, time_day

PERSONA = SCENARIO_BY_ID["balanced_paris_reference"].persona
KM_PER_DEGREE_LAT = 111.2


def two_stops(km: float):
    first = get_city_experiences("paris")[0].model_copy(update={"best_time_of_day": "any"})
    lat, lng = first.location
    second = first.model_copy(update={"id": "second", "location": (lat + km / KM_PER_DEGREE_LAT, lng)})
    return first, second


def test_short_leg_is_walked_with_no_wait():
    a, b = two_stops(1.0)
    leg = time_day([a, b], PERSONA)[1]
    assert leg.mode == "walk"
    assert leg.travel == travel_time_minutes(a, b, "walk")


def test_long_leg_takes_transit_and_pays_the_wait():
    a, b = two_stops(3.0)
    leg = time_day([a, b], PERSONA)[1]
    assert leg.mode == "transit"
    assert leg.travel == travel_time_minutes(a, b, "transit") + TRANSIT_WAIT_MIN


def test_long_walk_stays_a_walk_when_the_wait_makes_transit_slower(monkeypatch):
    monkeypatch.setattr(timing, "TRANSIT_WAIT_MIN", 30)
    a, b = two_stops(1.8)                       # about 24 min on foot, 5 + 30 by transit
    leg = time_day([a, b], PERSONA)[1]
    assert travel_time_minutes(a, b, "walk") > timing.WALK_LIMIT_MIN
    assert leg.mode == "walk"
