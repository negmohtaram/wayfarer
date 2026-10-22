"""Hard constraints filter, soft preferences rank: an infeasible stop is rejected, never outscored."""
import asyncio

from src.agents.nodes import experience_generator as gen
from src.agents.nodes.critics.budget import budget_critic
from src.agents.nodes.revision import revision_agent_node
from src.models.experience import Experience, ScheduledExperience
from src.models.itinerary import CritiqueResult, DayPlan, Itinerary
from src.models.persona import TravelerPersona


def exp(id, intensity="low", cost=0.0, kind="food", local=9.0):
    return Experience(id=id, name=id, location=(52.37, 4.89), experience_type=kind,
                      best_time_of_day="any", physical_intensity=intensity, cost_euros=cost,
                      local_score=local, energy_required=3.0, crowd_level=2.0)


def foodie(**kw):
    return TravelerPersona(destination="amsterdam", trip_days=2, budget_per_day=100, pace=3,
                           food_importance=10, local_vs_tourist=9, crowd_tolerance=9, **kw)


def pool_for(persona, experiences, monkeypatch):
    monkeypatch.setattr(gen, "get_city_experiences", lambda city: [e.model_copy() for e in experiences])
    return asyncio.run(gen.experience_generator_node({"persona": persona}))


# 1. A stop that matches every preference except access is rejected, with the reason kept.
def test_perfect_match_with_stairs_is_rejected_for_a_wheelchair_user(monkeypatch):
    stairs = exp("canal_tasting_upstairs", intensity="medium")
    others = [exp(f"ok_{i}", local=5.0) for i in range(6)]
    state = pool_for(foodie(physical_constraints=["wheelchair"]), [stairs, *others], monkeypatch)
    assert "canal_tasting_upstairs" not in {e.id for e in state["experience_pool"]}
    [rejected] = [v for v in state["rejected"] if "canal_tasting_upstairs" in v.affected_ids]
    assert rejected.code == "ACCESSIBILITY"


# 2. No bonus can bring it back: the hard rule is not a term in the score, so the score
#    can be anything and the stop is still filtered out before scoring.
def test_no_score_can_bring_back_an_inaccessible_stop(monkeypatch):
    stairs = exp("canal_tasting_upstairs", intensity="medium")
    monkeypatch.setattr(gen, "score_experience", lambda e, p, weather_good=True: 1.0)
    state = pool_for(foodie(physical_constraints=["wheelchair"]), [stairs, exp("ok")], monkeypatch)
    assert "canal_tasting_upstairs" not in {e.id for e in state["experience_pool"]}


def test_access_is_not_part_of_the_preference_score():
    stairs = exp("canal_tasting_upstairs", intensity="medium")
    with_chair = gen.score_experience(stairs, foodie(physical_constraints=["wheelchair"]))
    without = gen.score_experience(stairs, foodie())
    assert with_chair == without


# 3. Budget boundary: exactly at the ceiling is allowed; one cent over is a typed violation
#    and the critic can never approve it, whatever its score.
def day_costing(n, cost):
    return DayPlan(day_number=n, total_cost=cost)


def itinerary_with(costs):
    persona = foodie()
    return Itinerary(session_id="t", persona=persona, destination="amsterdam",
                     days=[day_costing(i + 1, c) for i, c in enumerate(costs)], total_cost=sum(costs))


def test_day_at_the_ceiling_passes():
    ceiling = foodie().hard.day_budget_ceiling                    # 100 x 1.2 = 120
    result = asyncio.run(budget_critic(itinerary_with([ceiling, 50])))
    assert result.violations == [] and result.verdict == "approve"


def test_one_cent_over_the_ceiling_is_a_violation_and_never_approved():
    ceiling = foodie().hard.day_budget_ceiling
    result = asyncio.run(budget_critic(itinerary_with([ceiling + 0.01, 50])))
    [v] = result.violations
    assert (v.code, v.day, v.required) == ("DAILY_BUDGET", 1, ceiling)
    assert v.observed == ceiling + 0.01
    assert result.verdict != "approve"          # the old critic scored this 8.5 and approved it


# 4. Nothing fits: a swap and a budget fix on the same day must not crash
#    (stress_ultra_low_budget: "'Experience' object has no attribute 'experience'").
def scheduled(e):
    return ScheduledExperience(experience=e, start_time="10:00", end_time="11:00")


def test_swap_and_budget_fix_on_the_same_day_do_not_crash():
    persona = TravelerPersona(destination="amsterdam", trip_days=1, budget_per_day=25, local_vs_tourist=9)
    day = DayPlan(day_number=1, experiences=[scheduled(exp("touristy", cost=40, local=2)),
                                             scheduled(exp("pricey", cost=60, local=6))], total_cost=100)
    itinerary = Itinerary(session_id="t", persona=persona, destination="amsterdam", days=[day], total_cost=100)
    critiques = [
        CritiqueResult(critic_type="authenticity", score=4, verdict="revise", affected_days=[1]),
        CritiqueResult(critic_type="budget", score=4, verdict="revise", affected_days=[1]),
    ]
    state = {"itinerary": itinerary, "critiques": critiques,
             "experience_pool": [exp("local_gem", cost=30, local=9.5)]}
    out = asyncio.run(revision_agent_node(state))
    ids = {s.experience.id for s in out["itinerary"].days[0].experiences}
    assert "touristy" not in ids            # swapped for the local stop
    assert len(ids) == 1                    # and the most expensive stop removed
