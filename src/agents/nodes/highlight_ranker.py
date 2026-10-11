from src.agents.state import PlanningState
from src.models.itinerary import HighlightItem


HIGHLIGHT_THRESHOLDS = {
    "must_do": 7.5,
    "should_do": 5.0,
}


async def highlight_ranker_node(state: PlanningState) -> PlanningState:
    """Rank itinerary experiences by personal highlight potential.

    Scores each scheduled experience on uniqueness, persona fit, preference
    depth, social context, and story value. Tiers them as must_do /
    should_do / nice_to_have so the user knows which experiences to protect
    if time or energy runs short.

    Note: ranks only experiences already in the itinerary. True regret
    minimization across excluded alternatives is a future phase.
    """
    state.setdefault("log", [])
    state["log"].append("💭 Ranking experience highlights...")

    itinerary = state.get("itinerary")
    if not itinerary:
        return state

    persona = itinerary.persona
    ranked_items = []

    for day in itinerary.days:
        for sched in day.experiences:
            exp = sched.experience

            uniqueness = exp.local_score / 10
            persona_match = exp.persona_score or 0.5
            # weather_dependent affects rescheduling priority but not highlight ranking

            deepest_pref = max(
                persona.food_importance if exp.experience_type == "food" else 0,
                persona.culture_importance if exp.experience_type == "cultural" else 0,
                persona.nature_importance if exp.experience_type in ("active", "nature", "scenic") else 0,
                persona.photography_interest if exp.best_time_of_day == "golden_hour" else 0,
            ) / 10

            social_match = 0.5
            if persona.social_context == "couple" and exp.romantic_score >= 8:
                social_match = 1.0
            elif persona.social_context == "solo" and exp.local_score >= 8:
                social_match = 0.9
            elif persona.social_context == "family" and exp.family_friendly:
                social_match = 0.8

            story_value = (
                exp.local_score + (exp.romantic_score if persona.social_context == "couple" else 0)
            ) / 20

            highlight_score = (
                0.30 * uniqueness +
                0.25 * persona_match +
                0.20 * deepest_pref +
                0.15 * social_match +
                0.10 * story_value
            ) * 10

            if highlight_score >= HIGHLIGHT_THRESHOLDS["must_do"]:
                tier = "must_do"
            elif highlight_score >= HIGHLIGHT_THRESHOLDS["should_do"]:
                tier = "should_do"
            else:
                tier = "nice_to_have"

            reason = _build_highlight_reason(exp, highlight_score, tier, persona)

            exp.highlight_score = round(highlight_score, 1)
            exp.highlight_tier = tier

            ranked_items.append(HighlightItem(
                experience_name=exp.name,
                experience_id=exp.id,
                day_number=day.day_number,
                highlight_score=round(highlight_score, 1),
                tier=tier,
                reason=reason,
            ))

    ranked_items.sort(key=lambda r: r.highlight_score, reverse=True)

    itinerary.highlight_rankings = ranked_items
    itinerary.status = "complete"
    state["itinerary"] = itinerary

    must_dos = sum(1 for r in ranked_items if r.tier == "must_do")
    state["log"].append(f"💭 Highlight rankings complete: {must_dos} must-dos identified")

    return state


def _build_highlight_reason(exp, score: float, tier: str, persona) -> str:
    if tier == "must_do":
        if exp.local_score >= 8:
            return "A genuinely rare experience most visitors never find. If you skip this, you'll wonder."
        elif exp.romantic_score >= 8 and persona.social_context == "couple":
            return "The most romantic moment of the trip. The kind of thing couples talk about for years."
        elif exp.experience_type == "food" and persona.food_importance >= 7:
            return "The food experience you came for. Nothing in your home city comes close."
        else:
            return "This is what makes this trip this trip. Don't skip it."
    elif tier == "should_do":
        return "A strong experience that fits who you are as a traveler. You'd feel it was missing."
    else:
        return "Pleasant and worth it if you have time, but won't haunt you if you skip it."
