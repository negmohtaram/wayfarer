from src.agents.state import PlanningState
from src.models.itinerary import RegretItem


REGRET_THRESHOLDS = {
    "must_do": 7.5,
    "should_do": 5.0,
}


async def regret_minimizer_node(state: PlanningState) -> PlanningState:
    """Score every experience by regret potential and tier them."""
    state.setdefault("log", [])
    state["log"].append("💭 Running regret minimizer...")

    itinerary = state.get("itinerary")
    if not itinerary:
        return state

    persona = itinerary.persona
    regret_items = []

    for day in itinerary.days:
        for sched in day.experiences:
            exp = sched.experience

            # Regret scoring factors
            uniqueness = exp.local_score / 10  # rarer experiences score higher regret
            persona_match = (exp.persona_score or 0.5)
            time_sensitivity = 0.8 if exp.weather_dependent else 0.5

            # How well does it match deepest preferences?
            deepest_pref = max(
                persona.food_importance if exp.experience_type == "food" else 0,
                persona.culture_importance if exp.experience_type == "cultural" else 0,
                persona.nature_importance if exp.experience_type in ("active", "nature", "scenic") else 0,
                persona.photography_interest if exp.best_time_of_day == "golden_hour" else 0,
            ) / 10

            # Social context match
            social_match = 0.5
            if persona.social_context == "couple" and exp.romantic_score >= 8:
                social_match = 1.0
            elif persona.social_context == "solo" and exp.local_score >= 8:
                social_match = 0.9
            elif persona.social_context == "family" and exp.family_friendly:
                social_match = 0.8

            # "Story value" — will they talk about this for years?
            story_value = (exp.local_score + (exp.romantic_score if persona.social_context == "couple" else 0)) / 20

            regret_score = (
                0.30 * uniqueness +
                0.25 * persona_match +
                0.20 * deepest_pref +
                0.15 * social_match +
                0.10 * story_value
            ) * 10  # 0-10 scale

            # Determine tier
            if regret_score >= REGRET_THRESHOLDS["must_do"]:
                tier = "must_do"
            elif regret_score >= REGRET_THRESHOLDS["should_do"]:
                tier = "should_do"
            else:
                tier = "nice_to_have"

            # Build human-readable reason
            reason = _build_regret_reason(exp, regret_score, tier, persona)

            exp.regret_score = round(regret_score, 1)
            exp.regret_tier = tier

            regret_items.append(RegretItem(
                experience_name=exp.name,
                experience_id=exp.id,
                day_number=day.day_number,
                regret_score=round(regret_score, 1),
                tier=tier,
                reason=reason,
            ))

    # Sort by regret score descending
    regret_items.sort(key=lambda r: r.regret_score, reverse=True)

    itinerary.regret_rankings = regret_items
    state["itinerary"] = itinerary
    itinerary.status = "complete"

    must_dos = sum(1 for r in regret_items if r.tier == "must_do")
    state["log"].append(f"💭 Regret rankings complete: {must_dos} must-dos identified")

    return state


def _build_regret_reason(exp, score: float, tier: str, persona) -> str:
    if tier == "must_do":
        if exp.local_score >= 8:
            return f"A genuinely rare experience most visitors never find. If you skip this, you'll wonder."
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
