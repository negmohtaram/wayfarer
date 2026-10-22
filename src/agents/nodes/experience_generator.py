from src.agents.state import PlanningState
from src.knowledge.rag import get_city_experiences
from src.models.experience import Experience
from src.models.persona import TravelerPersona
from src.models.constraints import check_experience
from src.config import EXPERIENCE_POOL_MULTIPLIER


def score_experience(experience: Experience, persona: TravelerPersona, weather_good: bool = True) -> float:
    """Score an experience against a traveler persona (0-1 scale)."""

    # Preference match
    type_scores = {
        "food": persona.food_importance / 10,
        "cultural": persona.culture_importance / 10,
        "active": persona.nature_importance / 10,
        "nature": persona.nature_importance / 10,
        "scenic": (persona.nature_importance + persona.culture_importance) / 20,
        "passive": (10 - persona.pace) / 10 * 0.8,
        "serendipity": persona.local_vs_tourist / 10,
        "nightlife": persona.nightlife_importance / 10,
    }
    preference_score = type_scores.get(experience.experience_type, 0.5)

    # Local alignment
    local_alignment = 1 - abs(experience.local_score - persona.local_vs_tourist) / 10

    # Energy alignment
    energy_alignment = 1 - abs(experience.energy_required - (10 - persona.pace)) / 10

    # Physical suitability
    physical_ok = 1.0
    if "bad_knee" in persona.physical_constraints or "elderly" in persona.physical_constraints:
        if experience.physical_intensity == "high":
            physical_ok = 0.1
        elif experience.physical_intensity == "medium":
            physical_ok = 0.6

    # Crowd tolerance
    crowd_penalty = 0.0
    if persona.crowd_tolerance < 4 and experience.crowd_level > 7:
        crowd_penalty = 0.3

    # Social context bonus
    social_bonus = 0.0
    if persona.social_context == "couple" and experience.romantic_score >= 7:
        social_bonus = 0.15
    elif persona.social_context == "family" and experience.family_friendly:
        social_bonus = 0.1
    elif persona.social_context == "solo" and experience.local_score >= 7:
        social_bonus = 0.05

    # Photography bonus
    photo_bonus = 0.0
    if persona.photography_interest >= 7 and experience.best_time_of_day in ("golden_hour", "early_morning"):
        photo_bonus = 0.1

    # Weather penalty
    weather_penalty = 0.15 if (experience.weather_dependent and not weather_good) else 0.0

    score = (
        0.30 * preference_score +
        0.25 * local_alignment +
        0.20 * energy_alignment +
        0.15 * physical_ok +
        0.10 * (1 - crowd_penalty)
    ) + social_bonus + photo_bonus - weather_penalty

    return max(0.0, min(1.0, score))


async def experience_generator_node(state: PlanningState) -> PlanningState:
    """Load experiences from city knowledge and score them against persona."""
    state.setdefault("log", [])
    state["log"].append("✨ Generating experience pool...")

    persona = state.get("persona")
    if not persona:
        state["error"] = "No persona found"
        return state

    city = persona.destination.lower()

    # Load all experiences from city knowledge
    all_experiences = get_city_experiences(city)

    if not all_experiences:
        state["error"] = f"No experiences found for {city}"
        return state

    # Hard constraints first: an infeasible stop is removed before scoring, so no
    # preference score can ever bring it back. The reasons are kept for the trace.
    feasible, rejected = [], []
    for exp in all_experiences:
        violations = check_experience(exp, persona)
        if violations:
            rejected.extend(violations)
        else:
            feasible.append(exp)
    state["rejected"] = rejected
    if rejected:
        state["log"].append(f"🚫 {len(rejected)} experiences rejected by hard constraints")

    # Then rank what is left by preference
    scored = []
    for exp in feasible:
        exp.persona_score = score_experience(exp, persona)
        scored.append(exp)
    scored.sort(key=lambda e: e.persona_score or 0, reverse=True)

    # Ensure variety across types
    experience_pool = _ensure_variety(scored, persona)

    state["experience_pool"] = experience_pool
    state["log"].append(f"✨ Experience pool ready ({len(experience_pool)} candidates)")

    return state


def _ensure_variety(scored: list[Experience], persona: TravelerPersona) -> list[Experience]:
    """Ensure the pool has variety across experience types.

    Uses ID-based tracking at every insertion point so duplicates can
    never enter the pool regardless of Pydantic equality semantics.
    """
    type_buckets: dict[str, list[Experience]] = {}
    for exp in scored:
        type_buckets.setdefault(exp.experience_type, []).append(exp)

    days = persona.trip_days
    target_total = min(len(scored), days * EXPERIENCE_POOL_MULTIPLIER * 3)

    seen_ids: set[str] = set()
    pool: list[Experience] = []

    # Guarantee at least 2 representatives of every available type
    for exps in type_buckets.values():
        for exp in exps[:2]:
            if exp.id not in seen_ids:
                seen_ids.add(exp.id)
                pool.append(exp)

    # Fill remaining slots from the globally ranked list
    for exp in scored:
        if len(pool) >= target_total:
            break
        if exp.id not in seen_ids:
            seen_ids.add(exp.id)
            pool.append(exp)

    return sorted(pool, key=lambda e: e.persona_score or 0, reverse=True)
