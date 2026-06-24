from src.agents.state import PlanningState
from src.models.experience import Experience
from src.models.persona import TravelerPersona
from src.optimization.route import cluster_by_day, optimize_route
from src.optimization.energy import sequence_by_energy, get_day_start_time
from src.knowledge.rag import get_city_coordinates


def _experiences_per_day(persona: TravelerPersona) -> int:
    """How many experiences fit in a day based on pace."""
    if persona.pace >= 8:
        return 3
    elif persona.pace >= 5:
        return 4
    else:
        return 5


async def optimizer_node(state: PlanningState) -> PlanningState:
    """Cluster experiences by day using route optimization + energy sequencing."""
    state.setdefault("log", [])
    state["log"].append("📐 Optimizing route and energy sequence...")

    persona = state.get("persona")
    pool = state.get("experience_pool", [])

    if not persona or not pool:
        state["error"] = "Missing persona or experience pool"
        return state

    days = persona.trip_days
    per_day = _experiences_per_day(persona)
    needed = days * per_day

    # Take top-scored experiences, ensuring we don't take tourist traps for high local-score personas
    candidates = pool
    if persona.local_vs_tourist >= 7:
        # Filter out experiences with very low local scores for local-seeking travelers
        filtered = [e for e in candidates if e.local_score >= 4.0]
        if len(filtered) >= needed:
            candidates = filtered

    selected = candidates[:needed]

    # Get city center for clustering reference
    city_coords = get_city_coordinates(persona.destination.lower())
    city_center = (city_coords[0], city_coords[1])

    # Cluster into days by geographic proximity
    day_clusters = cluster_by_day(selected, days, city_center)

    # Within each day, route-optimize then energy-sequence
    optimized_days = []
    for day_idx, day_experiences in enumerate(day_clusters):
        if not day_experiences:
            continue

        # Route optimize within the day
        routed = optimize_route(day_experiences)

        # Energy sequence (reorder within the geographic cluster)
        start_minutes = get_day_start_time(persona)
        sequenced = sequence_by_energy(routed, persona, start_minutes)

        optimized_days.append(sequenced)

    state["scored_experiences"] = optimized_days  # type: ignore[assignment]
    state["log"].append(f"📐 Route optimized: {days} days, {per_day} experiences/day")

    return state
