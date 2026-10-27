from src.agents.state import PlanningState
from src.models.persona import TravelerPersona
from src.optimization.route import cluster_by_day, optimize_route
from src.knowledge.rag import get_city_coordinates
from src.config import PLANNER
from src.planning.planner_v2a import plan_v2a


def _experiences_per_day(persona: TravelerPersona) -> int:
    """How many experiences fit in a day based on pace."""
    if persona.pace >= 8:
        return 3
    elif persona.pace >= 5:
        return 4
    else:
        return 5


async def optimizer_node(state: PlanningState) -> PlanningState:
    """Cluster experiences into geographic day-groups and route-optimize within each day.

    Ordering responsibility: optimizer produces geographically coherent clusters
    with route-optimized ordering within each day. The assembler owns the final
    within-day ordering (time-of-day sort). Energy sequencing is not applied here
    to avoid conflicting reorder passes — temporary state pending a unified planner.
    """
    state.setdefault("log", [])
    state["log"].append("📐 Optimizing route...")

    persona = state.get("persona")
    pool = state.get("experience_pool", [])

    if not persona or not pool:
        state["error"] = "Missing persona or experience pool"
        return state

    days = persona.trip_days
    per_day = _experiences_per_day(persona)

    if state.get("planner", PLANNER) == "v2a":
        result = plan_v2a(state.get("feasible") or pool, persona, per_day)
        state["planner_trace"] = {
            "planner": "v2a", "status": result.status, "score": round(result.score, 4),
            "moves_tried": result.moves_tried, "moves_kept": result.moves_kept,
            "stopped_by": result.stopped_by, "shortfall": result.shortfall, "reason": result.reason,
        }
        if result.status != "feasible":
            state["error"] = f"No plan: {result.reason}"
            return state
        state["optimized_days"] = result.days
        state["log"].append(f"📐 Planner v2a: {days} days, {result.moves_kept} improving moves")
        return state

    needed = days * per_day

    candidates = pool
    if persona.local_vs_tourist >= 7:
        filtered = [e for e in candidates if e.local_score >= 4.0]
        if len(filtered) >= needed:
            candidates = filtered

    selected = candidates[:needed]

    city_coords = get_city_coordinates(persona.destination.lower())
    city_center = (city_coords[0], city_coords[1])

    day_clusters = cluster_by_day(selected, days, city_center)

    optimized_days = []
    for day_experiences in day_clusters:
        if not day_experiences:
            continue
        routed = optimize_route(day_experiences)
        optimized_days.append(routed)

    state["optimized_days"] = optimized_days
    state["log"].append(f"📐 Route optimized: {days} days, {per_day} experiences/day")

    return state
