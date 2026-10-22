import logging

from src.agents.state import PlanningState
from src.models.itinerary import CritiqueResult
from src.models.experience import Experience

logger = logging.getLogger(__name__)


async def revision_agent_node(state: PlanningState) -> PlanningState:
    """Apply critic feedback with deterministic repair strategies.

    Each critic type has an explicit repair path or an explicit log entry
    explaining why it cannot be repaired in this pass.
    """
    state.setdefault("log", [])
    state["log"].append("🔄 Revision agent applying critic feedback...")

    critiques: list[CritiqueResult] = state.get("critiques", [])
    itinerary = state.get("itinerary")
    experience_pool = state.get("experience_pool", [])

    if not itinerary or not experience_pool:
        state["revision_count"] = state.get("revision_count", 0) + 1
        return state

    days_to_revise: set[int] = set()
    for critique in critiques:
        if critique.verdict in ("revise", "reject_day"):
            for day_num in critique.affected_days:
                days_to_revise.add(day_num)
            if not critique.affected_days:
                days_to_revise.update(range(1, itinerary.persona.trip_days + 1))

    if not days_to_revise:
        state["revision_count"] = state.get("revision_count", 0) + 1
        return state

    state["log"].append(f"🔄 Revising days: {sorted(days_to_revise)}")

    used_ids: set[str] = set()
    for day in itinerary.days:
        for sched in day.experiences:
            used_ids.add(sched.experience.id)

    unused = [e for e in experience_pool if e.id not in used_ids]
    unused.sort(key=lambda e: e.persona_score or 0, reverse=True)

    persona = itinerary.persona
    revised_day_numbers: set[int] = set()
    # Each day is edited as a plain list of stops and rebuilt into a schedule once at the end,
    # so scheduled stops and raw replacements never sit in the same list.
    stops_by_day: dict[int, list[Experience]] = {}

    for day in itinerary.days:
        if day.day_number not in days_to_revise:
            continue

        stops = [s.experience for s in day.experiences]
        changed = False

        for critique in critiques:
            if critique.verdict in ("approve", "error"):
                continue

            # ── Energy: trim overloaded days for slow travelers ──────────────
            if critique.critic_type == "energy":
                if persona.pace >= 7 and len(stops) > 4:
                    stops = stops[:3]
                    changed = True

            # ── Authenticity: swap lowest local-score experience ─────────────
            elif critique.critic_type == "authenticity" and day.day_number in critique.affected_days:
                stops.sort(key=lambda e: e.local_score)
                for stop in stops[:1]:
                    replacement = _find_replacement(stop, unused, persona, "local")
                    if replacement:
                        stops.remove(stop)
                        stops.append(replacement)
                        unused.remove(replacement)
                        used_ids.add(replacement.id)
                        changed = True

            # ── Experience: introduce type variety ───────────────────────────
            elif critique.critic_type == "experience" and day.day_number in critique.affected_days:
                cultural = [e for e in stops if e.experience_type == "cultural"]
                if len(cultural) >= 3:
                    to_replace = cultural[-1]
                    replacement = _find_replacement(to_replace, unused, persona, "variety")
                    if replacement:
                        stops.remove(to_replace)
                        stops.append(replacement)
                        unused.remove(replacement)
                        changed = True
                else:
                    logger.debug(
                        "Day %d: experience critic requested revision but no actionable fix available "
                        "(fewer than 3 cultural experiences to swap)",
                        day.day_number,
                    )

            # ── Budget: remove most expensive experience ─────────────────────
            elif critique.critic_type == "budget" and day.day_number in critique.affected_days:
                if stops:
                    most_expensive = max(stops, key=lambda e: e.cost_euros)
                    if most_expensive.cost_euros > 0:
                        stops.remove(most_expensive)
                        changed = True

            # ── Logistics: rebuild day schedule to fix timing violations ─────
            # Travel-distance issues require re-clustering and cannot be fixed
            # by a schedule rebuild alone — those are logged but not repaired.
            elif critique.critic_type == "logistics" and day.day_number in critique.affected_days:
                timing_issues = [i for i in critique.issues if "scheduled at" in i]
                if timing_issues:
                    changed = True  # mark for rebuild — _build_day_schedule re-sorts by time window

                travel_issues = [i for i in critique.issues if "Long travel" in i]
                if travel_issues:
                    state["log"].append(
                        f"⚠️  Day {day.day_number}: travel-distance issues require re-clustering "
                        f"— not repaired in this pass"
                    )

        if changed:
            revised_day_numbers.add(day.day_number)
            stops_by_day[day.day_number] = stops

    # Rebuild full schedule for every modified day
    if revised_day_numbers:
        from src.agents.nodes.assembler import _build_day_schedule

        for i, day in enumerate(itinerary.days):
            if day.day_number not in revised_day_numbers:
                continue
            exps = stops_by_day[day.day_number]
            weather_dict = day.weather.model_dump() if day.weather else {}
            itinerary.days[i] = _build_day_schedule(
                exps, persona, day.day_number, persona.trip_days, weather_dict
            )

    itinerary.total_cost = sum(d.total_cost for d in itinerary.days)
    itinerary.revision_count = state.get("revision_count", 0) + 1

    state["itinerary"] = itinerary
    state["revision_count"] = itinerary.revision_count
    state["log"].append(
        f"🔄 Revision {itinerary.revision_count} complete ({len(revised_day_numbers)} days rebuilt)"
    )

    return state


def _find_replacement(original, unused, persona, strategy):
    """Return the best unused replacement experience for a given strategy."""
    if not unused:
        return None

    if strategy == "local":
        candidates = [e for e in unused if e.local_score > original.local_score + 1]
        candidates.sort(key=lambda e: e.local_score, reverse=True)
    elif strategy == "variety":
        candidates = [e for e in unused if e.experience_type != "cultural"]
    else:
        candidates = unused

    return candidates[0] if candidates else None
