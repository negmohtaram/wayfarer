from src.agents.state import PlanningState
from src.models.itinerary import CritiqueResult
from src.models.experience import ScheduledExperience


async def revision_agent_node(state: PlanningState) -> PlanningState:
    """Apply critic feedback and regenerate weak days."""
    state.setdefault("log", [])
    state["log"].append("🔄 Revision agent applying critic feedback...")

    critiques: list[CritiqueResult] = state.get("critiques", [])
    itinerary = state.get("itinerary")
    experience_pool = state.get("experience_pool", [])

    if not itinerary or not experience_pool:
        state["revision_count"] = state.get("revision_count", 0) + 1
        return state

    # Collect affected days across all critics
    days_to_revise = set()
    for critique in critiques:
        if critique.verdict in ("revise", "reject_day"):
            for day_num in critique.affected_days:
                days_to_revise.add(day_num)
            if not critique.affected_days:
                # If no specific day, flag all days
                days_to_revise.update(range(1, itinerary.persona.trip_days + 1))

    if not days_to_revise:
        state["revision_count"] = state.get("revision_count", 0) + 1
        return state

    state["log"].append(f"🔄 Revising days: {sorted(days_to_revise)}")

    # Get currently used experience IDs
    used_ids = set()
    for day in itinerary.days:
        for sched in day.experiences:
            used_ids.add(sched.experience.id)

    # Get candidate replacement experiences (unused, high-scored)
    unused = [e for e in experience_pool if e.id not in used_ids]
    unused.sort(key=lambda e: e.persona_score or 0, reverse=True)

    persona = itinerary.persona

    # Apply revisions to flagged days
    for day in itinerary.days:
        if day.day_number not in days_to_revise:
            continue

        # Check specific issues and apply targeted fixes
        for critique in critiques:
            if critique.critic_type == "energy" and critique.verdict != "approve":
                # Reduce experience count for slow travelers
                if persona.pace >= 7 and len(day.experiences) > 4:
                    day.experiences = day.experiences[:3]

            if critique.critic_type == "authenticity" and day.day_number in critique.affected_days:
                # Replace lowest local-score experiences
                day.experiences.sort(key=lambda s: s.experience.local_score)
                to_replace = day.experiences[:1]
                for sched in to_replace:
                    replacement = _find_replacement(sched.experience, unused, persona, "local")
                    if replacement:
                        day.experiences.remove(sched)
                        day.experiences.append(ScheduledExperience(
                            experience=replacement,
                            start_time=sched.start_time,
                            end_time=sched.end_time,
                            travel_time_from_prev=sched.travel_time_from_prev,
                            travel_mode=sched.travel_mode,
                        ))
                        unused.remove(replacement)
                        used_ids.add(replacement.id)

            if critique.critic_type == "experience" and day.day_number in critique.affected_days:
                # Add variety — replace one cultural with something different
                cultural_exps = [s for s in day.experiences if s.experience.experience_type == "cultural"]
                if len(cultural_exps) >= 3:
                    to_replace = cultural_exps[-1]
                    replacement = _find_replacement(
                        to_replace.experience, unused, persona, "variety"
                    )
                    if replacement:
                        day.experiences.remove(to_replace)
                        day.experiences.append(ScheduledExperience(
                            experience=replacement,
                            start_time=to_replace.start_time,
                            end_time=to_replace.end_time,
                            travel_time_from_prev=to_replace.travel_time_from_prev,
                            travel_mode=to_replace.travel_mode,
                        ))
                        unused.remove(replacement)

        # Recalculate day cost
        day.total_cost = sum(s.experience.cost_euros for s in day.experiences)

    # Update total
    itinerary.total_cost = sum(d.total_cost for d in itinerary.days)
    itinerary.revision_count = state.get("revision_count", 0) + 1

    state["itinerary"] = itinerary
    state["revision_count"] = itinerary.revision_count
    state["log"].append(f"🔄 Revision {itinerary.revision_count} complete")

    return state


def _find_replacement(original, unused, persona, strategy):
    """Find a replacement experience based on strategy."""
    if not unused:
        return None

    if strategy == "local":
        candidates = [e for e in unused if e.local_score > original.local_score + 1]
        candidates.sort(key=lambda e: e.local_score, reverse=True)
    elif strategy == "variety":
        non_cultural = [e for e in unused if e.experience_type not in ("cultural",)]
        candidates = non_cultural
    else:
        candidates = unused

    if not candidates:
        return None

    return candidates[0]
