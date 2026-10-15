import asyncio
import logging
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

from src.agents.state import ReplanningState
from src.evaluation.recording import note_fallback

logger = logging.getLogger(__name__)


class DisruptionAnalysis(BaseModel):
    disruption_type: Literal["weather", "closed", "delay", "tired", "love_neighborhood", "other"]
    affected_experiences: list[str] = Field(default_factory=list)
    recommendation: str = ""


async def disruption_analyzer_node(state: ReplanningState) -> ReplanningState:
    """Classify the disruption using structured LLM output.

    Uses with_structured_output so the LLM is forced to return a valid
    DisruptionAnalysis — no regex, no manual JSON parsing.
    Falls back to an empty analysis dict on failure.
    """
    from src.agents.llm import call_with_retry, get_llm
    from src.config import GROQ_MODEL

    state.setdefault("log", [])
    disruption = state.get("disruption", "")
    itinerary = state.get("itinerary")
    completed = state.get("completed_day_indices", [])

    state["log"].append(f"🔄 Analyzing disruption: {disruption}")

    if not itinerary or not disruption:
        return state

    remaining_days = [
        day for day in itinerary.days
        if day.day_number - 1 not in completed
    ]
    if not remaining_days:
        state["log"].append("No remaining days to replan")
        return state

    day_summary = "\n".join(
        f"Day {d.day_number}: " + ", ".join(s.experience.name for s in d.experiences)
        for d in remaining_days
    )

    messages = [
        SystemMessage(
            "You are a travel replanner. Classify the traveler's disruption and identify "
            "which scheduled experiences are affected."
        ),
        HumanMessage(
            f'Traveler reported: "{disruption}"\n\nRemaining schedule:\n{day_summary}'
        ),
    ]

    try:
        llm = get_llm(temperature=0.3).with_structured_output(DisruptionAnalysis)
        result: DisruptionAnalysis = await call_with_retry(
            llm.ainvoke, messages,
            meta={"site": "disruption_analyzer", "model": GROQ_MODEL, "temperature": 0.3,
                  "schema": "DisruptionAnalysis"},
        )
        state["disruption_analysis"] = result.model_dump()
        state["log"].append(f"🔄 Disruption classified as: {result.disruption_type}")
    except Exception as e:
        logger.warning("Disruption analysis failed: %s", e)
        note_fallback("disruption_analyzer", e)
        state["disruption_analysis"] = {}

    return state


async def replan_assembler_node(state: ReplanningState) -> ReplanningState:
    """Rebuild only the remaining days, guided by the disruption analysis."""
    from src.agents.nodes.experience_generator import score_experience
    from src.agents.nodes.assembler import _build_day_schedule, fetch_weather
    from src.optimization.route import optimize_route, cluster_by_day
    from src.optimization.energy import sequence_by_energy, get_day_start_time
    from src.knowledge.rag import get_city_experiences, get_city_coordinates
    from datetime import datetime

    state.setdefault("log", [])
    itinerary = state.get("itinerary")
    persona = state.get("persona") or (itinerary.persona if itinerary else None)
    completed = state.get("completed_day_indices", [])
    disruption = state.get("disruption", "")

    # Prefer LLM analysis over raw keyword matching
    analysis = state.get("disruption_analysis") or {}
    disruption_type = analysis.get("disruption_type", "other")
    affected_names = {n.lower() for n in analysis.get("affected_experiences", [])}

    # Fallback: keyword hints if the LLM analysis didn't produce a usable type
    if disruption_type == "other":
        low = disruption.lower()
        if "rain" in low or "weather" in low or "storm" in low:
            disruption_type = "weather"
        elif "tired" in low or "exhausted" in low or "rest" in low:
            disruption_type = "tired"
        elif "closed" in low or "shut" in low:
            disruption_type = "closed"

    if not itinerary or not persona:
        return state

    remaining_day_numbers = [
        day.day_number for day in itinerary.days
        if day.day_number - 1 not in completed
    ]
    if not remaining_day_numbers:
        return state

    # Exclude already-completed experiences
    done_ids: set[str] = set()
    for day_idx in completed:
        if day_idx < len(itinerary.days):
            for sched in itinerary.days[day_idx].experiences:
                done_ids.add(sched.experience.id)

    all_exps = get_city_experiences(persona.destination.lower())
    available = [e for e in all_exps if e.id not in done_ids]

    # Apply disruption-type-aware filtering
    if disruption_type == "weather":
        available = [e for e in available if not e.weather_dependent]
        state["log"].append("🌧️  Weather disruption: filtering to indoor activities")

    elif disruption_type == "tired":
        available = [e for e in available if e.energy_required <= 4]
        state["log"].append("😴 Fatigue: filtering to low-energy activities")

    elif disruption_type == "closed" and affected_names:
        # Exclude venues the LLM identified as affected
        available = [e for e in available if e.name.lower() not in affected_names]
        state["log"].append(f"🚪 Closed venues removed: {', '.join(affected_names)}")

    elif disruption_type == "love_neighborhood":
        # Traveler wants to stay in the current area — sort by proximity to last experience
        last_completed_exp = None
        if completed:
            last_day_idx = max(completed)
            if last_day_idx < len(itinerary.days):
                day_exps = itinerary.days[last_day_idx].experiences
                if day_exps:
                    last_completed_exp = day_exps[-1].experience
        if last_completed_exp:
            from src.optimization.route import haversine
            available.sort(
                key=lambda e: haversine(
                    e.location[0], e.location[1],
                    last_completed_exp.location[0], last_completed_exp.location[1],
                )
            )
            state["log"].append("💛 Staying in the neighbourhood: prioritising nearby experiences")

    # Re-score and select
    for exp in available:
        exp.persona_score = score_experience(exp, persona)
    available.sort(key=lambda e: e.persona_score or 0, reverse=True)

    num_remaining = len(remaining_day_numbers)
    per_day = max(3, 5 - len(completed))
    selected = available[: num_remaining * per_day]

    city_coords = get_city_coordinates(persona.destination.lower())
    day_clusters = cluster_by_day(selected, num_remaining, tuple(city_coords))

    first_remaining_date = (
        itinerary.days[remaining_day_numbers[0] - 1].date
        or datetime.now().strftime("%Y-%m-%d")
    )
    weather_list = await fetch_weather(persona.destination, first_remaining_date, num_remaining)

    for i, day_number in enumerate(remaining_day_numbers):
        day_exps = day_clusters[i] if i < len(day_clusters) else []
        routed = optimize_route(day_exps)
        sequenced = sequence_by_energy(routed, persona, get_day_start_time(persona))
        weather = weather_list[i] if i < len(weather_list) else None
        new_day = _build_day_schedule(sequenced, persona, day_number, persona.trip_days, weather)

        for j, day in enumerate(itinerary.days):
            if day.day_number == day_number:
                itinerary.days[j] = new_day
                break

    itinerary.total_cost = sum(d.total_cost for d in itinerary.days)
    itinerary.status = "replanning"
    state["itinerary"] = itinerary
    state["log"].append(f"🔄 Replanned {num_remaining} remaining days (disruption: {disruption_type})")

    return state


async def replan_validator_node(state: ReplanningState) -> ReplanningState:
    """Run lightweight deterministic critics on the replanned days as a sanity check.

    This mirrors the planning graph's critic pass but without revision cycles —
    replanning is already a recovery action, so we surface issues to the user
    rather than triggering another full replan loop.
    """
    from src.agents.nodes.critics.logistics import logistics_critic
    from src.agents.nodes.critics.budget import budget_critic

    state.setdefault("log", [])
    itinerary = state.get("itinerary")
    if not itinerary:
        return state

    results = await asyncio.gather(
        logistics_critic(itinerary),
        budget_critic(itinerary),
        return_exceptions=True,
    )

    issues_found = False
    for result in results:
        if isinstance(result, Exception):
            logger.warning("Replan validator critic failed: %s", result)
            continue
        if result.verdict != "approve":
            issues_found = True
            for issue in result.issues[:2]:
                state["log"].append(f"⚠️  {result.critic_type.title()}: {issue}")

    if not issues_found:
        state["log"].append("✓ Replanned days passed logistics and budget checks")

    return state


def build_replan_graph():
    graph = StateGraph(ReplanningState)

    graph.add_node("disruption_analyzer", disruption_analyzer_node)
    graph.add_node("replan_assembler", replan_assembler_node)
    graph.add_node("replan_validator", replan_validator_node)

    graph.add_edge(START, "disruption_analyzer")
    graph.add_edge("disruption_analyzer", "replan_assembler")
    graph.add_edge("replan_assembler", "replan_validator")
    graph.add_edge("replan_validator", END)

    return graph.compile()


replan_graph = build_replan_graph()
