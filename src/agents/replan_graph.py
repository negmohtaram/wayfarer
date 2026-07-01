from langgraph.graph import StateGraph, START, END
from src.agents.state import ReplanningState


async def disruption_analyzer_node(state: ReplanningState) -> ReplanningState:
    """Analyze disruption and prepare state for replanning remaining days."""
    from src.agents.llm import get_llm

    state.setdefault("log", [])
    disruption = state.get("disruption", "")
    itinerary = state.get("itinerary")
    completed = state.get("completed_day_indices", [])

    state["log"].append(f"🔄 Analyzing disruption: {disruption}")

    if not itinerary or not disruption:
        return state

    # Identify remaining days
    remaining_days = [
        day for day in itinerary.days
        if day.day_number - 1 not in completed
    ]

    if not remaining_days:
        state["log"].append("No remaining days to replan")
        return state

    # Use LLM to interpret disruption and suggest changes
    llm = get_llm(temperature=0.5)

    day_summary = "\n".join(
        f"Day {d.day_number}: " + ", ".join(s.experience.name for s in d.experiences)
        for d in remaining_days
    )

    prompt = f"""
You are a travel replanner. A traveler reported: "{disruption}"

Remaining schedule:
{day_summary}

Respond in JSON only:
{{
  "disruption_type": "weather|closed|delay|tired|love_neighborhood|other",
  "affected_experiences": ["list of experience names affected"],
  "recommendation": "brief recommendation for the replanner"
}}
"""

    try:
        import json, re
        response = await llm.ainvoke([{"role": "user", "content": prompt}])
        json_match = re.search(r"\{.*\}", response.content, re.DOTALL)
        if json_match:
            analysis = json.loads(json_match.group(0))
            state["log"].append(f"🔄 Disruption type: {analysis.get('disruption_type', 'other')}")
    except Exception:
        pass

    return state


async def replan_assembler_node(state: ReplanningState) -> ReplanningState:
    """Rebuild only the remaining days of the itinerary."""
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

    if not itinerary or not persona:
        return state

    remaining_day_numbers = [
        day.day_number for day in itinerary.days
        if day.day_number - 1 not in completed
    ]

    if not remaining_day_numbers:
        return state

    # Load experience pool excluding already-done experiences
    done_ids = set()
    for day_idx in completed:
        if day_idx < len(itinerary.days):
            for sched in itinerary.days[day_idx].experiences:
                done_ids.add(sched.experience.id)

    all_exps = get_city_experiences(persona.destination.lower())
    available = [e for e in all_exps if e.id not in done_ids]

    # Filter for disruption type (rain → indoor activities)
    if "rain" in disruption.lower() or "raining" in disruption.lower():
        available = [e for e in available if not e.weather_dependent]
        state["log"].append("🌧️  Rain detected: filtering to indoor activities")

    if "tired" in disruption.lower():
        available = [e for e in available if e.energy_required <= 4]
        state["log"].append("😴 Tired: filtering to low-energy activities")

    # Re-score and select
    for exp in available:
        exp.persona_score = score_experience(exp, persona)
    available.sort(key=lambda e: e.persona_score or 0, reverse=True)

    num_remaining = len(remaining_day_numbers)
    per_day = max(3, 5 - (len(completed)))
    selected = available[:num_remaining * per_day]

    city_coords = get_city_coordinates(persona.destination.lower())
    day_clusters = cluster_by_day(selected, num_remaining, tuple(city_coords))

    # Fetch weather for remaining days
    first_remaining_date = itinerary.days[remaining_day_numbers[0] - 1].date or datetime.now().strftime("%Y-%m-%d")
    weather_list = await fetch_weather(persona.destination, first_remaining_date, num_remaining)

    # Replace remaining days
    for i, day_number in enumerate(remaining_day_numbers):
        day_exps = day_clusters[i] if i < len(day_clusters) else []
        routed = optimize_route(day_exps)
        sequenced = sequence_by_energy(routed, persona, get_day_start_time(persona))
        weather = weather_list[i] if i < len(weather_list) else {}
        new_day = _build_day_schedule(sequenced, persona, day_number, persona.trip_days, weather)

        # Replace in itinerary
        for j, day in enumerate(itinerary.days):
            if day.day_number == day_number:
                itinerary.days[j] = new_day
                break

    itinerary.total_cost = sum(d.total_cost for d in itinerary.days)
    itinerary.status = "replanning"
    state["itinerary"] = itinerary
    state["log"].append(f"🔄 Replanned {num_remaining} remaining days")

    return state


def build_replan_graph():
    graph = StateGraph(ReplanningState)

    graph.add_node("disruption_analyzer", disruption_analyzer_node)
    graph.add_node("replan_assembler", replan_assembler_node)

    graph.add_edge(START, "disruption_analyzer")
    graph.add_edge("disruption_analyzer", "replan_assembler")
    graph.add_edge("replan_assembler", END)

    return graph.compile()


replan_graph = build_replan_graph()
