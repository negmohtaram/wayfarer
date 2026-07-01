import httpx
from datetime import datetime, timedelta
from src.agents.state import PlanningState
from src.models.experience import Experience, ScheduledExperience
from src.models.itinerary import Itinerary, DayPlan, WeatherForecast
from src.models.persona import TravelerPersona
from src.optimization.route import travel_time_minutes
from src.optimization.energy import minutes_to_time, get_day_start_time


ARC_LABELS = {
    1: ("Orientation Day", "Settle in, get your bearings, claim the city as your own. Easy wins and first impressions."),
    2: ("Deep Dive Day", "Go further, explore the layers beneath the surface."),
    3: ("Peak Day", "The day you came for. Your most memorable experiences, perfectly paced."),
    4: ("Discovery Day", "Wander further, find the unexpected."),
    5: ("Reflection Day", "Slow down, absorb everything. The city is yours now."),
}

FINAL_DAY_LABEL = ("The Farewell Day", "End on a high. See something beautiful. Buy nothing. Leave full.")


async def fetch_weather(city: str, start_date: str, days: int) -> list[dict]:
    """Fetch weather from Open-Meteo API (free, no key required)."""
    city_coords = {
        "amsterdam": (52.3676, 4.9041),
        "paris": (48.8566, 2.3522),
        "tokyo": (35.6762, 139.6503),
        "london": (51.5074, -0.1278),
    }
    coords = city_coords.get(city.lower(), (52.3676, 4.9041))

    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={coords[0]}&longitude={coords[1]}"
            f"&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max,weathercode"
            f"&timezone=auto&start_date={start_date}&end_date="
        )
        end = (datetime.fromisoformat(start_date) + timedelta(days=days - 1)).strftime("%Y-%m-%d")
        url += end

        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                daily = data.get("daily", {})
                result = []
                for i in range(days):
                    wcode = daily.get("weathercode", [0] * days)[i] if i < len(daily.get("weathercode", [])) else 0
                    condition = _wmo_to_condition(wcode)
                    result.append({
                        "date": daily.get("time", [""] * days)[i] if i < len(daily.get("time", [])) else "",
                        "condition": condition,
                        "temp_high": daily.get("temperature_2m_max", [20] * days)[i] if i < len(daily.get("temperature_2m_max", [])) else 20,
                        "temp_low": daily.get("temperature_2m_min", [12] * days)[i] if i < len(daily.get("temperature_2m_min", [])) else 12,
                        "rain_probability": daily.get("precipitation_probability_max", [0] * days)[i] if i < len(daily.get("precipitation_probability_max", [])) else 0,
                    })
                return result
    except Exception:
        pass

    # Fallback: dummy weather
    return [
        {"date": "", "condition": "partly_cloudy", "temp_high": 18, "temp_low": 11, "rain_probability": 20}
        for _ in range(days)
    ]


def _wmo_to_condition(code: int) -> str:
    if code == 0:
        return "sunny"
    elif code <= 3:
        return "partly_cloudy"
    elif code <= 48:
        return "cloudy"
    elif code <= 67:
        return "rainy"
    elif code <= 77:
        return "snowy"
    elif code <= 82:
        return "rainy"
    else:
        return "stormy"


def _get_arc_for_day(day_num: int, total_days: int) -> tuple[str, str]:
    if total_days == 1:
        return ("The Full Day", "Everything in one perfect day.")
    if day_num == total_days:
        return FINAL_DAY_LABEL
    if total_days <= 3:
        if day_num == 1:
            return ARC_LABELS[1]
        elif day_num == total_days - 1 or (total_days == 3 and day_num == 2):
            return ARC_LABELS[3]
    else:
        mid = total_days // 2 + 1
        if day_num == 1:
            return ARC_LABELS[1]
        elif day_num == mid:
            return ARC_LABELS[3]
        elif day_num == 2:
            return ARC_LABELS[2]
        elif day_num <= mid:
            return ARC_LABELS[4]
        else:
            return ARC_LABELS[5]
    return ARC_LABELS.get(day_num, ARC_LABELS[2])


TIME_OF_DAY_WINDOWS = {
    "early_morning": (6 * 60, 8 * 60),
    "morning": (8 * 60, 12 * 60),
    "afternoon": (12 * 60, 17 * 60),
    "golden_hour": (17 * 60, 20 * 60),
    "evening": (19 * 60, 22 * 60),
    "night": (21 * 60, 23 * 60),
    "any": (8 * 60, 21 * 60),
}


def _sort_by_time_of_day(experiences: list[Experience], persona: TravelerPersona) -> list[Experience]:
    """Reorder experiences to respect best_time_of_day constraints."""
    time_order = {
        "early_morning": 0,
        "morning": 1,
        "afternoon": 2,
        "golden_hour": 3,
        "evening": 4,
        "night": 5,
        "any": 2,
    }
    return sorted(experiences, key=lambda e: time_order.get(e.best_time_of_day, 2))


def _build_day_schedule(
    experiences: list[Experience],
    persona: TravelerPersona,
    day_number: int,
    total_days: int,
    weather: dict,
) -> DayPlan:
    start_minutes = get_day_start_time(persona)

    # Reorder to respect time-of-day preferences
    ordered = _sort_by_time_of_day(experiences, persona)

    current_time = start_minutes
    scheduled = []
    total_cost = 0.0
    total_walk_km = 0.0

    prev_exp = None
    for exp in ordered:
        # Advance time to meet best_time_of_day window if needed
        window = TIME_OF_DAY_WINDOWS.get(exp.best_time_of_day, (8 * 60, 21 * 60))
        if current_time < window[0]:
            current_time = window[0]

        # Travel time from previous
        travel_mins = 0
        travel_mode = "walk"
        if prev_exp:
            travel_mins = travel_time_minutes(prev_exp, exp, "walk")
            if travel_mins > 20:
                travel_mode = "transit"
                travel_mins = travel_time_minutes(prev_exp, exp, "transit")

        current_time += travel_mins

        start_str = minutes_to_time(current_time)
        end_str = minutes_to_time(current_time + exp.duration_minutes)

        scheduled.append(ScheduledExperience(
            experience=exp,
            start_time=start_str,
            end_time=end_str,
            travel_time_from_prev=travel_mins,
            travel_mode=travel_mode,
        ))

        current_time += exp.duration_minutes
        total_cost += exp.cost_euros

        # Add rest buffer for slow travelers
        if persona.pace >= 7:
            current_time += 30  # 30-min breathing room

        prev_exp = exp

    arc_label, arc_desc = _get_arc_for_day(day_number, total_days)

    weather_forecast = WeatherForecast(
        date=weather.get("date", ""),
        condition=weather.get("condition", "partly_cloudy"),
        temp_high=weather.get("temp_high", 18),
        temp_low=weather.get("temp_low", 10),
        rain_probability=weather.get("rain_probability", 20),
    )

    return DayPlan(
        day_number=day_number,
        date=weather.get("date", ""),
        arc_label=arc_label,
        arc_description=arc_desc,
        experiences=scheduled,
        total_cost=round(total_cost, 2),
        total_walking_km=round(total_walk_km, 2),
        weather=weather_forecast,
        day_summary=f"A {arc_label.lower()}: {arc_desc}",
    )


async def assembler_node(state: PlanningState) -> PlanningState:
    """Assemble the final day-by-day itinerary with timing and arc."""
    state.setdefault("log", [])
    state["log"].append("📅 Assembling itinerary...")

    persona = state.get("persona")
    optimized_days: list[list[Experience]] = state.get("scored_experiences", [])  # type: ignore

    if not persona or not optimized_days:
        state["error"] = "Missing persona or optimized experiences"
        return state

    # Fetch weather
    start_date = persona.trip_start_date or datetime.now().strftime("%Y-%m-%d")
    weather_list = await fetch_weather(persona.destination, start_date, persona.trip_days)

    # Build each day
    day_plans = []
    for day_idx, day_exps in enumerate(optimized_days):
        day_num = day_idx + 1
        weather = weather_list[day_idx] if day_idx < len(weather_list) else {}
        day_plan = _build_day_schedule(day_exps, persona, day_num, persona.trip_days, weather)
        if weather.get("date"):
            day_plan.date = weather["date"]
        else:
            trip_date = datetime.fromisoformat(start_date) + timedelta(days=day_idx)
            day_plan.date = trip_date.strftime("%Y-%m-%d")
        day_plans.append(day_plan)

    itinerary = Itinerary(
        session_id=state.get("session_id", ""),
        persona=persona,
        destination=persona.destination,
        days=day_plans,
        total_cost=round(sum(d.total_cost for d in day_plans), 2),
        status="planning",
        generation_log=list(state.get("log", [])),
    )

    state["itinerary"] = itinerary
    state["log"].append(f"📅 Itinerary assembled: {len(day_plans)} days")

    return state
