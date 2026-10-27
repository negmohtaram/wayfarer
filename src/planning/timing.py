"""Start and end times for one day's stops in a given order. The planner and the assembler both use
this, so the times the planner scores are the times the traveler sees."""
from dataclasses import dataclass

from src.models.experience import Experience
from src.models.persona import TravelerPersona
from src.optimization.energy import get_day_start_time
from src.optimization.route import haversine, travel_time_minutes

TIME_OF_DAY_WINDOWS = {
    "early_morning": (6 * 60, 8 * 60),
    "morning": (8 * 60, 12 * 60),
    "afternoon": (12 * 60, 17 * 60),
    "golden_hour": (17 * 60, 20 * 60),
    "evening": (19 * 60, 22 * 60),
    "night": (21 * 60, 23 * 60),
    "any": (8 * 60, 21 * 60),
}
WALK_LIMIT_MIN = 20      # longer walks become transit


@dataclass
class Slot:
    experience: Experience
    start: int               # minutes from midnight; may pass 24:00 (the logistics critic flags it)
    end: int
    travel: int              # minutes from the previous stop
    mode: str
    walk_km: float
    late: int                # minutes the start falls after its best time-of-day window


def time_day(ordered: list[Experience], persona: TravelerPersona) -> list[Slot]:
    now = get_day_start_time(persona)
    slots: list[Slot] = []
    prev = None
    for exp in ordered:
        window = TIME_OF_DAY_WINDOWS.get(exp.best_time_of_day, TIME_OF_DAY_WINDOWS["any"])
        now = max(now, window[0])                     # wait for the window to open
        travel, mode, walk_km = 0, "walk", 0.0
        if prev is not None:
            travel = travel_time_minutes(prev, exp, "walk")
            if travel > WALK_LIMIT_MIN:
                mode, travel = "transit", travel_time_minutes(prev, exp, "transit")
            else:
                walk_km = haversine(prev.location[0], prev.location[1], exp.location[0], exp.location[1])
        now += travel
        slots.append(Slot(exp, now, now + exp.duration_minutes, travel, mode, walk_km, max(0, now - window[1])))
        now += exp.duration_minutes
        if persona.pace >= 7:
            now += 30                                 # breathing room for slow travelers
        prev = exp
    return slots
