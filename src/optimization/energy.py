from src.models.experience import Experience
from src.models.persona import TravelerPersona


TIME_SLOTS = [
    ("06:00", "early_morning"),
    ("08:00", "morning"),
    ("10:00", "morning"),
    ("12:00", "afternoon"),
    ("14:00", "afternoon"),
    ("16:00", "golden_hour"),
    ("18:00", "golden_hour"),
    ("20:00", "evening"),
    ("22:00", "night"),
]


def build_energy_curve(energy_profile: str) -> dict[str, float]:
    """Returns energy level (0-10) at each time of day."""
    curves = {
        "morning_person": {
            "06:00": 9.0, "08:00": 9.5, "10:00": 8.5,
            "12:00": 7.0, "14:00": 5.0, "16:00": 6.5,
            "18:00": 6.0, "20:00": 4.0, "22:00": 2.0,
        },
        "evening_person": {
            "06:00": 3.0, "08:00": 5.0, "10:00": 6.5,
            "12:00": 7.0, "14:00": 7.5, "16:00": 8.0,
            "18:00": 9.0, "20:00": 9.5, "22:00": 8.0,
        },
        "consistent": {
            "06:00": 6.0, "08:00": 7.5, "10:00": 8.0,
            "12:00": 7.0, "14:00": 6.5, "16:00": 7.5,
            "18:00": 7.0, "20:00": 6.0, "22:00": 4.0,
        },
    }
    return curves.get(energy_profile, curves["consistent"])


def time_to_minutes(t: str) -> int:
    h, m = t.split(":")
    return int(h) * 60 + int(m)


def minutes_to_time(m: int) -> str:
    m = min(m, 23 * 60 + 59)  # clamp — scheduler catches overflows via logistics critic
    h = m // 60
    mins = m % 60
    return f"{h:02d}:{mins:02d}"


def get_energy_at_time(time_str: str, curve: dict[str, float]) -> float:
    """Interpolate energy level at a given time."""
    t = time_to_minutes(time_str)
    times = [(time_to_minutes(k), v) for k, v in curve.items()]
    times.sort()

    if t <= times[0][0]:
        return times[0][1]
    if t >= times[-1][0]:
        return times[-1][1]

    for i in range(len(times) - 1):
        t1, e1 = times[i]
        t2, e2 = times[i + 1]
        if t1 <= t <= t2:
            ratio = (t - t1) / (t2 - t1)
            return e1 + ratio * (e2 - e1)

    return 5.0


def sequence_by_energy(
    experiences: list[Experience],
    persona: TravelerPersona,
    start_time_minutes: int = 480,  # 8am default
) -> list[Experience]:
    """
    Sequence experiences so high-energy activities align with the traveler's
    peak energy periods. Preserves geographic order as much as possible
    while optimizing energy match.
    """
    if not experiences:
        return []

    curve = build_energy_curve(persona.energy_profile)

    # Score how well each experience fits each time slot
    num_slots = len(experiences)
    slot_times = []
    for i in range(num_slots):
        slot_min = start_time_minutes + i * 90  # rough 90-min slots
        slot_times.append(minutes_to_time(slot_min))

    # Use a greedy approach: assign most energy-demanding to best energy slots
    sorted_by_energy = sorted(experiences, key=lambda e: e.energy_required, reverse=True)
    slot_energies = [(t, get_energy_at_time(t, curve)) for t in slot_times]
    slot_energies_sorted = sorted(slot_energies, key=lambda x: x[1], reverse=True)

    # Map high-energy experiences to high-energy slots
    assignment: dict[str, Experience] = {}
    for i, (slot_time, _) in enumerate(slot_energies_sorted):
        if i < len(sorted_by_energy):
            assignment[slot_time] = sorted_by_energy[i]

    # Return in time order
    result = [assignment[t] for t, _ in sorted(slot_energies, key=lambda x: time_to_minutes(x[0]))]
    return result


def get_day_start_time(persona: TravelerPersona) -> int:
    """Returns day start time in minutes based on energy profile."""
    if persona.energy_profile == "morning_person":
        return 7 * 60  # 7am
    elif persona.energy_profile == "evening_person":
        return 9 * 60  # 9am
    else:
        return 8 * 60  # 8am


def time_of_day_for_slot(minutes_from_midnight: int) -> str:
    """Returns time-of-day label for a given minute offset."""
    hour = minutes_from_midnight // 60
    if hour < 7:
        return "early_morning"
    elif hour < 10:
        return "morning"
    elif hour < 13:
        return "morning"
    elif hour < 17:
        return "afternoon"
    elif hour < 19:
        return "golden_hour"
    elif hour < 21:
        return "evening"
    else:
        return "night"
