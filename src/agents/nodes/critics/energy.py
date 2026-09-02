from src.models.itinerary import Itinerary, CritiqueResult
from src.optimization.energy import build_energy_curve, get_energy_at_time, time_to_minutes

# An experience is considered energy-mismatched if it demands more than this
# many units above the traveler's available energy at that time.
ENERGY_MISMATCH_THRESHOLD = 3.0


async def energy_critic(itinerary: Itinerary) -> CritiqueResult:
    issues = []
    suggestions = []
    score = 10.0
    affected_days = []

    persona = itinerary.persona
    curve = build_energy_curve(persona.energy_profile)

    for day in itinerary.days:
        total_experiences = len(day.experiences)
        if total_experiences == 0:
            continue

        if persona.pace >= 7 and total_experiences > 4:
            issues.append(
                f"Day {day.day_number}: {total_experiences} experiences is too many for a slow-paced traveler"
            )
            suggestions.append(f"Day {day.day_number}: Reduce to 3-4 experiences with more breathing room")
            score -= 1.5
            if day.day_number not in affected_days:
                affected_days.append(day.day_number)

        if persona.pace <= 3 and total_experiences < 4:
            issues.append(f"Day {day.day_number}: Only {total_experiences} experiences for a fast-paced traveler")
            suggestions.append(f"Day {day.day_number}: Add 1-2 more experiences")
            score -= 0.5

        for sched in day.experiences:
            exp = sched.experience
            traveler_energy = get_energy_at_time(sched.start_time, curve)

            # Flag experiences that demand significantly more energy than the
            # traveler has available at that time.
            energy_mismatch = exp.energy_required - traveler_energy
            if energy_mismatch > ENERGY_MISMATCH_THRESHOLD:
                issues.append(
                    f"Day {day.day_number}: '{exp.name}' requires energy {exp.energy_required:.0f}/10 "
                    f"but traveler energy at {sched.start_time} is {traveler_energy:.0f}/10"
                )
                suggestions.append(f"Move '{exp.name}' to a time when traveler energy is higher")
                score -= 0.8
                if day.day_number not in affected_days:
                    affected_days.append(day.day_number)

            # Post-lunch dip: 12-14h is low energy for most profiles
            start_min = time_to_minutes(sched.start_time)
            if 720 <= start_min <= 840 and exp.energy_required >= 7:
                issues.append(
                    f"Day {day.day_number}: High-energy '{exp.name}' scheduled during "
                    f"typical post-lunch dip (12-2pm)"
                )
                suggestions.append(f"Move '{exp.name}' to morning or late afternoon")
                score -= 1.0
                if day.day_number not in affected_days:
                    affected_days.append(day.day_number)

        if persona.physical_constraints:
            for sched in day.experiences:
                exp = sched.experience
                if "bad_knee" in persona.physical_constraints and exp.physical_intensity == "high":
                    issues.append(
                        f"Day {day.day_number}: '{exp.name}' has high physical intensity "
                        f"but traveler has bad knee"
                    )
                    score -= 2.0
                    if day.day_number not in affected_days:
                        affected_days.append(day.day_number)

    score = max(0.0, score)
    verdict = "approve" if score >= 7 else ("revise" if score >= 4 else "reject_day")

    return CritiqueResult(
        critic_type="energy",
        score=round(score, 1),
        issues=issues,
        suggestions=suggestions if suggestions else ["Energy pacing matches traveler profile well"],
        verdict=verdict,
        affected_days=affected_days,
    )
