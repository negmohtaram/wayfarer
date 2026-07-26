from src.models.itinerary import Itinerary, CritiqueResult
from src.optimization.energy import time_to_minutes


OPENING_HOURS = {
    "early_morning": (6 * 60, 9 * 60),
    "morning": (8 * 60, 13 * 60),
    "afternoon": (12 * 60, 18 * 60),
    "golden_hour": (17 * 60, 20 * 60),
    "evening": (18 * 60, 23 * 60),
    "night": (21 * 60, 24 * 60),
    "any": (6 * 60, 23 * 60),
}


async def logistics_critic(itinerary: Itinerary) -> CritiqueResult:
    issues = []
    suggestions = []
    score = 10.0
    affected_days = []

    for day in itinerary.days:
        for sched_exp in day.experiences:
            start_mins = time_to_minutes(sched_exp.start_time)
            exp = sched_exp.experience

            # Check timing vs best_time_of_day
            valid_window = OPENING_HOURS.get(exp.best_time_of_day, (8 * 60, 22 * 60))
            if not (valid_window[0] <= start_mins <= valid_window[1]):
                issues.append(
                    f"Day {day.day_number}: '{exp.name}' scheduled at {sched_exp.start_time} "
                    f"but best at {exp.best_time_of_day}"
                )
                score -= 1.0
                if day.day_number not in affected_days:
                    affected_days.append(day.day_number)

            # Check no meal gap > 5 hours
            meal_types = {"food"}
            food_times = [
                time_to_minutes(s.start_time)
                for s in day.experiences
                if s.experience.experience_type == "food"
            ]
            if food_times:
                max_gap = max(
                    food_times[i + 1] - food_times[i]
                    for i in range(len(food_times) - 1)
                ) if len(food_times) > 1 else 0
                if max_gap > 5 * 60:
                    issues.append(f"Day {day.day_number}: Long gap ({max_gap // 60}h) between meals")
                    suggestions.append(f"Day {day.day_number}: Add a food stop or snack break")
                    score -= 0.5
                    if day.day_number not in affected_days:
                        affected_days.append(day.day_number)

            # Check travel time realism
            if sched_exp.travel_time_from_prev > 45:
                issues.append(
                    f"Day {day.day_number}: Long travel ({sched_exp.travel_time_from_prev} min) to '{exp.name}'"
                )
                suggestions.append(f"Day {day.day_number}: Consider grouping experiences geographically")
                score -= 0.5

        # Check day doesn't end past midnight
        if day.experiences:
            last = day.experiences[-1]
            end_mins = time_to_minutes(last.end_time)
            if end_mins > 24 * 60:
                issues.append(f"Day {day.day_number}: Schedule runs past midnight")
                score -= 1.0
                if day.day_number not in affected_days:
                    affected_days.append(day.day_number)

    score = max(0.0, score)
    verdict = "approve" if score >= 7 else ("revise" if score >= 4 else "reject_day")

    return CritiqueResult(
        critic_type="logistics",
        score=round(score, 1),
        issues=issues,
        suggestions=suggestions if suggestions else ["Logistics look solid"],
        verdict=verdict,
        affected_days=affected_days,
    )
