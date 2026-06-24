from src.models.itinerary import Itinerary, CritiqueResult


async def budget_critic(itinerary: Itinerary) -> CritiqueResult:
    issues = []
    suggestions = []
    score = 10.0
    affected_days = []

    persona = itinerary.persona
    daily_budget = persona.budget_per_day

    for day in itinerary.days:
        if day.total_cost > daily_budget * 1.2:
            over = day.total_cost - daily_budget
            issues.append(
                f"Day {day.day_number}: Estimated cost €{day.total_cost:.0f} exceeds daily budget "
                f"€{daily_budget} by €{over:.0f}"
            )
            suggestions.append(
                f"Day {day.day_number}: Consider free alternatives or skip one paid activity"
            )
            score -= 1.5
            if day.day_number not in affected_days:
                affected_days.append(day.day_number)

        elif day.total_cost < daily_budget * 0.3 and daily_budget > 50:
            issues.append(
                f"Day {day.day_number}: Very low spend (€{day.total_cost:.0f}) — may be too bare"
            )
            suggestions.append(f"Day {day.day_number}: Consider adding a special meal or experience")
            score -= 0.3

    total = itinerary.total_cost
    total_budget = daily_budget * persona.trip_days
    if total > total_budget * 1.3:
        issues.append(f"Total trip cost €{total:.0f} is significantly over total budget €{total_budget}")
        suggestions.append("Review the most expensive days and find budget alternatives")
        score -= 1.0

    score = max(0.0, score)
    verdict = "approve" if score >= 7 else ("revise" if score >= 4 else "reject_day")

    return CritiqueResult(
        critic_type="budget",
        score=round(score, 1),
        issues=issues,
        suggestions=suggestions if suggestions else [f"Budget on track: ~€{itinerary.total_cost:.0f} total"],
        verdict=verdict,
        affected_days=affected_days,
    )
