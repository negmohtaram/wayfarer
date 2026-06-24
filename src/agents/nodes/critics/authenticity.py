from src.models.itinerary import Itinerary, CritiqueResult


async def authenticity_critic(itinerary: Itinerary) -> CritiqueResult:
    issues = []
    suggestions = []
    score = 10.0
    affected_days = []

    persona = itinerary.persona
    target_local_score = persona.local_vs_tourist

    for day in itinerary.days:
        if not day.experiences:
            continue

        avg_local = sum(s.experience.local_score for s in day.experiences) / len(day.experiences)

        # For local-seeking travelers, flag tourist-heavy days
        if target_local_score >= 6 and avg_local < 4:
            issues.append(
                f"Day {day.day_number}: Average local score {avg_local:.1f}/10 — too tourist-heavy "
                f"for a traveler seeking local experiences"
            )
            suggestions.append(
                f"Day {day.day_number}: Replace 1-2 tourist spots with neighborhood experiences"
            )
            score -= 2.0
            if day.day_number not in affected_days:
                affected_days.append(day.day_number)

        # For tourist-friendly travelers, flag missing iconic experiences
        elif target_local_score <= 3 and avg_local > 7:
            issues.append(
                f"Day {day.day_number}: Very local-focused day for a tourist-experience seeker"
            )
            suggestions.append(
                f"Day {day.day_number}: Include at least one iconic landmark experience"
            )
            score -= 1.0

        # Detect tourist traps (local_score <= 2 is a red flag)
        for sched in day.experiences:
            exp = sched.experience
            if exp.local_score <= 2 and target_local_score >= 5:
                issues.append(
                    f"Day {day.day_number}: '{exp.name}' is a tourist trap "
                    f"(local score {exp.local_score}/10)"
                )
                suggestions.append(f"Replace '{exp.name}' with a more authentic alternative")
                score -= 1.5
                if day.day_number not in affected_days:
                    affected_days.append(day.day_number)

        # Check food authenticity
        food_exps = [s for s in day.experiences if s.experience.experience_type == "food"]
        if food_exps and persona.food_importance >= 6:
            avg_food_local = sum(s.experience.local_score for s in food_exps) / len(food_exps)
            if avg_food_local < 5:
                issues.append(
                    f"Day {day.day_number}: Food experiences skew tourist (avg local score {avg_food_local:.1f})"
                )
                suggestions.append(f"Day {day.day_number}: Seek out more locally loved food spots")
                score -= 0.5

    score = max(0.0, score)
    verdict = "approve" if score >= 7 else ("revise" if score >= 4 else "reject_day")

    return CritiqueResult(
        critic_type="authenticity",
        score=round(score, 1),
        issues=issues,
        suggestions=suggestions if suggestions else ["Good balance of authentic and accessible experiences"],
        verdict=verdict,
        affected_days=affected_days,
    )
