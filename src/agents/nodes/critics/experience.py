from src.models.itinerary import Itinerary, CritiqueResult


async def experience_critic(itinerary: Itinerary) -> CritiqueResult:
    issues = []
    suggestions = []
    score = 10.0
    affected_days = []

    total_days = len(itinerary.days)

    for day in itinerary.days:
        types = [s.experience.experience_type for s in day.experiences]
        type_set = set(types)

        # Check variety — no more than 3 of same type
        type_counts = {t: types.count(t) for t in type_set}
        for exp_type, count in type_counts.items():
            if exp_type == "cultural" and count >= 4:
                issues.append(f"Day {day.day_number}: {count} cultural experiences in a row — museum fatigue risk")
                suggestions.append(f"Day {day.day_number}: Replace one cultural visit with a food or walking experience")
                score -= 1.5
                if day.day_number not in affected_days:
                    affected_days.append(day.day_number)

        # Check for at least one memorable/peak experience
        has_high_local = any(s.experience.local_score >= 7 for s in day.experiences)
        has_good_description = any(len(s.experience.why_this_matters) > 50 for s in day.experiences)
        if not has_high_local and not has_good_description:
            issues.append(f"Day {day.day_number}: Lacks a memorable anchor experience")
            suggestions.append(f"Day {day.day_number}: Add at least one experience with a compelling story")
            score -= 1.0

        # Check emotional arc — peak day should be middle-ish
        if total_days >= 3:
            peak_day = total_days // 2 + 1
            if day.day_number == peak_day:
                avg_score = sum(
                    s.experience.persona_score or 0 for s in day.experiences
                ) / max(len(day.experiences), 1)
                if avg_score < 0.6:
                    issues.append(f"Day {day.day_number} is the peak day but has low-scoring experiences")
                    suggestions.append("Schedule the best experiences on the peak day")
                    score -= 1.0
                    if day.day_number not in affected_days:
                        affected_days.append(day.day_number)

        # Check last day ends memorably
        if day.day_number == total_days and day.experiences:
            last_exp = day.experiences[-1].experience
            if last_exp.experience_type not in ("scenic", "food", "serendipity") or last_exp.romantic_score < 6:
                issues.append(f"Day {day.day_number}: Itinerary doesn't end on a high note")
                suggestions.append("End the final day with something scenic or emotionally resonant")
                score -= 0.5

    score = max(0.0, score)
    verdict = "approve" if score >= 7 else ("revise" if score >= 4 else "reject_day")

    return CritiqueResult(
        critic_type="experience",
        score=round(score, 1),
        issues=issues,
        suggestions=suggestions if suggestions else ["Strong emotional arc and variety"],
        verdict=verdict,
        affected_days=affected_days,
    )
