from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.llm import call_with_retry, get_llm
from src.evaluation.recording import note_fallback
from src.config import GROQ_MODEL
from src.models.itinerary import CritiqueResult, Itinerary


async def experience_critic(itinerary: Itinerary) -> CritiqueResult:
    """Evaluate experience quality and narrative arc using LLM judgment.

    Why LLM here instead of rules?
    Rules can check "is there variety?" by counting types. They can't judge
    whether the itinerary tells a coherent story, whether the emotional arc
    builds well, or whether the final experience is truly memorable. Those
    are qualitative judgments — exactly what LLMs are good at.

    with_structured_output(CritiqueResult) tells LangChain to use the model's
    function-calling feature to return data in exactly the shape of CritiqueResult.
    No manual json.loads() or parsing — Pydantic validates it directly.
    """
    persona = itinerary.persona

    days_summary = []
    for day in itinerary.days:
        names = [s.experience.name for s in day.experiences]
        types = [s.experience.experience_type for s in day.experiences]
        local_scores = [s.experience.local_score for s in day.experiences]
        avg_score = sum(s.experience.persona_score or 0 for s in day.experiences) / max(len(day.experiences), 1)
        days_summary.append(
            f"Day {day.day_number} ({day.arc_label}): {', '.join(names)}\n"
            f"  Types: {', '.join(types)} | Avg local score: {sum(local_scores)/max(len(local_scores),1):.1f} "
            f"| Avg fit score: {avg_score:.2f}"
        )

    prompt = f"""You are evaluating a {persona.trip_days}-day itinerary for {persona.destination.title()}.

Traveler: {persona.social_context}, pace {persona.pace}/10 (10=very slow),
food {persona.food_importance}/10, culture {persona.culture_importance}/10,
local preference {persona.local_vs_tourist}/10 (10=very local).

Itinerary:
{chr(10).join(days_summary)}

Evaluate on these dimensions:
1. Variety — is there a healthy mix of experience types or is it repetitive?
2. Emotional arc — does intensity/excitement build toward a peak day and land well?
3. Narrative coherence — does each day feel like a chapter with a theme?
4. Ending quality — does the final day close on something memorable?
5. Persona fit — do the experience types match what this traveler actually wants?

Return critic_type as exactly "experience".
Score 0-10. Verdict must be "approve" (score>=7), "revise" (4-7), or "reject_day" (<4).
List specific issues and concrete suggestions. affected_days should be day numbers with problems."""

    try:
        llm = get_llm(temperature=0.3).with_structured_output(CritiqueResult)
        messages = [
            SystemMessage("You are a precise travel itinerary critic. Return structured evaluations."),
            HumanMessage(prompt),
        ]
        result = await call_with_retry(
            llm.ainvoke, messages,
            meta={"site": "experience_critic", "model": GROQ_MODEL, "temperature": 0.3, "schema": "CritiqueResult"},
        )
        return result
    except Exception as exc:
        note_fallback("experience_critic", exc)
        return _fallback_critique(itinerary)


def _fallback_critique(itinerary: Itinerary) -> CritiqueResult:
    """Simple rule-based fallback used only when the LLM call fails."""
    issues = []
    score = 10.0
    for day in itinerary.days:
        types = [s.experience.experience_type for s in day.experiences]
        cultural_count = types.count("cultural")
        if cultural_count >= 4:
            issues.append(f"Day {day.day_number}: {cultural_count} cultural experiences — museum fatigue risk")
            score -= 1.5
    verdict = "approve" if score >= 7 else ("revise" if score >= 4 else "reject_day")
    return CritiqueResult(
        critic_type="experience",
        score=round(score, 1),
        issues=issues,
        suggestions=["Review for variety"] if issues else ["Strong variety"],
        verdict=verdict,
        affected_days=[],
    )
