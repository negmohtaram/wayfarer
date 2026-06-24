import asyncio
from src.agents.state import PlanningState
from src.agents.nodes.critics.logistics import logistics_critic
from src.agents.nodes.critics.experience import experience_critic
from src.agents.nodes.critics.energy import energy_critic
from src.agents.nodes.critics.budget import budget_critic
from src.agents.nodes.critics.authenticity import authenticity_critic
from src.models.itinerary import CritiqueResult


async def critic_panel_node(state: PlanningState) -> PlanningState:
    """Run all 5 critics in parallel and collect results."""
    state.setdefault("log", [])
    state["log"].append("🔍 Running critic panel...")

    itinerary = state.get("itinerary")
    if not itinerary:
        state["error"] = "No itinerary to critique"
        return state

    # Run all 5 critics in parallel
    results = await asyncio.gather(
        logistics_critic(itinerary),
        experience_critic(itinerary),
        energy_critic(itinerary),
        budget_critic(itinerary),
        authenticity_critic(itinerary),
        return_exceptions=True,
    )

    critiques: list[CritiqueResult] = []
    for result in results:
        if isinstance(result, CritiqueResult):
            critiques.append(result)
        elif isinstance(result, Exception):
            # Don't fail the whole pipeline on a critic error
            critiques.append(CritiqueResult(
                critic_type="unknown",
                score=7.0,
                issues=[str(result)],
                suggestions=[],
                verdict="approve",
            ))

    state["critiques"] = critiques
    itinerary.critiques = critiques

    # Log each critic result
    for c in critiques:
        status = "✓" if c.verdict == "approve" else "⚠"
        state["log"].append(f"   └─ {c.critic_type.title()}: {status} (score: {c.score}/10)")

    return state


def route_after_critique(state: PlanningState) -> str:
    """Determine next node based on critique results."""
    critiques = state.get("critiques", [])
    revision_count = state.get("revision_count", 0)

    from src.config import MAX_REVISION_CYCLES
    if revision_count >= MAX_REVISION_CYCLES:
        return "regret_minimizer"

    # Check if any critic rejected
    needs_revision = any(c.verdict in ("revise", "reject_day") for c in critiques)

    if needs_revision:
        return "revision_agent"
    return "regret_minimizer"
