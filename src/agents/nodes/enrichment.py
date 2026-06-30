from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from src.agents.llm import get_llm
from src.agents.state import PlanningState
from src.agents.tools.search import web_search


async def enrichment_node(state: PlanningState) -> PlanningState:
    """Enrich the top-scored experiences with live web search data.

    How tool calling works here:
    1. We bind the search tool to the LLM so it knows the tool exists.
    2. We ask the LLM to find current info about an experience.
    3. The LLM responds with tool_calls instead of plain text — it's saying
       "I want to call web_search with this query."
    4. WE execute the actual search (the LLM never touches the internet directly).
    5. We feed the result back as a ToolMessage and let the LLM summarize.
    """
    experience_pool = state.get("experience_pool", [])
    persona = state.get("persona")
    if not experience_pool or not persona:
        return state

    state.setdefault("log", [])
    state["log"].append("🔍 Enriching top experiences with live search...")

    # Only enrich top 3 — keeps latency low and avoids rate limits
    top = experience_pool[:3]
    llm_with_tools = get_llm(temperature=0).bind_tools([web_search])
    summarizer = get_llm(temperature=0)

    for exp in top:
        try:
            messages = [
                SystemMessage(
                    "You help travelers by finding current practical information. "
                    "Use web_search to look up recent updates about this travel experience. "
                    "Focus on anything that might have changed in 2025-2026: hours, prices, closures, new highlights."
                ),
                HumanMessage(
                    f"Find current practical info for: '{exp.name}' in {persona.destination.title()}."
                ),
            ]

            # Step 1: ask the LLM — it may respond with a tool_call
            response = await llm_with_tools.ainvoke(messages)

            if not response.tool_calls:
                # LLM chose not to search — nothing to update
                continue

            # Step 2: execute the tool the LLM requested
            tool_call = response.tool_calls[0]
            search_result = await web_search.ainvoke(tool_call["args"])

            # Step 3: give the LLM the search result so it can synthesize an answer
            messages.append(response)  # the AIMessage that contained tool_calls
            messages.append(ToolMessage(
                content=str(search_result),
                tool_call_id=tool_call["id"],
            ))
            final = await summarizer.ainvoke(messages)

            update = final.content.strip()
            if update and len(update) > 20:
                exp.practical_tips = exp.practical_tips + f"\n\n[Current info, 2026]: {update[:400]}"

        except Exception:
            # Search failures are non-fatal — keep the original static tips
            continue

    state["experience_pool"] = experience_pool
    state["log"].append(f"🔍 Live enrichment done for top {len(top)} experiences")
    return state
