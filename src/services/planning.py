import json
from typing import AsyncGenerator

from src.agents.graph import planning_graph
from src.agents.replan_graph import replan_graph
from src.agents.state import PlanningState, ReplanningState
from src.models.itinerary import Itinerary
from src.models.persona import TravelerPersona
from src.services.session import load_session, save_session


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def stream_planning(session_id: str, persona: TravelerPersona) -> AsyncGenerator[str, None]:
    yield _sse("start", {"message": "Starting WAYFARER planning pipeline...", "session_id": session_id})

    state: PlanningState = {
        "session_id": session_id,
        "persona": persona,
        "log": [],
        "revision_count": 0,
    }

    try:
        log_seen: set[str] = set()

        async for chunk in planning_graph.astream(state):
            node_name = list(chunk.keys())[0] if chunk else None
            node_state = chunk.get(node_name, {}) if node_name else {}

            for entry in node_state.get("log", []):
                if entry not in log_seen:
                    log_seen.add(entry)
                    yield _sse("progress", {"message": entry, "node": node_name})

            if node_name == "critic_panel":
                for critique in node_state.get("critiques", []):
                    if hasattr(critique, "model_dump"):
                        yield _sse("critique", critique.model_dump())

            if node_name == "regret_minimizer":
                itinerary = node_state.get("itinerary")
                if itinerary and hasattr(itinerary, "model_dump"):
                    itin_data = itinerary.model_dump()
                    await save_session(session_id, {
                        "session_id": session_id,
                        "persona": persona.model_dump(),
                        "itinerary": itin_data,
                        "status": "complete",
                    })
                    yield _sse("complete", {"itinerary": itin_data})

        yield _sse("done", {"message": "Planning complete!"})

    except Exception as e:
        yield _sse("error", {"message": str(e)})


async def stream_replanning(
    session_id: str,
    disruption: str,
    completed: list[int],
) -> AsyncGenerator[str, None]:
    yield _sse("start", {"message": f"Replanning: {disruption}"})

    try:
        data = await load_session(session_id)
        persona = TravelerPersona(**data["persona"])
        itinerary = Itinerary(**data["itinerary"])

        state: ReplanningState = {
            "session_id": session_id,
            "persona": persona,
            "itinerary": itinerary,
            "disruption": disruption,
            "completed_day_indices": completed,
            "log": [],
        }

        async for chunk in replan_graph.astream(state):
            node_name = list(chunk.keys())[0] if chunk else None
            node_state = chunk.get(node_name, {}) if node_name else {}

            for entry in node_state.get("log", []):
                yield _sse("progress", {"message": entry})

            if node_name == "replan_assembler":
                new_itinerary = node_state.get("itinerary")
                if new_itinerary and hasattr(new_itinerary, "model_dump"):
                    itin_data = new_itinerary.model_dump()
                    data["itinerary"] = itin_data
                    await save_session(session_id, data)
                    yield _sse("complete", {"itinerary": itin_data})

        yield _sse("done", {"message": "Replanning complete!"})

    except Exception as e:
        yield _sse("error", {"message": str(e)})
