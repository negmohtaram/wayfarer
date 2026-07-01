import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.agents.graph import planning_graph
from src.agents.replan_graph import replan_graph
from src.agents.nodes.persona_builder import run_persona_turn, build_persona_from_result
from src.agents.state import PlanningState, ReplanningState
from src.models.persona import TravelerPersona
from src.models.itinerary import Itinerary
from src.config import SESSION_DIR, SUPPORTED_CITIES

router = APIRouter()


def _session_path(session_id: str) -> Path:
    return SESSION_DIR / f"{session_id}.json"


def _save_session(session_id: str, data: dict):
    path = _session_path(session_id)
    path.write_text(json.dumps(data, default=str, indent=2))


def _load_session(session_id: str) -> dict:
    path = _session_path(session_id)
    if not path.exists():
        raise HTTPException(404, f"Session {session_id} not found")
    return json.loads(path.read_text())


class PersonaTurnRequest(BaseModel):
    destination: str
    trip_days: int
    conversation_history: list[dict] = []
    user_message: str = ""
    session_id: str = ""


class PersonaTurnResponse(BaseModel):
    session_id: str
    complete: bool = False
    question: str = ""
    question_number: int = 0
    hint: str = ""
    persona: dict = {}
    persona_narrative: str = ""
    persona_tags: list[str] = []


class PlanRequest(BaseModel):
    session_id: str
    persona: dict
    stream: bool = True


class ReplanRequest(BaseModel):
    session_id: str
    disruption: str
    completed_day_indices: list[int] = []


@router.post("/persona/turn", response_model=PersonaTurnResponse)
async def persona_turn(req: PersonaTurnRequest):
    """Handle one turn of the persona-building conversation."""
    session_id = req.session_id or str(uuid.uuid4())

    result = await run_persona_turn(
        destination=req.destination,
        trip_days=req.trip_days,
        conversation_history=req.conversation_history,
        user_message=req.user_message,
    )

    response = PersonaTurnResponse(session_id=session_id)
    response.complete = result.get("complete", False)

    if response.complete:
        persona = build_persona_from_result(result, req.destination, req.trip_days)
        response.persona = persona.model_dump()
        response.persona_narrative = result.get("persona_narrative", "")
        response.persona_tags = persona.persona_tags()

        # Save partial session
        _save_session(session_id, {
            "session_id": session_id,
            "persona": persona.model_dump(),
            "status": "persona_complete",
        })
    else:
        response.question = result.get("question", "Tell me about your travel style...")
        response.question_number = result.get("question_number", 1)
        response.hint = result.get("hint", "")

    return response


async def _stream_planning(session_id: str, persona: TravelerPersona) -> AsyncGenerator[str, None]:
    """Stream planning pipeline progress as SSE events."""

    def _sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    yield _sse("start", {"message": "Starting WAYFARER planning pipeline...", "session_id": session_id})

    state: PlanningState = {
        "session_id": session_id,
        "persona": persona,
        "log": [],
        "revision_count": 0,
    }

    # Run graph step by step, streaming progress
    try:
        # We'll run the graph and stream intermediate states
        log_seen = set()

        async for chunk in planning_graph.astream(state):
            node_name = list(chunk.keys())[0] if chunk else None
            node_state = chunk.get(node_name, {}) if node_name else {}

            # Stream new log entries
            new_logs = node_state.get("log", [])
            for log_entry in new_logs:
                if log_entry not in log_seen:
                    log_seen.add(log_entry)
                    yield _sse("progress", {"message": log_entry, "node": node_name})

            # Stream critiques as they come in
            if node_name == "critic_panel":
                critiques = node_state.get("critiques", [])
                for critique in critiques:
                    if hasattr(critique, "model_dump"):
                        yield _sse("critique", critique.model_dump())

            # Stream final itinerary
            if node_name == "regret_minimizer":
                itinerary = node_state.get("itinerary")
                if itinerary and hasattr(itinerary, "model_dump"):
                    itin_data = itinerary.model_dump()
                    _save_session(session_id, {
                        "session_id": session_id,
                        "persona": persona.model_dump(),
                        "itinerary": itin_data,
                        "status": "complete",
                    })
                    yield _sse("complete", {"itinerary": itin_data})

        yield _sse("done", {"message": "Planning complete!"})

    except Exception as e:
        yield _sse("error", {"message": str(e)})


@router.post("/plan")
async def start_planning(req: PlanRequest):
    """Start trip planning — streams progress via SSE."""
    try:
        persona = TravelerPersona(**req.persona)
    except Exception as e:
        raise HTTPException(400, f"Invalid persona: {e}")

    if not req.session_id:
        raise HTTPException(400, "session_id required")

    return StreamingResponse(
        _stream_planning(req.session_id, persona),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get("/itinerary/{session_id}")
async def get_itinerary(session_id: str):
    """Retrieve the full itinerary for a session."""
    data = _load_session(session_id)
    return data


@router.get("/regret/{session_id}")
async def get_regret_rankings(session_id: str):
    """Get regret minimizer rankings for a session."""
    data = _load_session(session_id)
    itinerary = data.get("itinerary", {})
    return {
        "session_id": session_id,
        "rankings": itinerary.get("regret_rankings", []),
    }


async def _stream_replanning(session_id: str, disruption: str, completed: list[int]) -> AsyncGenerator[str, None]:
    """Stream replanning progress as SSE events."""

    def _sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    yield _sse("start", {"message": f"Replanning: {disruption}"})

    try:
        data = _load_session(session_id)
        itinerary_data = data.get("itinerary")
        persona_data = data.get("persona")

        if not itinerary_data or not persona_data:
            yield _sse("error", {"message": "Session data incomplete"})
            return

        persona = TravelerPersona(**persona_data)
        itinerary = Itinerary(**itinerary_data)

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

            new_logs = node_state.get("log", [])
            for log_entry in new_logs:
                yield _sse("progress", {"message": log_entry})

            if node_name == "replan_assembler":
                new_itinerary = node_state.get("itinerary")
                if new_itinerary and hasattr(new_itinerary, "model_dump"):
                    itin_data = new_itinerary.model_dump()
                    data["itinerary"] = itin_data
                    _save_session(session_id, data)
                    yield _sse("complete", {"itinerary": itin_data})

        yield _sse("done", {"message": "Replanning complete!"})

    except Exception as e:
        yield _sse("error", {"message": str(e)})


@router.post("/replan")
async def replan(req: ReplanRequest):
    """Handle a disruption and replan remaining days."""
    return StreamingResponse(
        _stream_replanning(req.session_id, req.disruption, req.completed_day_indices),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/cities")
async def list_cities():
    return {"cities": SUPPORTED_CITIES}


@router.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}
