import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.agents.nodes.persona_builder import build_persona_from_result, run_persona_turn
from src.config import SUPPORTED_CITIES
from src.models.persona import TravelerPersona
from src.services.planning import stream_planning, stream_replanning
from src.services.session import load_session, save_session

router = APIRouter()

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}


# --- Request / Response models ---

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


# --- Routes ---

@router.post("/persona/turn", response_model=PersonaTurnResponse)
async def persona_turn(req: PersonaTurnRequest):
    if req.destination.lower() not in SUPPORTED_CITIES:
        raise HTTPException(
            400,
            f"Unsupported destination '{req.destination}'. Supported cities: {', '.join(SUPPORTED_CITIES)}",
        )
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
        await save_session(session_id, {
            "session_id": session_id,
            "persona": persona.model_dump(),
            "status": "persona_complete",
        })
    else:
        response.question = result.get("question", "Tell me about your travel style...")
        response.question_number = result.get("question_number", 1)
        response.hint = result.get("hint", "")

    return response


@router.post("/plan")
async def start_planning(req: PlanRequest):
    if not req.session_id:
        raise HTTPException(400, "session_id required")
    try:
        persona = TravelerPersona(**req.persona)
    except Exception as e:
        raise HTTPException(400, f"Invalid persona: {e}")

    return StreamingResponse(
        stream_planning(req.session_id, persona),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@router.post("/replan")
async def replan(req: ReplanRequest):
    return StreamingResponse(
        stream_replanning(req.session_id, req.disruption, req.completed_day_indices),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@router.get("/itinerary/{session_id}")
async def get_itinerary(session_id: str):
    return await load_session(session_id)


@router.get("/regret/{session_id}")
async def get_regret_rankings(session_id: str):
    data = await load_session(session_id)
    return {
        "session_id": session_id,
        "rankings": data.get("itinerary", {}).get("regret_rankings", []),
    }


@router.get("/cities")
async def list_cities():
    return {"cities": SUPPORTED_CITIES}


@router.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}
