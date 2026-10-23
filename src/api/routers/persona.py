import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.agents.nodes.persona_builder import build_persona_from_result, run_persona_turn
from src.config import SUPPORTED_CITIES
from src.services.session import save_session

router = APIRouter(tags=["persona"])


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
