from typing import Any, Callable

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.api.deps import get_llm_factory
from src.models.persona import TravelerPersona
from src.services.planning import stream_planning, stream_replanning

router = APIRouter(tags=["planning"])

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}


class PlanRequest(BaseModel):
    session_id: str
    persona: dict
    stream: bool = True


class ReplanRequest(BaseModel):
    session_id: str
    disruption: str
    completed_day_indices: list[int] = []


@router.post("/plan")
async def start_planning(req: PlanRequest, llm_factory: Callable[..., Any] = Depends(get_llm_factory)):
    if not req.session_id:
        raise HTTPException(400, "session_id required")
    try:
        persona = TravelerPersona(**req.persona)
    except Exception as e:
        raise HTTPException(400, f"Invalid persona: {e}")

    return StreamingResponse(
        stream_planning(req.session_id, persona, llm_factory),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@router.post("/replan")
async def replan(req: ReplanRequest, llm_factory: Callable[..., Any] = Depends(get_llm_factory)):
    return StreamingResponse(
        stream_replanning(req.session_id, req.disruption, req.completed_day_indices, llm_factory),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
