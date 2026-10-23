from fastapi import APIRouter

from src.services.session import load_session

router = APIRouter(tags=["itineraries"])


@router.get("/itinerary/{session_id}")
async def get_itinerary(session_id: str):
    return await load_session(session_id)


@router.get("/highlights/{session_id}")
async def get_highlight_rankings(session_id: str):
    data = await load_session(session_id)
    return {
        "session_id": session_id,
        "rankings": data.get("itinerary", {}).get("highlight_rankings", []),
    }
