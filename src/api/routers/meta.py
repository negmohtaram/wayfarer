from datetime import datetime

from fastapi import APIRouter

from src.config import SUPPORTED_CITIES

router = APIRouter(tags=["meta"])


@router.get("/cities")
async def list_cities():
    return {"cities": SUPPORTED_CITIES}


@router.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}
