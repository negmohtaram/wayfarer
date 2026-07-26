import json
from pathlib import Path

import aiofiles
from fastapi import HTTPException

from src.config import SESSION_DIR


def _path(session_id: str) -> Path:
    return SESSION_DIR / f"{session_id}.json"


async def save_session(session_id: str, data: dict) -> None:
    async with aiofiles.open(_path(session_id), "w") as f:
        await f.write(json.dumps(data, default=str, indent=2))


async def load_session(session_id: str) -> dict:
    path = _path(session_id)
    if not path.exists():
        raise HTTPException(404, f"Session {session_id} not found")
    async with aiofiles.open(path) as f:
        return json.loads(await f.read())
