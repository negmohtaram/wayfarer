import json
from pathlib import Path

from fastapi import HTTPException

from src.config import SESSION_DIR


def _path(session_id: str) -> Path:
    return SESSION_DIR / f"{session_id}.json"


def save_session(session_id: str, data: dict) -> None:
    _path(session_id).write_text(json.dumps(data, default=str, indent=2))


def load_session(session_id: str) -> dict:
    path = _path(session_id)
    if not path.exists():
        raise HTTPException(404, f"Session {session_id} not found")
    return json.loads(path.read_text())
