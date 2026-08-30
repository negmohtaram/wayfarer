import asyncio
import json
import os
from collections import defaultdict
from pathlib import Path

import aiofiles
from fastapi import HTTPException

from src.config import SESSION_DIR


def _path(session_id: str) -> Path:
    return SESSION_DIR / f"{session_id}.json"


# One lock per session_id so concurrent requests for the same session
# serialise their writes without blocking unrelated sessions.
_session_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)


async def save_session(session_id: str, data: dict) -> None:
    """Write session data atomically.

    Writes to a .tmp file first, then uses os.replace() (atomic on POSIX)
    to swap it in.  The per-session lock prevents two concurrent writes from
    racing on the same tmp file.
    """
    async with _session_locks[session_id]:
        path = _path(session_id)
        tmp = path.with_suffix(".tmp")
        async with aiofiles.open(tmp, "w") as f:
            await f.write(json.dumps(data, default=str, indent=2))
        os.replace(str(tmp), str(path))


async def load_session(session_id: str) -> dict:
    path = _path(session_id)
    if not path.exists():
        raise HTTPException(404, f"Session {session_id} not found")
    async with _session_locks[session_id]:
        async with aiofiles.open(path) as f:
            return json.loads(await f.read())
