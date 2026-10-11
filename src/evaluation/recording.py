"""Record model and weather calls once, replay them exactly.

Usage:
    with recording("record", Path("eval_results/recordings/baseline_v1")):
        ...run scenarios against the live model...
    with recording("replay", Path("eval_results/recordings/baseline_v1")):
        ...same scenarios, no network, identical results...

Each call is stored under a key made from what determines the answer (call site, model,
temperature, output schema, messages) plus an occurrence counter, so the same request made
twice in one run (e.g. the critic seeing the same itinerary again) keeps both answers.
In replay mode a missing recording raises ReplayMiss; nothing falls back to the network.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Awaitable, Callable

from pydantic import BaseModel

_ACTIVE: ContextVar["Recorder | None"] = ContextVar("recorder", default=None)


class ReplayMiss(LookupError):
    """No recording for this call. Record again on purpose."""


def active() -> "Recorder | None":
    return _ACTIVE.get()


def _messages_to_json(messages: Any) -> Any:
    if messages is None:
        return None
    return [{"role": getattr(m, "type", "unknown"), "content": getattr(m, "content", str(m))}
            for m in messages]


def _encode(result: Any) -> dict:
    if isinstance(result, BaseModel):
        cls = type(result)
        return {"kind": "pydantic", "cls": f"{cls.__module__}:{cls.__qualname__}",
                "data": result.model_dump(mode="json")}
    if hasattr(result, "content") and hasattr(result, "type"):          # LangChain message
        return {"kind": "message", "content": result.content,
                "tool_calls": getattr(result, "tool_calls", []) or []}
    return {"kind": "json", "data": result}


def _decode(stored: dict) -> Any:
    if stored["kind"] == "pydantic":
        module, name = stored["cls"].split(":")
        cls = importlib.import_module(module)
        for part in name.split("."):
            cls = getattr(cls, part)
        return cls.model_validate(stored["data"])
    if stored["kind"] == "message":
        from langchain_core.messages import AIMessage
        return AIMessage(content=stored["content"], tool_calls=stored["tool_calls"])
    return stored["data"]


class Recorder:
    def __init__(self, mode: str, folder: Path):
        if mode not in ("record", "replay"):
            raise ValueError(mode)
        self.mode, self.folder = mode, Path(folder)
        self.seen: dict[str, int] = {}
        self.calls = 0                         # real calls made (record mode)
        self.replayed = 0

    def key(self, meta: dict, messages: Any) -> str:
        blob = json.dumps({"meta": meta, "messages": _messages_to_json(messages)},
                          sort_keys=True, default=str)
        base = hashlib.sha256(blob.encode()).hexdigest()[:16]
        n = self.seen.get(base, 0) + 1
        self.seen[base] = n
        return f"{meta.get('site', 'call')}-{base}-{n}"

    async def call(self, meta: dict, messages: Any, live: Callable[[], Awaitable[Any]]) -> Any:
        key = self.key(meta, messages)
        path = self.folder / f"{key}.json"
        if self.mode == "replay":
            if not path.exists():
                raise ReplayMiss(f"no recording {path.name} for {meta}")
            self.replayed += 1
            return _decode(json.loads(path.read_text())["result"])
        result = await live()
        self.calls += 1
        self.folder.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"meta": meta, "messages": _messages_to_json(messages),
                                    "result": _encode(result)}, indent=2, default=str))
        return result


@contextlib.contextmanager
def recording(mode: str, folder: Path):
    recorder = Recorder(mode, folder)
    token = _ACTIVE.set(recorder)
    try:
        yield recorder
    finally:
        _ACTIVE.reset(token)
