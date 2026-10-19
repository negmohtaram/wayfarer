"""One chat turn: load the trip, route the message, then work out the change and where it applies.

The router needs the trip length, so the trip loads first. After routing, the change itself
(extract) and the place it applies to (scope, then that day's weather) do not depend on each
other, so they run together.
"""
import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from src.chat.router import Intent, Route

NEEDS_CHANGE = {Intent.CHANGE_SETTING, Intent.WHAT_IF}


@dataclass
class TurnSteps:
    load_session: Callable[[str], Awaitable[dict]]
    route: Callable[[str, int], Awaitable[Route]]
    extract: Callable[[str, Route, dict], Awaitable[dict]]
    scope: Callable[[Route, dict], Awaitable[dict]]
    weather: Callable[[str, dict], Awaitable[Optional[dict]]]


@dataclass
class TurnResult:
    route: Route
    change: Optional[dict] = None
    scope: Optional[dict] = None
    weather: Optional[dict] = None
    steps: list[str] = field(default_factory=list)


async def _place(steps: TurnSteps, route: Route, trip: dict) -> tuple[Optional[dict], Optional[dict]]:
    if route.day is None:
        return None, None
    scope = await steps.scope(route, trip)
    return scope, await steps.weather(trip["destination"], scope)


async def _change(steps: TurnSteps, text: str, route: Route, trip: dict) -> Optional[dict]:
    if route.intent not in NEEDS_CHANGE:
        return None
    return await steps.extract(text, route, trip)


async def handle_turn(text: str, session_id: str, steps: TurnSteps) -> TurnResult:
    trip = await steps.load_session(session_id)
    route = await steps.route(text, trip["trip_days"])
    change, (scope, weather) = await asyncio.gather(
        _change(steps, text, route, trip),
        _place(steps, route, trip),
    )
    return TurnResult(route=route, change=change, scope=scope, weather=weather)


async def handle_turn_sequential(text: str, session_id: str, steps: TurnSteps) -> TurnResult:
    """Same steps one after another. Kept to measure what running them together saves."""
    trip = await steps.load_session(session_id)
    route = await steps.route(text, trip["trip_days"])
    change = await _change(steps, text, route, trip)
    scope, weather = await _place(steps, route, trip)
    return TurnResult(route=route, change=change, scope=scope, weather=weather)
