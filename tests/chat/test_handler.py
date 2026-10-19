import asyncio
import time

from src.chat.handler import TurnSteps, handle_turn, handle_turn_sequential
from src.chat.router import Intent, Route


def make_steps(route: Route, calls: list, delay: float = 0.05) -> TurnSteps:
    async def load_session(session_id):
        calls.append("load")
        return {"trip_days": 4, "destination": "lisbon"}

    async def route_step(text, trip_days):
        assert calls == ["load"], "the router needs the trip length, so the trip loads first"
        calls.append("route")
        return route

    async def extract(text, r, trip):
        calls.append("extract")
        await asyncio.sleep(delay)
        return {"setting": "budget"}

    async def scope(r, trip):
        calls.append("scope")
        return {"day": r.day}

    async def weather(city, sc):
        calls.append("weather")
        await asyncio.sleep(delay)
        return {"day": sc["day"], "rain": False}

    return TurnSteps(load_session, route_step, extract, scope, weather)


def test_change_and_place_run_together():
    route = Route(intent=Intent.CHANGE_SETTING, day=2)
    start = time.perf_counter()
    result = asyncio.run(handle_turn("cheaper day two", "s", make_steps(route, [])))
    together = time.perf_counter() - start
    start = time.perf_counter()
    asyncio.run(handle_turn_sequential("cheaper day two", "s", make_steps(route, [])))
    one_by_one = time.perf_counter() - start
    assert result.change == {"setting": "budget"} and result.weather["day"] == 2
    assert together < one_by_one * 0.75


def test_both_versions_return_the_same_result():
    for route in [Route(intent=Intent.CHANGE_SETTING, day=3), Route(intent=Intent.EXPLAIN, day=1),
                  Route(intent=Intent.FEEDBACK), Route(intent=Intent.WHAT_IF)]:
        a = asyncio.run(handle_turn("m", "s", make_steps(route, [], delay=0)))
        b = asyncio.run(handle_turn_sequential("m", "s", make_steps(route, [], delay=0)))
        assert a == b


def test_skips_steps_the_message_does_not_need():
    calls = []
    asyncio.run(handle_turn("nice", "s", make_steps(Route(intent=Intent.FEEDBACK), calls)))
    assert calls == ["load", "route"]
