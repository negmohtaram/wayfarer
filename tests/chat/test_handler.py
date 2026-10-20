import asyncio
import time

from src.chat.handler import TurnRunner, TurnSteps, handle_turn, handle_turn_sequential
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


def test_new_message_cancels_the_running_turn():
    # Given: message A's change step waits forever; B's finishes at once.
    route = Route(intent=Intent.CHANGE_SETTING, day=2)
    steps = make_steps(route, [])
    started, cancelled, extract_calls = asyncio.Event(), [], []
    fast_extract = steps.extract

    async def extract(text, r, trip):
        extract_calls.append(text)
        if text == "A":
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.append(text)
                raise
        return await fast_extract(text, r, trip)

    async def route_step(text, trip_days):
        return route

    steps.extract, steps.route = extract, route_step    # two turns share one steps object
    delivered = []

    async def deliver(text, result):
        delivered.append(text)

    async def scenario():
        runner = TurnRunner("s", steps, deliver)
        a = await runner.submit("A")
        await started.wait()
        b = await runner.submit("B")     # When: B arrives while A is still running
        await b
        return a, b

    a, b = asyncio.run(scenario())
    # Then
    assert a.cancelled() and cancelled == ["A"]     # A stopped, it did not keep running
    assert delivered == ["B"]                       # A's result is never delivered
    assert b.result().change == {"setting": "budget"}
    assert extract_calls.count("A") == 1            # cancellation is never retried


def test_a_stale_turn_that_finishes_anyway_is_not_delivered():
    # A step that ignores the cancel and returns must still not reach the traveler.
    route = Route(intent=Intent.CHANGE_SETTING, day=2)
    steps = make_steps(route, [])
    delivered = []

    async def deliver(text, result):
        delivered.append(text)

    async def scenario():
        runner = TurnRunner("s", steps, deliver)
        runner._latest = 5                           # a newer message already arrived
        await runner._run("old", 4)

    asyncio.run(scenario())
    assert delivered == []
