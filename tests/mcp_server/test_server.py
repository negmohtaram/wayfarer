"""The MCP tools against the real API app on a fake model, in one event loop, without a network."""

import asyncio
from contextlib import asynccontextmanager

import httpx
import pytest
from mcp.client import Client

from src.agents.nodes import assembler
from src.api import main
from src.jobs.models import JobStatus
from src.mcp_server.server import build_server
from src.services import session
from tests.api.conftest import FakeModel
from tests.api.test_plan_jobs import PERSONA

PERSONA_JSON = PERSONA.model_dump(mode="json")


@pytest.fixture
def wayfarer(tmp_path, monkeypatch):
    """An async context that yields an MCP client wired to a running app."""
    monkeypatch.setattr(main, "GROQ_API_KEY", "")
    monkeypatch.setattr(session, "SESSION_DIR", tmp_path)

    async def no_weather(city, start_date, days):
        return [None] * days

    monkeypatch.setattr(assembler, "_fetch_weather_live", no_weather)

    @asynccontextmanager
    async def connect(http: httpx.AsyncClient | None = None):
        app = main.create_app(lambda temperature=0.7: FakeModel([]), require_key=False, database_url="")
        async with app.router.lifespan_context(app):          # starts the worker, as uvicorn would
            async with http or httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                                 base_url="http://wayfarer") as api:
                async with Client(build_server(api)) as client:
                    yield client
    return connect


async def _until_finished(client, job_id: str) -> dict:
    for _ in range(200):
        state = (await client.call_tool("get_job_status", {"job_id": job_id})).structured_content
        if state["status"] in (JobStatus.DONE, JobStatus.FAILED):
            return state
        await asyncio.sleep(0.05)
    raise AssertionError(f"job {job_id} never finished")


def _text(result) -> str:
    return " ".join(c.text for c in result.content)


# ── contract: what a client sees in tools/list ─────────────────────────────────

def test_tool_list_is_the_contract(wayfarer):
    async def run():
        async with wayfarer() as client:
            return {t.name: t for t in (await client.list_tools()).tools}
    tools = asyncio.run(run())

    assert set(tools) == {"plan_trip", "get_job_status", "validate_itinerary"}
    plan = tools["plan_trip"].input_schema
    assert plan["required"] == ["persona"]
    persona = plan["$defs"]["TravelerPersona"]["properties"]
    assert {"destination", "trip_days", "budget_per_day", "physical_constraints"} <= set(persona)
    assert persona["trip_days"]["maximum"] == 14
    assert plan["properties"]["attempt"]["minimum"] == 1
    assert tools["get_job_status"].input_schema["required"] == ["job_id"]
    status = tools["get_job_status"].output_schema["$defs"]["JobStatus"]["enum"]
    assert set(status) == {s.value for s in JobStatus}
    assert tools["validate_itinerary"].input_schema["required"] == ["plan_id"]
    assert {"passed", "failures"} <= set(tools["validate_itinerary"].output_schema["properties"])


# ── behaviour through the protocol ─────────────────────────────────────────────

def test_plan_then_status_then_validate(wayfarer):
    async def run():
        async with wayfarer() as client:
            started = await client.call_tool("plan_trip", {"persona": PERSONA_JSON})
            assert not started.is_error, _text(started)
            done = await _until_finished(client, started.structured_content["job_id"])
            checked = await client.call_tool("validate_itinerary", {"plan_id": done["plan_id"]})
            return done, checked
    done, checked = asyncio.run(run())

    assert done["status"] == "done"
    assert done["outline"].count("Day ") == PERSONA.trip_days
    assert not checked.is_error, _text(checked)
    report = checked.structured_content
    assert report["plan_id"] == done["plan_id"]
    assert "NO_DUPLICATE_EXPERIENCES" in report["checks_run"]
    assert report["passed"] == (not [f for f in report["failures"] if f["severity"] == "error"])


def test_a_retry_finds_the_same_job_and_a_new_attempt_starts_another(wayfarer):
    async def run():
        async with wayfarer() as client:
            first = await client.call_tool("plan_trip", {"persona": PERSONA_JSON})
            again = await client.call_tool("plan_trip", {"persona": PERSONA_JSON})
            second = await client.call_tool("plan_trip", {"persona": PERSONA_JSON, "attempt": 2})
            return [r.structured_content["job_id"] for r in (first, again, second)]
    first, again, second = asyncio.run(run())

    assert first == again
    assert second != first


@pytest.mark.parametrize("call, args, expected", [
    ("plan_trip", {"persona": {**PERSONA_JSON, "destination": "atlantis"}}, "Supported:"),
    ("plan_trip", {"persona": {**PERSONA_JSON, "trip_days": 40}}, "trip_days"),
    ("plan_trip", {"persona": PERSONA_JSON, "attempt": 0}, "attempt"),
    ("get_job_status", {"job_id": "nope"}, "Not found"),
    ("validate_itinerary", {"plan_id": "nope"}, "Not found"),
    ("validate_itinerary", {}, "plan_id"),
])
def test_errors_reach_the_model_as_readable_text(wayfarer, call, args, expected):
    async def run():
        async with wayfarer() as client:
            return await client.call_tool(call, args)
    result = asyncio.run(run())

    assert result.is_error
    assert expected in _text(result)


def test_an_id_cannot_walk_to_another_route(wayfarer):
    async def run():
        async with wayfarer() as client:
            started = await client.call_tool("plan_trip", {"persona": PERSONA_JSON})
            done = await _until_finished(client, started.structured_content["job_id"])
            # Unescaped, this path would resolve to GET /api/plans/<real id> and return a plan.
            return await client.call_tool("get_job_status", {"job_id": f"../plans/{done['plan_id']}"})
    result = asyncio.run(run())

    assert result.is_error
    assert "Not found" in _text(result)


def test_api_down_says_so(wayfarer):
    async def run():
        down = httpx.AsyncClient(base_url="http://127.0.0.1:9", timeout=2)     # nothing listens on port 9
        async with wayfarer(down) as client:
            return await client.call_tool("get_job_status", {"job_id": "abc"})
    result = asyncio.run(run())

    assert result.is_error
    assert "not reachable" in _text(result)
