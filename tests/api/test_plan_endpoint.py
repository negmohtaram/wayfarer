"""POST /api/plan runs the real planning graph on a fake model, with no Groq key."""

import json

import pytest
from fastapi.testclient import TestClient

from src.agents.nodes import assembler
from src.api import main
from src.models.itinerary import CritiqueResult
from src.services import session
from src.evaluation.scenarios.baseline_scenarios import SCENARIO_BY_ID


class FakeModel:
    """Answers every structured call with an approving critique and counts the calls."""

    def __init__(self, log: list):
        self.log = log

    def with_structured_output(self, schema, **kwargs):
        assert schema is CritiqueResult
        return self

    async def ainvoke(self, messages, **kwargs):
        self.log.append("call")
        return CritiqueResult(critic_type="experience", score=8.0, verdict="approve")


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(main, "GROQ_API_KEY", "")
    monkeypatch.setattr(session, "SESSION_DIR", tmp_path)

    async def no_weather(city, start_date, days):
        return [None] * days

    monkeypatch.setattr(assembler, "_fetch_weather_live", no_weather)
    calls: list = []
    app = main.create_app(lambda temperature=0.7: FakeModel(calls), require_key=False)
    with TestClient(app) as c:
        c.calls = calls
        yield c


def _events(body: str) -> list[tuple[str, dict]]:
    out = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def test_plan_runs_the_graph_on_the_fake_model(client):
    persona = SCENARIO_BY_ID["balanced_paris_reference"].persona
    resp = client.post("/api/plan", json={"session_id": "s1", "persona": persona.model_dump(mode="json")})

    assert resp.status_code == 200
    events = _events(resp.text)
    names = [e for e, _ in events]
    assert "error" not in names, events
    assert names[-1] == "done"
    complete = next(d for e, d in events if e == "complete")
    assert len(complete["itinerary"]["days"]) == persona.trip_days
    assert client.calls, "the experience critic should have used the fake model"
    critiques = [d for e, d in events if e == "critique" and d["critic_type"] == "experience"]
    assert critiques and critiques[0]["verdict"] == "approve"     # the fake's answer, not the fallback


def test_real_app_without_a_key_does_not_start(monkeypatch):
    monkeypatch.setattr(main, "GROQ_API_KEY", "")
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        with TestClient(main.create_app(lambda temperature=0.7: None)):
            pass
