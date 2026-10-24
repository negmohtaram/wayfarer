"""POST /api/plan runs the real planning graph on a fake model, with no Groq key."""

import json

import pytest
from fastapi.testclient import TestClient

from src.api import main
from src.evaluation.scenarios.baseline_scenarios import SCENARIO_BY_ID


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
