"""Errors a stranger meets first: the model provider down, a city we have no data for."""

from src.agents.llm import LLMCallError
from src.api.routers import persona


def test_model_provider_refusal_is_a_503_with_a_reason(client, monkeypatch):
    async def refused(**kwargs):
        raise LLMCallError("permanent", 403, [("permanent", 403, "Access denied")])

    monkeypatch.setattr(persona, "run_persona_turn", refused)
    resp = client.post("/api/persona/turn", json={"destination": "paris", "trip_days": 2})

    assert resp.status_code == 503
    assert resp.headers["retry-after"] == "30"
    detail = resp.json()["detail"]
    assert detail["type"] == "model_unavailable" and detail["upstream_status"] == 403
    assert "Access denied" not in detail["message"]       # provider text stays in the log


def test_unknown_city_is_rejected_before_a_job_exists(client):
    resp = client.post("/api/plans", json={"persona": {"destination": "tehran", "trip_days": 2}},
                       headers={"Idempotency-Key": "k1"})

    assert resp.status_code == 422
    assert resp.json()["detail"]["type"] == "unsupported_destination"
    # No job was stored, so the key is still free for a valid request.
    ok = client.post("/api/plans", json={"persona": {"destination": "paris", "trip_days": 2}},
                     headers={"Idempotency-Key": "k1"})
    assert ok.status_code == 202


def test_city_names_are_case_insensitive(client):
    resp = client.post("/api/plans", json={"persona": {"destination": "Paris", "trip_days": 1}})
    assert resp.status_code == 202
