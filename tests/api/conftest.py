"""Shared API fixtures: the real app on a fake model, no Groq key, sessions in a temp dir."""

import pytest
from fastapi.testclient import TestClient

from src.agents.nodes import assembler
from src.api import main
from src.models.itinerary import CritiqueResult
from src.services import session


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
