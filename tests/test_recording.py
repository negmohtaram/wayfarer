"""Record model and weather calls once, replay them exactly, never fall back to the network."""
import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agents.llm import call_with_retry
from src.evaluation.recording import ReplayMiss, recording
from src.models.itinerary import CritiqueResult

META = {"site": "experience_critic", "model": "m", "temperature": 0.3, "schema": "CritiqueResult"}
MSGS = [SystemMessage("You are a critic."), HumanMessage("Evaluate day 1.")]


class FakeModel:
    def __init__(self, *answers):
        self.answers, self.calls = list(answers), 0

    async def ainvoke(self, messages):
        self.calls += 1
        return self.answers.pop(0)


def run(coro):
    return asyncio.run(coro)


def critique(score, verdict="approve"):
    return CritiqueResult(critic_type="experience", score=score, verdict=verdict)


def test_record_then_replay_returns_the_same_typed_result_without_calling_the_model(tmp_path):
    live = FakeModel(critique(7.5))
    with recording("record", tmp_path):
        recorded = run(call_with_retry(live.ainvoke, MSGS, meta=META))
    with recording("replay", tmp_path):
        replayed = run(call_with_retry(FakeModel().ainvoke, MSGS, meta=META))
    assert replayed == recorded and isinstance(replayed, CritiqueResult)
    assert live.calls == 1


def test_same_request_twice_keeps_both_answers_in_order(tmp_path):
    live = FakeModel(critique(5.0, "revise"), critique(8.0))
    with recording("record", tmp_path):
        first = run(call_with_retry(live.ainvoke, MSGS, meta=META))
        second = run(call_with_retry(live.ainvoke, MSGS, meta=META))
    with recording("replay", tmp_path):
        again = [run(call_with_retry(FakeModel().ainvoke, MSGS, meta=META)) for _ in range(2)]
    assert [first.verdict, second.verdict] == ["revise", "approve"]
    assert again == [first, second]


def test_missing_recording_fails_loudly(tmp_path):
    with recording("replay", tmp_path), pytest.raises(ReplayMiss):
        run(call_with_retry(FakeModel(critique(1)).ainvoke, MSGS, meta=META))


def test_changed_prompt_or_temperature_is_a_different_recording(tmp_path):
    with recording("record", tmp_path):
        run(call_with_retry(FakeModel(critique(7)).ainvoke, MSGS, meta=META))
    with recording("replay", tmp_path), pytest.raises(ReplayMiss):
        run(call_with_retry(FakeModel().ainvoke, MSGS, meta=META | {"temperature": 0.7}))
    with recording("replay", tmp_path), pytest.raises(ReplayMiss):
        run(call_with_retry(FakeModel().ainvoke, [HumanMessage("Evaluate day 2.")], meta=META))


def test_a_model_call_without_meta_is_refused_while_recording(tmp_path):
    with recording("record", tmp_path), pytest.raises(ValueError, match="meta"):
        run(call_with_retry(FakeModel(critique(1)).ainvoke, MSGS))


def test_outside_evaluation_calls_go_straight_to_the_model():
    live = FakeModel(critique(6))
    assert run(call_with_retry(live.ainvoke, MSGS)).score == 6 and live.calls == 1


def test_plain_messages_round_trip(tmp_path):
    with recording("record", tmp_path):
        run(call_with_retry(FakeModel(AIMessage(content="hello")).ainvoke, MSGS, meta=META | {"schema": None}))
    with recording("replay", tmp_path):
        out = run(call_with_retry(FakeModel().ainvoke, MSGS, meta=META | {"schema": None}))
    assert isinstance(out, AIMessage) and out.content == "hello"


def test_weather_is_recorded_and_replayed_offline(tmp_path, monkeypatch):
    from src.agents.nodes import assembler
    calls = []

    async def fake_live(city, start, days):
        calls.append(city)
        return [{"date": start, "condition": "sunny", "temp_high": 21, "temp_low": 12, "rain_probability": 5}]

    monkeypatch.setattr(assembler, "_fetch_weather_live", fake_live)
    with recording("record", tmp_path):
        rec = run(assembler.fetch_weather("Paris", "2026-11-02", 1))
    with recording("replay", tmp_path):
        rep = run(assembler.fetch_weather("Paris", "2026-11-02", 1))
    assert rep == rec and calls == ["Paris"]
