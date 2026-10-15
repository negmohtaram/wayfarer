"""A recorded scenario replays to the identical itinerary, with the network blocked."""
import asyncio
import random
import socket

import pytest

import src.agents.llm as llm_module
import src.agents.nodes.assembler as assembler
import src.agents.nodes.critics.experience as experience
from src.agents.replan_graph import DisruptionAnalysis
from src.evaluation.recording import recording
from src.evaluation.runner import run_scenario
from src.evaluation.scenarios.baseline_scenarios import SCENARIO_BY_ID
from src.models.itinerary import CritiqueResult

SCENARIO = "disruption_weather_amsterdam"   # planning + critic + replanning + live weather


class RandomModel:
    """Stands in for the LLM. Answers are random, so only a faithful replay can match them."""

    def __init__(self, rng):
        self.rng, self.calls = rng, 0

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        if self.schema is CritiqueResult:
            return CritiqueResult(critic_type="experience", score=round(self.rng.uniform(4, 9), 2),
                                  verdict=self.rng.choice(["approve", "revise"]))
        return DisruptionAnalysis(disruption_type="weather", recommendation=f"r{self.rng.random():.6f}")


class NoModel:
    def with_structured_output(self, schema):
        return self

    async def ainvoke(self, messages):
        raise AssertionError("replay called the model")


def _use_model(monkeypatch, factory):
    monkeypatch.setattr(experience, "get_llm", lambda temperature=0.7: factory())
    monkeypatch.setattr(llm_module, "get_llm", lambda temperature=0.7: factory())


def _comparable(result):
    """What a replay must reproduce. session_id and run_at hold the clock time, runtimes vary."""
    data = result.model_dump(mode="json")
    for key in ("output_itinerary", "initial_itinerary"):
        if data.get(key):
            data[key].pop("session_id", None)
    data.pop("run_at")
    data.pop("operational_metrics")
    return data


def test_recorded_scenario_replays_to_the_identical_itinerary(tmp_path, monkeypatch):
    scenario = SCENARIO_BY_ID[SCENARIO]
    rng = random.Random()

    async def random_weather(city, start_date, days):
        return [{"condition": rng.choice(["sunny", "rainy"]), "temp_high": rng.uniform(10, 25),
                 "temp_low": 5.0, "rain_probability": rng.uniform(0, 100)} for _ in range(days)]

    model = RandomModel(rng)
    _use_model(monkeypatch, lambda: model)
    monkeypatch.setattr(assembler, "_fetch_weather_live", random_weather)
    with recording("record", tmp_path) as rec:
        recorded = asyncio.run(run_scenario(scenario))
    assert recorded.planning_error is None
    assert rec.calls >= 3 and model.calls >= 2          # weather + critic + disruption analyzer

    async def no_weather(*args, **kwargs):
        raise AssertionError("replay fetched weather")

    def no_network(*args, **kwargs):
        raise AssertionError("replay opened a network connection")

    _use_model(monkeypatch, NoModel)
    monkeypatch.setattr(assembler, "_fetch_weather_live", no_weather)
    monkeypatch.setattr(socket.socket, "connect", no_network)
    replays = []
    for _ in range(2):
        with recording("replay", tmp_path) as rep:
            replays.append(asyncio.run(run_scenario(scenario)))
        assert rep.replayed == rec.calls

    assert _comparable(replays[0]) == _comparable(recorded)
    assert _comparable(replays[1]) == _comparable(recorded)


def test_replay_needs_no_api_key(tmp_path, monkeypatch):
    """Replays use the real client factory with no key set and still match the recording."""
    scenario = SCENARIO_BY_ID[SCENARIO]
    rng = random.Random()

    async def random_weather(city, start_date, days):
        return [{"condition": rng.choice(["sunny", "rainy"]), "temp_high": rng.uniform(10, 25),
                 "temp_low": 5.0, "rain_probability": rng.uniform(0, 100)} for _ in range(days)]

    model = RandomModel(rng)
    _use_model(monkeypatch, lambda: model)
    monkeypatch.setattr(assembler, "_fetch_weather_live", random_weather)
    with recording("record", tmp_path) as rec:
        recorded = asyncio.run(run_scenario(scenario))
    monkeypatch.undo()

    async def no_weather(*args, **kwargs):
        raise AssertionError("replay fetched weather")

    def no_network(*args, **kwargs):
        raise AssertionError("replay opened a network connection")

    monkeypatch.setattr(llm_module, "GROQ_API_KEY", "")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(assembler, "_fetch_weather_live", no_weather)
    monkeypatch.setattr(socket.socket, "connect", no_network)
    with recording("replay", tmp_path) as rep:
        replayed = asyncio.run(run_scenario(scenario))

    assert rep.failures == []
    assert rep.replayed == rec.calls
    assert _comparable(replayed) == _comparable(recorded)


def test_client_is_built_on_the_first_call(monkeypatch):
    monkeypatch.setattr(llm_module, "GROQ_API_KEY", "")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    model = llm_module.get_llm(temperature=0.3).with_structured_output(CritiqueResult)
    with pytest.raises(Exception):
        asyncio.run(model.ainvoke([]))


def test_fallback_outside_a_recorded_call_makes_the_run_invalid(tmp_path, monkeypatch):
    """An error before the model call (here: building the client) must not pass as a valid run."""
    def broken_client(temperature=0.7):
        raise RuntimeError("client could not be built")

    monkeypatch.setattr(experience, "get_llm", broken_client)
    monkeypatch.setattr(assembler, "_fetch_weather_live", _no_weather_data)
    with recording("record", tmp_path) as rec:
        result = asyncio.run(run_scenario(SCENARIO_BY_ID["balanced_london_reference"]))

    assert result.planning_error is None
    assert any(f.startswith("experience_critic: RuntimeError") for f in rec.failures)


async def _no_weather_data(city, start_date, days):
    return [None] * days
