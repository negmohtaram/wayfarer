"""Repeated runs are recorded separately, replay to themselves, and failed calls make a run invalid."""
import asyncio
import random
import socket

import src.agents.llm as llm_module
import src.agents.nodes.assembler as assembler
from src.evaluation.runner import recording_folder, run_all
from src.evaluation.scenarios.baseline_scenarios import SCENARIO_BY_ID
from tests.test_replay_runner import SCENARIO, NoModel, RandomModel, _comparable, _use_model


def _random_weather(rng):
    async def weather(city, start_date, days):
        return [{"condition": rng.choice(["sunny", "rainy"]), "temp_high": rng.uniform(10, 25),
                 "temp_low": 5.0, "rain_probability": rng.uniform(0, 100)} for _ in range(days)]
    return weather


def _block_live_calls(monkeypatch):
    async def no_weather(*args, **kwargs):
        raise AssertionError("replay fetched weather")

    def no_network(*args, **kwargs):
        raise AssertionError("replay opened a network connection")

    _use_model(monkeypatch, NoModel)
    monkeypatch.setattr(assembler, "_fetch_weather_live", no_weather)
    monkeypatch.setattr(socket.socket, "connect", no_network)


def _run(mode, root, repeats=1):
    scenarios = [SCENARIO_BY_ID[SCENARIO]]
    return asyncio.run(run_all(save=False, mode=mode, repeats=repeats,
                               scenarios=scenarios, recordings_root=root))


def test_each_repeat_records_its_own_folder_and_replays_to_itself(tmp_path, monkeypatch):
    rng = random.Random()
    model = RandomModel(rng)
    _use_model(monkeypatch, lambda: model)
    monkeypatch.setattr(assembler, "_fetch_weather_live", _random_weather(rng))
    recorded = _run("record", tmp_path, repeats=2)

    assert [r.run for r in recorded] == [0, 1]
    assert all(r.eval_error is None and r.recorded_calls >= 3 for r in recorded)
    for run in (0, 1):
        assert any(recording_folder(tmp_path, run, SCENARIO).iterdir())

    _block_live_calls(monkeypatch)
    replayed = _run("replay", tmp_path, repeats=2)

    for original, replay in zip(recorded, replayed):
        assert replay.eval_error is None
        assert replay.recorded_calls == original.recorded_calls
        assert _comparable(replay) == _comparable(original)


def test_replay_without_recordings_marks_the_run_invalid(tmp_path, monkeypatch):
    _block_live_calls(monkeypatch)
    [result] = _run("replay", tmp_path)
    assert result.eval_error is not None
    assert "no recording" in result.eval_error


class FailingModel:
    """A model that is always down, like a provider at its daily limit."""

    def with_structured_output(self, schema):
        return self

    async def ainvoke(self, messages):
        raise RuntimeError("rate limit")


async def _no_wait(seconds):
    return None


def test_failed_live_call_marks_the_run_invalid_even_if_the_fallback_plans(tmp_path, monkeypatch):
    _use_model(monkeypatch, FailingModel)
    monkeypatch.setattr(assembler, "_fetch_weather_live", _random_weather(random.Random(0)))
    monkeypatch.setattr(llm_module.asyncio, "sleep", _no_wait)
    [result] = _run("record", tmp_path)
    assert result.eval_error is not None
    assert "experience_critic: RuntimeError" in result.eval_error


def test_resume_records_only_what_is_missing_and_replays_the_rest(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_module.asyncio, "sleep", _no_wait)
    rng = random.Random()
    monkeypatch.setattr(assembler, "_fetch_weather_live", _random_weather(rng))

    _use_model(monkeypatch, FailingModel)                  # the provider is down
    [first] = _run("record", tmp_path)
    assert first.eval_error is not None
    assert not (recording_folder(tmp_path, 0, SCENARIO) / "_complete").exists()

    model = RandomModel(rng)                               # the provider is back
    _use_model(monkeypatch, lambda: model)
    second = asyncio.run(run_all(save=False, mode="record", scenarios=[SCENARIO_BY_ID[SCENARIO]],
                                 recordings_root=tmp_path, resume=True))[0]
    assert second.eval_error is None
    files = list(recording_folder(tmp_path, 0, SCENARIO).glob("*.json"))
    assert len(files) == second.recorded_calls             # nothing left over from the failed run
    assert (recording_folder(tmp_path, 0, SCENARIO) / "_complete").exists()

    _block_live_calls(monkeypatch)                         # resuming again calls nothing live
    third = asyncio.run(run_all(save=False, mode="record", scenarios=[SCENARIO_BY_ID[SCENARIO]],
                                recordings_root=tmp_path, resume=True))[0]
    assert third.eval_error is None
    assert _comparable(third) == _comparable(second)
