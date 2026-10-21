"""Retry policy: which failures are retried, how long we wait, and when we give up.

No network and no real waiting: sleep and the clock are replaced by a fake clock.
"""
import asyncio
import random

import groq
import httpx
import pytest

from src.agents import llm, ratelimit
from src.agents.llm import BATCH, CHAT, PERMANENT, TRANSIENT, LLMCallError, RetryPolicy, call_with_retry


def status_error(cls, status: int, retry_after: str | None = None):
    headers = {"retry-after": retry_after} if retry_after else {}
    response = httpx.Response(status, headers=headers, request=httpx.Request("POST", "https://api.test"))
    return cls(f"status {status}", response=response, body=None)


class Clock:
    def __init__(self):
        self.t = 0.0
        self.waits: list[float] = []

    def now(self) -> float:
        return self.t

    async def sleep(self, seconds: float) -> None:
        self.waits.append(seconds)
        self.t += seconds


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(llm, "_now", c.now)
    monkeypatch.setattr(llm, "_sleep", c.sleep)
    monkeypatch.setattr(ratelimit, "_now", c.now)
    monkeypatch.setattr(ratelimit, "_sleep", c.sleep)
    monkeypatch.setattr(llm, "_random", random.Random(0))
    return c


class Flaky:
    """Raises the given errors in order, then answers "ok"."""

    def __init__(self, *errors):
        self.errors = list(errors)
        self.calls = 0

    async def __call__(self):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return "ok"


def run(coro):
    return asyncio.run(coro)


@pytest.mark.parametrize("cls,status", [
    (groq.BadRequestError, 400),
    (groq.AuthenticationError, 401),
    (groq.PermissionDeniedError, 403),
])
def test_permanent_errors_are_called_once_and_never_wait(clock, cls, status):
    fake = Flaky(*[status_error(cls, status)] * 3)
    with pytest.raises(LLMCallError) as err:
        run(call_with_retry(fake))
    assert fake.calls == 1
    assert clock.waits == []
    assert (err.value.kind, err.value.status) == (PERMANENT, status)


def test_our_own_bug_is_not_retried(clock):
    fake = Flaky(KeyError("trip_days"))
    with pytest.raises(LLMCallError) as err:
        run(call_with_retry(fake))
    assert fake.calls == 1 and err.value.kind == PERMANENT


def test_server_errors_are_retried_with_full_jitter(clock):
    fake = Flaky(status_error(groq.InternalServerError, 503), status_error(groq.InternalServerError, 500))
    assert run(call_with_retry(fake)) == "ok"
    assert fake.calls == 3
    assert len(clock.waits) == 2
    assert 0 <= clock.waits[0] <= 1.0 and 0 <= clock.waits[1] <= 2.0   # caps 1 s, then 2 s


def test_retry_after_is_a_minimum_wait(clock):
    fake = Flaky(status_error(groq.RateLimitError, 429, retry_after="2"))
    assert run(call_with_retry(fake)) == "ok"
    assert fake.calls == 2 and clock.waits[0] >= 2.0


def test_timeout_is_retried(clock):
    fake = Flaky(TimeoutError())
    assert run(call_with_retry(fake)) == "ok"
    assert fake.calls == 2


def test_gives_up_when_retry_after_is_longer_than_the_time_left(clock):
    fake = Flaky(status_error(groq.RateLimitError, 429, retry_after="3600"))   # daily quota gone
    with pytest.raises(LLMCallError) as err:
        run(call_with_retry(fake, policy=CHAT))
    assert fake.calls == 1 and clock.waits == []
    assert (err.value.kind, err.value.status) == (TRANSIENT, 429)


def test_last_transient_error_reports_every_attempt(clock):
    fake = Flaky(*[status_error(groq.InternalServerError, 503)] * 3)
    with pytest.raises(LLMCallError) as err:
        run(call_with_retry(fake))
    assert fake.calls == BATCH.max_attempts
    assert [a[:2] for a in err.value.attempts] == [(TRANSIENT, 503)] * 3


def test_a_slow_call_is_cut_by_the_attempt_timeout():
    calls = 0

    async def slow():
        nonlocal calls
        calls += 1
        await asyncio.sleep(10)

    policy = RetryPolicy(max_attempts=2, base_delay=0.0, attempt_timeout=0.01, deadline=1.0)
    with pytest.raises(LLMCallError) as err:
        run(call_with_retry(slow, policy=policy))
    assert calls == 2 and err.value.kind == TRANSIENT
