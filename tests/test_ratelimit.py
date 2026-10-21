"""Rate limiter: two token buckets, requests and tokens, with no real waiting."""
import asyncio
import random

import groq
import pytest

from src.agents import llm, ratelimit
from src.agents.llm import CHAT, LLMCallError
from src.agents.ratelimit import Limiter, estimate_tokens, limiter_for
from tests.test_retry import Flaky, status_error


class Clock:
    def __init__(self):
        self.t = 0.0

    def now(self) -> float:
        return self.t

    async def sleep(self, seconds: float) -> None:
        self.t += seconds


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    for module in (llm, ratelimit):
        monkeypatch.setattr(module, "_now", c.now)
        monkeypatch.setattr(module, "_sleep", c.sleep)
    monkeypatch.setattr(llm, "_random", random.Random(0))
    return c


async def start_times(limiter: Limiter, clock: Clock, sizes: list[int]) -> list[float]:
    times: list[float] = []

    async def one(n):
        await limiter.acquire(n)
        times.append(clock.t)

    await asyncio.gather(*(one(n) for n in sizes))
    return times


def test_tokens_bind_before_requests(clock):
    # 30 RPM, 6,000 TPM, a full minute of burst: six 1,000-token calls fit, then 100 tokens/s.
    limiter = Limiter(rpm=30, tpm=6_000, burst_seconds=60)
    times = asyncio.run(start_times(limiter, clock, [1_000] * 12))
    assert times[:6] == [0.0] * 6
    assert times[6:] == pytest.approx([10, 20, 30, 40, 50, 60])


def test_requests_bind_for_small_calls(clock):
    # 10 s of burst at 30 RPM = 5 requests; then one every 2 s.
    limiter = Limiter(rpm=30, tpm=6_000)
    times = asyncio.run(start_times(limiter, clock, [10] * 7))
    assert times[:5] == [0.0] * 5
    assert times[5:] == pytest.approx([2, 4])


def test_a_call_larger_than_the_bucket_fails_at_once(clock):
    limiter = Limiter(rpm=30, tpm=6_000)          # token bucket holds 1,000
    with pytest.raises(ValueError):
        asyncio.run(limiter.acquire(1_001))
    assert clock.t == 0


def test_unused_reservation_is_given_back_and_overuse_is_owed(clock):
    limiter = Limiter(rpm=30, tpm=6_000)
    asyncio.run(limiter.acquire(1_000))
    limiter.reconcile(reserved=1_000, used=400)
    assert limiter.tokens.level == pytest.approx(600)
    limiter.reconcile(reserved=0, used=1_000)       # the reply was bigger than we thought
    assert limiter.tokens.level == pytest.approx(-400)


def test_every_retry_is_admitted_by_the_limiter(clock, monkeypatch):
    limiter = limiter_for("m")
    taken: list[float] = []
    real = limiter.acquire

    async def counting(n, max_wait=float("inf")):
        taken.append(n)
        await real(n, max_wait)

    monkeypatch.setattr(limiter, "acquire", counting)
    fake = Flaky(status_error(groq.InternalServerError, 503), status_error(groq.InternalServerError, 503))
    asyncio.run(llm._retry(fake, (), {}, llm.BATCH, "m"))
    assert fake.calls == 3 and len(taken) == 3


def test_a_429_empties_the_token_bucket(clock):
    limiter = limiter_for("m")
    fake = Flaky(status_error(groq.RateLimitError, 429))
    asyncio.run(llm._retry(fake, (), {}, llm.BATCH, "m"))
    # Drained to 0, then the retry had to wait for its 150 reserved tokens to refill.
    assert fake.calls == 2 and clock.t >= 150 / limiter.tokens.rate - 1e-9 - 1e-9


def test_no_capacity_before_the_deadline_is_a_failure_not_a_hang(clock):
    limiter = limiter_for("m")
    limiter.tokens.level = -10_000                  # owed far more than 20 s of refill
    fake = Flaky()
    with pytest.raises(LLMCallError) as err:
        asyncio.run(llm._retry(fake, (), {}, CHAT, "m"))
    assert fake.calls == 0 and "capacity" in err.value.attempts[-1][2]


def test_never_more_than_the_limit_in_any_rolling_minute(clock):
    # Greedy callers for 5 minutes: the provider must never see more than 8,000 tokens
    # or 30 requests inside any 60 s window.
    limiter = limiter_for("openai/gpt-oss-120b")
    starts: list[tuple[float, int]] = []

    async def caller(size):
        while clock.t < 300:
            await limiter.acquire(size)
            starts.append((clock.t, size))

    async def main():
        await asyncio.gather(caller(1_000), caller(700), caller(150))

    asyncio.run(main())
    for t, _ in starts:
        window = [n for s, n in starts if t <= s < t + 60]
        assert sum(window) <= 8_000 and len(window) <= 30


def test_one_limiter_per_model():
    assert limiter_for("a") is limiter_for("a")
    assert limiter_for("a") is not limiter_for("b")


def test_estimate_counts_message_text_plus_the_reply():
    assert estimate_tokens([{"role": "user", "content": "x" * 400}]) == 100 + ratelimit.OUTPUT_ESTIMATE
    assert estimate_tokens(None) == ratelimit.OUTPUT_ESTIMATE
