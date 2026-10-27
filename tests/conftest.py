import asyncio
import os

import pytest

from src.agents import llm, ratelimit

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL", "")


class _LimiterClock:
    def __init__(self):
        self.t = 0.0

    def now(self) -> float:
        return self.t

    async def sleep(self, seconds: float) -> None:
        self.t += seconds


@pytest.fixture(autouse=True)
def fresh_rate_limiters(monkeypatch):
    """Each test starts with full buckets on a fake clock, so no test waits for real time."""
    clock = _LimiterClock()
    monkeypatch.setattr(ratelimit, "_now", clock.now)
    monkeypatch.setattr(ratelimit, "_sleep", clock.sleep)
    ratelimit.reset()
    yield
    ratelimit.reset()


@pytest.fixture(autouse=True)
def no_real_groq(monkeypatch):
    """Fail any test that builds a real Groq client, so a missed fake can't reach the network."""
    def refuse(*args, **kwargs):
        raise RuntimeError("real Groq client built inside a test")
    monkeypatch.setattr(llm, "ChatGroq", refuse)


@pytest.fixture
def fresh_pg():
    """A real Postgres with empty tables for this test only. Skips when TEST_DATABASE_URL is not set."""
    if not TEST_DATABASE_URL:
        pytest.skip("TEST_DATABASE_URL not set")
    from src.jobs.pg_store import PostgresJobStore
    from src.storage.tables import metadata

    async def reset():
        store = PostgresJobStore(TEST_DATABASE_URL)
        async with store.engine.begin() as conn:
            await conn.run_sync(metadata.drop_all)
            await conn.run_sync(metadata.create_all)
        await store.close()
    asyncio.run(reset())
    return TEST_DATABASE_URL
