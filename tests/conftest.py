import pytest

from src.agents import ratelimit


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
