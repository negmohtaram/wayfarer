"""Client-side rate limiting: stay under the provider's requests and tokens per minute.

Groq limits each model separately on requests per minute and tokens per minute. One limiter
per model holds two token buckets, and a call starts only when both have room. Retries pass
through the limiter too, because the provider counts every attempt.

One process only: the lock is an asyncio.Lock. Several processes sharing one key would need
the buckets in a shared store (for example Redis) with an atomic take.
"""
import asyncio
import time

from src.config import RATE_LIMITS

OUTPUT_ESTIMATE = 150       # tokens reserved for the reply; corrected after the call
CHARS_PER_TOKEN = 4

# Replaced in tests so no test waits for real time.
_now = time.monotonic
_sleep = asyncio.sleep


class NoCapacity(Exception):
    """The buckets will not have room before the caller's deadline."""


class TokenBucket:
    """Holds up to `capacity` permits and refills `per_minute / 60` permits every second."""

    def __init__(self, capacity: float, per_minute: float):
        self.capacity = capacity
        self.rate = per_minute / 60.0
        self.level = capacity
        self.updated = _now()

    def refill(self) -> None:
        now = _now()
        self.level = min(self.capacity, self.level + (now - self.updated) * self.rate)
        self.updated = now

    def wait_time(self, n: float) -> float:
        self.refill()
        return max(0.0, (n - self.level) / self.rate)


class Limiter:
    def __init__(self, rpm: float, tpm: float, burst_seconds: float = 10.0):
        # A small burst: the provider counts over a rolling minute, so a full minute's
        # budget sent at once would still be refused.
        self.requests = TokenBucket(max(1.0, rpm * burst_seconds / 60), rpm)
        self.tokens = TokenBucket(tpm * burst_seconds / 60, tpm)
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: float, max_wait: float = float("inf")) -> None:
        """Wait until one request and `tokens` tokens are free, then take both together.

        Raises NoCapacity instead of waiting past max_wait seconds (time in the queue counts).
        """
        give_up = _now() + max_wait
        if tokens > self.tokens.capacity:
            raise ValueError(f"a call of {tokens:.0f} tokens can never fit a bucket of "
                             f"{self.tokens.capacity:.0f}")
        async with self._lock:          # first come, first served; both buckets change together
            while True:
                wait = max(self.requests.wait_time(1), self.tokens.wait_time(tokens))
                if wait <= 1e-9:
                    self.requests.level -= 1
                    self.tokens.level -= tokens
                    return
                if _now() + wait > give_up:
                    raise NoCapacity(f"needs {wait:.1f}s, {max(0.0, give_up - _now()):.1f}s left")
                await _sleep(wait)

    def reconcile(self, reserved: float, used: float) -> None:
        """Correct the reservation with what the call really used. A deficit carries over."""
        self.tokens.refill()
        self.tokens.level = min(self.tokens.capacity, self.tokens.level + reserved - used)

    def drain(self) -> None:
        """The provider said 429: our estimate of its budget was wrong, so assume none is left."""
        self.tokens.refill()
        self.tokens.level = min(self.tokens.level, 0.0)


_limiters: dict[str, Limiter] = {}


def limiter_for(model: str) -> Limiter:
    """One shared limiter per model: every call site draws from the same quota."""
    if model not in _limiters:
        rpm, tpm = RATE_LIMITS.get(model, RATE_LIMITS["default"])
        _limiters[model] = Limiter(rpm, tpm)
    return _limiters[model]


def reset() -> None:
    _limiters.clear()


def estimate_tokens(messages) -> int:
    """A rough size of the request before sending it: characters / 4 plus room for the reply."""
    if messages is None:
        chars = 0
    elif isinstance(messages, str):
        chars = len(messages)
    else:
        chars = 0
        for m in messages:
            content = m.get("content") if isinstance(m, dict) else getattr(m, "content", m)
            chars += len(str(content or ""))
    return chars // CHARS_PER_TOKEN + OUTPUT_ESTIMATE


def used_tokens(result) -> int | None:
    """Tokens the provider reports for a reply, when the reply carries usage."""
    usage = getattr(result, "usage_metadata", None)
    return usage.get("total_tokens") if isinstance(usage, dict) else None
