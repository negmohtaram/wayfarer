import asyncio
import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Coroutine

import groq
from langchain_groq import ChatGroq

from src.agents.ratelimit import NoCapacity, estimate_tokens, limiter_for, used_tokens
from src.config import GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger(__name__)


class LazyModel:
    """A Groq chat model that is only built when it is first called.

    Building ChatGroq fails without an API key. Deferring it to ainvoke keeps that failure
    inside call_with_retry, where the recorder sees it, and lets replays run with no key.
    """

    def __init__(self, temperature: float, steps: tuple = ()):
        self.temperature = temperature
        self.steps = steps                     # (method, args, kwargs) applied after building

    def with_structured_output(self, *args: Any, **kwargs: Any) -> "LazyModel":
        return LazyModel(self.temperature, self.steps + (("with_structured_output", args, kwargs),))

    def bind_tools(self, *args: Any, **kwargs: Any) -> "LazyModel":
        return LazyModel(self.temperature, self.steps + (("bind_tools", args, kwargs),))

    def build(self) -> Any:
        # max_retries=0: the SDK would otherwise retry inside our retries (up to 3 x 3 calls),
        # and those attempts would never reach our logs or the recorder.
        model: Any = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=self.temperature,
                              max_retries=0)
        for method, args, kwargs in self.steps:
            model = getattr(model, method)(*args, **kwargs)
        return model

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        return await self.build().ainvoke(*args, **kwargs)


def get_llm(temperature: float = 0.7) -> LazyModel:
    return LazyModel(temperature)


TRANSIENT = "transient"      # time can fix it: rate limit, server error, timeout, dropped connection
PERMANENT = "permanent"      # the same request fails the same way: bad request, bad key, no access

# Replaced in tests so no test waits for real time.
_sleep = asyncio.sleep
_now = time.monotonic
_random = random.Random()


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay: float = 1.0        # first backoff cap; doubles each attempt
    max_delay: float = 8.0
    attempt_timeout: float = 60.0  # one call
    deadline: float = 300.0        # all attempts and waits together


BATCH = RetryPolicy()
CHAT = RetryPolicy(attempt_timeout=10.0, deadline=20.0)   # a traveler will not wait longer


class LLMCallError(Exception):
    """The call failed for good. attempts lists (kind, status, error) for every try."""

    def __init__(self, kind: str, status: int | None, attempts: list[tuple[str, int | None, str]]):
        self.kind, self.status, self.attempts = kind, status, attempts
        last = attempts[-1][2] if attempts else ""
        super().__init__(f"{kind} error after {len(attempts)} attempt(s), status {status}: {last}")


def classify(exc: BaseException) -> tuple[str, int | None]:
    """Whether waiting and calling again can help, and the HTTP status if there was one.

    Unknown exceptions count as permanent: a bug in our own code must not be retried.
    """
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError, groq.APIConnectionError)):
        return TRANSIENT, None
    status = getattr(exc, "status_code", None)
    if status == 429 or (isinstance(status, int) and status >= 500):
        return TRANSIENT, status
    return PERMANENT, status


def retry_after(exc: BaseException) -> float | None:
    """Seconds from the Retry-After header, when the provider sent one as a number."""
    response = getattr(exc, "response", None)
    value = getattr(response, "headers", {}).get("retry-after") if response is not None else None
    try:
        return max(0.0, float(value)) if value is not None else None
    except ValueError:
        return None


async def call_with_retry(
    fn: Callable[..., Coroutine[Any, Any, Any]],
    *args: Any,
    policy: RetryPolicy = BATCH,
    meta: dict | None = None,
    **kwargs: Any,
) -> Any:
    """Call an async LLM function, retrying only transient failures, within the policy's deadline.

    meta identifies the call (site, model, temperature, output schema). When a recorder is
    active (evaluation), the call is recorded or replayed; a call without meta is refused so
    no model call can escape recording.
    """
    from src.evaluation.recording import active

    recorder = active()
    model = (meta or {}).get("model", GROQ_MODEL)
    if recorder is not None:
        if meta is None:
            raise ValueError("model call without meta while recording: add meta at the call site")
        messages = args[0] if args else kwargs.get("input")
        return await recorder.call(meta, messages, lambda: _retry(fn, args, kwargs, policy, model))
    return await _retry(fn, args, kwargs, policy, model)


async def _retry(fn, args, kwargs, policy: RetryPolicy, model: str = GROQ_MODEL):
    start = _now()
    attempts: list[tuple[str, int | None, str]] = []
    limiter = limiter_for(model)
    reserved = estimate_tokens(args[0] if args else kwargs.get("input"))
    for attempt in range(policy.max_attempts):
        left = policy.deadline - (_now() - start)
        try:
            # Every attempt is a request the provider counts, so every attempt is admitted.
            await limiter.acquire(reserved, max_wait=left)
        except NoCapacity as exc:
            attempts.append((TRANSIENT, None, f"no rate-limit capacity before the deadline: {exc}"))
            raise LLMCallError(TRANSIENT, None, attempts) from None
        left = policy.deadline - (_now() - start)
        try:
            result = await asyncio.wait_for(fn(*args, **kwargs), timeout=min(policy.attempt_timeout, left))
        except Exception as exc:       # CancelledError is not an Exception: cancellation is never retried
            kind, status = classify(exc)
            attempts.append((kind, status, repr(exc)[:300]))
            if status == 429:
                limiter.drain()
            if kind == PERMANENT or attempt == policy.max_attempts - 1:
                raise LLMCallError(kind, status, attempts) from exc
            # Full jitter: a random wait up to the backoff cap, so clients that failed together
            # do not all come back together. Retry-After from the provider is a minimum.
            delay = _random.uniform(0, min(policy.max_delay, policy.base_delay * 2 ** attempt))
            wait = retry_after(exc)
            if wait is not None:
                delay = max(delay, wait)
            left = policy.deadline - (_now() - start)
            if delay >= left:
                raise LLMCallError(kind, status, attempts) from exc
            logger.warning("LLM call failed (attempt %d/%d, %s, status %s): %s; retrying in %.1fs",
                           attempt + 1, policy.max_attempts, kind, status, exc, delay)
            await _sleep(delay)
        else:
            used = used_tokens(result)
            if used is not None:
                limiter.reconcile(reserved, used)
            return result
