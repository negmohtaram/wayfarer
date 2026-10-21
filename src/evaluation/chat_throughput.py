"""How fast the chat test set runs against Groq's limits: a fixed pause vs the rate limiter.

python -m src.evaluation.chat_throughput

No network. A fake provider enforces 30 requests and 8,000 tokens in any rolling minute and
answers 429 with Retry-After when a call would cross either. Each call uses the real messages
and the real token usage from the Lisbon recordings, and takes a model time from the lognormal
fitted to real Groq calls. Time runs faster than real time by SCALE.
"""
import asyncio
import json
import logging
import math
import random
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import groq
import httpx

from src.agents import llm, ratelimit
from src.agents.llm import BATCH, LLMCallError
from src.agents.ratelimit import Limiter
from src.config import GROQ_MODEL, RATE_LIMITS
from src.evaluation.chat_latency import MODEL_MEDIAN, MODEL_P95

RECORDINGS = Path("eval_results/recordings/chat_lisbon_v1_retrieval")
SCALE = 200.0
PAUSE = 3.0                     # the fixed pause chat_eval used between live calls


class Clock:
    def __init__(self):
        self.start = time.monotonic()

    def now(self) -> float:
        return (time.monotonic() - self.start) * SCALE

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds / SCALE)


class FakeGroq:
    """Refuses a call that would put more than rpm requests or tpm tokens in the last 60 s."""

    def __init__(self, clock: Clock, rpm: int, tpm: int):
        self.clock, self.rpm, self.tpm = clock, rpm, tpm
        self.window: deque[tuple[float, int]] = deque()
        self.refused = 0
        self.tokens_served = 0

    def _rate_limited(self, wait: float) -> groq.RateLimitError:
        response = httpx.Response(429, headers={"retry-after": str(math.ceil(wait))},
                                  request=httpx.Request("POST", "https://fake.groq"))
        return groq.RateLimitError("rate limit reached", response=response, body=None)

    async def call(self, tokens: int, model_time: float):
        now = self.clock.now()
        while self.window and now - self.window[0][0] >= 60:
            self.window.popleft()
        used = sum(t for _, t in self.window)
        if len(self.window) + 1 > self.rpm or used + tokens > self.tpm:
            self.refused += 1
            raise self._rate_limited(60 - (now - self.window[0][0]))
        self.window.append((now, tokens))
        self.tokens_served += tokens
        await self.clock.sleep(model_time)
        return SimpleNamespace(usage_metadata={"total_tokens": tokens})


def load_calls() -> list[tuple[list, int]]:
    calls = []
    for f in sorted(RECORDINGS.glob("*.json")):
        rec = json.loads(f.read_text())
        usage = rec["result"]["data"]["usage_metadata"]
        calls.append((rec["messages"], usage["total_tokens"]))
    return calls


async def simulate(calls, concurrency: int, pause: float, limited: bool, seed: int) -> dict:
    clock = Clock()
    rng = random.Random(seed)
    sigma = math.log(MODEL_P95 / MODEL_MEDIAN) / 1.645
    rpm, tpm = RATE_LIMITS[GROQ_MODEL]
    provider = FakeGroq(clock, rpm, tpm)
    llm._now, llm._sleep = clock.now, clock.sleep
    ratelimit._now, ratelimit._sleep = clock.now, clock.sleep
    llm._random = random.Random(seed)
    ratelimit.reset()
    if not limited:
        ratelimit._limiters[GROQ_MODEL] = Limiter(10**9, 10**12)     # the old code: no limiter
    gate = asyncio.Semaphore(concurrency)
    failed = 0

    async def one(messages, tokens):
        nonlocal failed
        model_time = MODEL_MEDIAN * math.exp(rng.gauss(0, sigma))
        async with gate:
            try:
                await llm._retry(lambda m: provider.call(tokens, model_time), (messages,), {}, BATCH,
                                 GROQ_MODEL)
            except LLMCallError:
                failed += 1
            if pause:
                await clock.sleep(pause)

    await asyncio.gather(*(one(m, t) for m, t in calls))
    minutes = clock.now() / 60
    done = len(calls) - failed
    return {"minutes": minutes, "calls_per_min": done / minutes, "tokens_per_min": provider.tokens_served / minutes,
            "refused_429": provider.refused, "failed": failed}


def main() -> None:
    logging.getLogger("src.agents.llm").setLevel(logging.ERROR)   # one line per retry is noise here
    calls = load_calls()
    tokens = sorted(t for _, t in calls)
    print(f"{len(calls)} calls · tokens per call median {tokens[len(tokens) // 2]} · max {tokens[-1]} · "
          f"limits {RATE_LIMITS[GROQ_MODEL][0]} RPM, {RATE_LIMITS[GROQ_MODEL][1]:,} TPM")
    conditions = [("fixed 3 s pause, one at a time", 1, PAUSE, False),
                  ("no pause, 4 in flight, no limiter", 4, 0, False),
                  ("limiter, 4 in flight", 4, 0, True)]
    print(f"\n{'condition':36} {'minutes':>8} {'calls/min':>10} {'tokens/min':>11} {'429s':>5} {'failed':>7}")
    for name, conc, pause, limited in conditions:
        runs = [asyncio.run(simulate(calls, conc, pause, limited, seed)) for seed in (0, 1, 2)]
        avg = {k: sum(r[k] for r in runs) / len(runs) for k in runs[0]}
        print(f"{name:36} {avg['minutes']:8.1f} {avg['calls_per_min']:10.1f} {avg['tokens_per_min']:11,.0f} "
              f"{avg['refused_429']:5.1f} {avg['failed']:7.1f}")
    print("\nMean of 3 seeds. The fake counts tokens when a call starts; Groq's exact accounting may differ.")


if __name__ == "__main__":
    main()
