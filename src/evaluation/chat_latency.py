"""Turn latency of the chat handler, one step after another vs. independent steps together.

No network: each step sleeps for a sampled time. Model calls follow a lognormal fitted to
measured Groq call times (median 1.23 s, p95 1.73 s, 100 calls); weather uses an assumed
median of 0.3 s with a longer tail. Routes come from the labelled Lisbon messages. Both
versions get the same sampled times per message, so the comparison is paired.

    python -m src.evaluation.chat_latency [--scale 0.05] [--seed 0]
"""
import argparse
import asyncio
import math
import random
import statistics
import time

from src.chat.handler import TurnSteps, handle_turn, handle_turn_sequential
from src.chat.router import Intent, Route
from src.evaluation.chat_messages_lisbon import LISBON_MESSAGES, TRIP_DAYS

MODEL_MEDIAN, MODEL_P95 = 1.23, 1.73
WEATHER_MEDIAN, WEATHER_SIGMA = 0.30, 0.5
SESSION_S, SCOPE_S = 0.005, 0.001


def _lognormal(rng: random.Random, median: float, sigma: float) -> float:
    return median * math.exp(rng.gauss(0, sigma))


def sample_times(rng: random.Random) -> dict:
    model_sigma = math.log(MODEL_P95 / MODEL_MEDIAN) / 1.645
    return {"route": _lognormal(rng, MODEL_MEDIAN, model_sigma),
            "extract": _lognormal(rng, MODEL_MEDIAN, model_sigma),
            "weather": _lognormal(rng, WEATHER_MEDIAN, WEATHER_SIGMA)}


def fake_steps(route: Route, times: dict, scale: float) -> TurnSteps:
    async def wait(seconds):
        await asyncio.sleep(seconds * scale)

    async def load_session(session_id):
        await wait(SESSION_S)
        return {"trip_days": TRIP_DAYS, "destination": "lisbon"}

    async def route_step(text, trip_days):
        await wait(times["route"])
        return route

    async def extract(text, r, trip):
        await wait(times["extract"])
        return {}

    async def scope(r, trip):
        await wait(SCOPE_S)
        return {"day": r.day}

    async def weather(city, sc):
        await wait(times["weather"])
        return {}

    return TurnSteps(load_session, route_step, extract, scope, weather)


async def measure(handler, route: Route, times: dict, scale: float) -> float:
    start = time.perf_counter()
    await handler("", "lisbon", fake_steps(route, times, scale))
    return (time.perf_counter() - start) / scale


def pct(xs: list[float], q: float) -> float:
    return statistics.quantiles(xs, n=100, method="inclusive")[q - 1]


async def main(scale: float, seed: int) -> None:
    rng = random.Random(seed)
    rows = []
    for text, intent, day, _ in LISBON_MESSAGES:
        route, times = Route(intent=Intent(intent), day=day), sample_times(rng)
        seq = await measure(handle_turn_sequential, route, times, scale)
        con = await measure(handle_turn, route, times, scale)
        both = route.intent.value in {"change_setting", "what_if"} and day is not None
        rows.append((seq, con, both))

    def line(name, sel):
        seq, con = [r[0] for r in sel], [r[1] for r in sel]
        p50s, p50c, p95s, p95c = pct(seq, 50), pct(con, 50), pct(seq, 95), pct(con, 95)
        print(f"| {name} | {len(sel)} | {p50s:.2f} | {p50c:.2f} | {1 - p50c / p50s:.0%} "
              f"| {p95s:.2f} | {p95c:.2f} | {1 - p95c / p95s:.0%} |")

    print("| messages | n | p50 one by one (s) | p50 together (s) | saved | p95 one by one (s) | p95 together (s) | saved |")
    print("|---|---|---|---|---|---|---|---|")
    line("all", rows)
    line("change + day (both branches)", [r for r in rows if r[2]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=float, default=0.05, help="sleep this fraction of each sampled time")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    asyncio.run(main(args.scale, args.seed))
