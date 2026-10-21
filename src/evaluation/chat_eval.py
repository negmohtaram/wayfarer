"""Route the labelled chat messages and report how well the router does.

python -m src.evaluation.chat_eval --mode record   # live calls, saved
python -m src.evaluation.chat_eval --mode resume   # after an interruption: reuse saved calls
python -m src.evaluation.chat_eval                 # replay, no key needed
python -m src.evaluation.chat_eval --mode record --concurrency 4 --name batch
                                                   # a batch: several calls in flight, paced by the limiter
"""
import argparse
import asyncio
import json
import time
from collections import Counter
from pathlib import Path

from src.agents.llm import BATCH
from src.agents.ratelimit import limiter_for
from src.config import GROQ_MODEL
from src.chat.examples import ExampleBank, render_examples
from src.chat.router import Intent, route_message
from src.evaluation.chat_messages import CHAT_MESSAGES
from src.evaluation.chat_messages_lisbon import LISBON_MESSAGES
from src.evaluation.chat_messages_lisbon import TRIP_DAYS as LISBON_DAYS
from src.evaluation.recording import recording
from src.evaluation.stats import cluster_bootstrap_ci

SETS = {"amsterdam": (CHAT_MESSAGES, 3), "lisbon": (LISBON_MESSAGES, LISBON_DAYS)}
NEAR_COPY = 0.8     # a test message this similar to a bank message is reported separately


def condition(name: str, prompt: str, retrieval: bool) -> str:
    # The first run (Amsterdam, v0, no retrieval) keeps its original folder and report names.
    if (name, prompt, retrieval) == ("amsterdam", "v0", False):
        return "v0"
    return f"{name}_{prompt}" + ("_retrieval" if retrieval else "")


async def run(messages: list, trip_days: int, pause: float, prompt: str = "v0",
              bank: ExampleBank | None = None, concurrency: int = 1) -> list[dict]:
    if concurrency > 1:
        # A batch: up to `concurrency` messages in flight; the rate limiter paces the calls.
        gate = asyncio.Semaphore(concurrency)

        async def one(n, message):
            async with gate:
                return await route_one(n, message, trip_days, prompt, bank)

        return list(await asyncio.gather(*(one(n, m) for n, m in enumerate(messages, 1))))
    rows = []
    for n, message in enumerate(messages, 1):
        start = time.perf_counter()
        rows.append(await route_one(n, message, trip_days, prompt, bank))
        if pause and time.perf_counter() - start > 0.05:      # live call, not a replay
            await asyncio.sleep(pause)
    return rows


async def route_one(n: int, message: tuple, trip_days: int, prompt: str,
                    bank: ExampleBank | None) -> dict:
    text, intent, day, tag = message
    examples = render_examples(bank.retrieve(text)) if bank else None
    route = await route_message(text, trip_days=trip_days, prompt=prompt, examples=examples,
                                policy=BATCH)
    ok_intent = route.intent.value == intent
    ok = ok_intent and route.day == day
    mark = "✓" if ok else ("~" if ok_intent else "✗")
    got = route.intent.value + (f" day={route.day}" if route.day else "")
    want = intent + (f" day={day}" if day else "")
    print(f"{n:3} {mark} {text[:48]:48} {want:22} → {got}"
          + (f"  [{route.error}]" if route.error else ""), flush=True)
    return {"text": text, "intent": intent, "day": day, "tag": tag, "got_day_ref": route.day_ref,
            "got_intent": route.intent.value, "got_day": route.day,
            "error": route.error, "ok_intent": ok_intent, "ok": ok}


def report(rows: list[dict]) -> dict:
    n = len(rows)
    intents = [i.value for i in Intent]
    acc = sum(r["ok"] for r in rows) / n
    acc_intent = sum(r["ok_intent"] for r in rows) / n
    _, lo, hi = cluster_bootstrap_ci({r["text"]: [float(r["ok_intent"])] for r in rows})
    print(f"\nIntent correct: {acc_intent:.1%} [95% CI {lo:.1%}, {hi:.1%}] · intent and day: {acc:.1%} · n={n}")

    print("\nPer intent (recall = expected intent found, precision = predicted intent right):")
    for i in intents:
        exp = [r for r in rows if r["intent"] == i]
        got = [r for r in rows if r["got_intent"] == i]
        rec = sum(r["ok_intent"] for r in exp) / len(exp) if exp else 0
        prec = sum(r["ok_intent"] for r in got) / len(got) if got else 0
        print(f"  {i:15} n={len(exp):3}  recall {rec:5.0%}  precision {prec:5.0%}")

    print("\nConfusion (rows expected, columns predicted):")
    short = [i[:6] for i in intents]
    print(" " * 16 + " ".join(f"{s:>6}" for s in short))
    matrix = {}
    for i in intents:
        c = Counter(r["got_intent"] for r in rows if r["intent"] == i)
        matrix[i] = {j: c.get(j, 0) for j in intents}
        print(f"  {i:14}" + " ".join(f"{c.get(j, 0) or '.':>6}" for j in intents))

    print("\nBy tag:")
    for tag in sorted({r["tag"] for r in rows}):
        part = [r for r in rows if r["tag"] == tag]
        print(f"  {tag or 'plain':10} {sum(r['ok_intent'] for r in part)}/{len(part)} intent correct")
    wrong_day = [r for r in rows if r["ok_intent"] and not r["ok"]]
    print(f"\nRight intent, wrong day: {len(wrong_day)}")
    for r in wrong_day:
        print(f"  {r['text'][:60]:60} expected day={r['day']} got day={r['got_day']}")
    fell_back = sum(1 for r in rows if r["error"])
    print(f"Router fell back after two invalid calls: {fell_back}")
    return {"n": n, "intent_accuracy": acc_intent, "intent_ci": [lo, hi],
            "intent_and_day_accuracy": acc, "confusion": matrix, "fallbacks": fell_back, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["record", "replay", "resume"], default="replay")
    parser.add_argument("--set", choices=list(SETS), default="amsterdam")
    parser.add_argument("--prompt", choices=["v0", "v1"], default="v0")
    parser.add_argument("--retrieval", action="store_true", help="add similar labelled examples")
    parser.add_argument("--pause", type=float, default=3.0, help="seconds between live calls")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="messages in flight at once; above 1 the rate limiter alone paces calls")
    parser.add_argument("--name", default="", help="suffix for a separate recording folder and report")
    args = parser.parse_args()
    if args.set == "amsterdam" and args.retrieval:
        raise SystemExit("the Amsterdam set is the example bank: testing on it would leak")
    messages, trip_days = SETS[args.set]
    name = condition(args.set, args.prompt, args.retrieval) + (f"_{args.name}" if args.name else "")
    folder = Path(f"eval_results/recordings/chat_{name}")
    out_path = Path("eval_results/chat_router_report.json" if name == "v0"
                    else f"eval_results/chat_router_{name}.json")
    bank = ExampleBank() if args.retrieval or args.set != "amsterdam" else None
    with recording(args.mode, folder) as rec:
        started = time.perf_counter()
        pause = args.pause if args.mode != "replay" and args.concurrency == 1 else 0
        rows = asyncio.run(run(messages, trip_days, pause, args.prompt,
                               bank if args.retrieval else None, args.concurrency))
        minutes = (time.perf_counter() - started) / 60
    if bank:
        for row in rows:
            row["closest_in_bank"] = round(bank.closest(row["text"])[0], 3)
        near = [r for r in rows if r["closest_in_bank"] >= NEAR_COPY]
        rest = [r for r in rows if r["closest_in_bank"] < NEAR_COPY]
        print(f"Near copies of a bank message (similarity >= {NEAR_COPY}): {len(near)}; "
              f"intent correct without them: {sum(r['ok_intent'] for r in rest)}/{len(rest)}")
    out = report(rows)
    limiter = limiter_for(GROQ_MODEL)
    out.update(condition=name, live_calls=rec.calls, replayed=rec.replayed, failed_calls=len(rec.failures),
               concurrency=args.concurrency, minutes=round(minutes, 2),
               live_calls_per_minute=round(rec.calls / minutes, 1) if rec.calls else None,
               attempts_admitted=limiter.admitted, rate_limited_429=limiter.drains)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"live calls {rec.calls} · replayed {rec.replayed} · failed {len(rec.failures)} · "
          f"{minutes:.1f} min · attempts {limiter.admitted} · 429s {limiter.drains} · saved {out_path}")


if __name__ == "__main__":
    main()
