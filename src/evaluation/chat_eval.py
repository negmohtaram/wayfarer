"""Route the labelled chat messages and report how well the router does.

python -m src.evaluation.chat_eval --mode record   # live calls, saved
python -m src.evaluation.chat_eval --mode resume   # after an interruption: reuse saved calls
python -m src.evaluation.chat_eval                 # replay, no key needed
"""
import argparse
import asyncio
import json
import time
from collections import Counter
from pathlib import Path

from src.chat.router import Intent, route_message
from src.evaluation.chat_messages import CHAT_MESSAGES
from src.evaluation.recording import recording
from src.evaluation.stats import cluster_bootstrap_ci

FOLDER = Path("eval_results/recordings/chat_v0")
REPORT = Path("eval_results/chat_router_report.json")


async def run(pause: float, trip_days: int = 3) -> list[dict]:
    rows = []
    for n, (text, intent, day, tag) in enumerate(CHAT_MESSAGES, 1):
        start = time.perf_counter()
        route = await route_message(text, trip_days=trip_days)
        ok_intent = route.intent.value == intent
        ok = ok_intent and route.day == day
        rows.append({"text": text, "intent": intent, "day": day, "tag": tag,
                     "got_intent": route.intent.value, "got_day": route.day,
                     "error": route.error, "ok_intent": ok_intent, "ok": ok})
        mark = "✓" if ok else ("~" if ok_intent else "✗")
        got = route.intent.value + (f" day={route.day}" if route.day else "")
        want = intent + (f" day={day}" if day else "")
        print(f"{n:3} {mark} {text[:48]:48} {want:22} → {got}"
              + (f"  [{route.error}]" if route.error else ""), flush=True)
        if pause and time.perf_counter() - start > 0.05:      # live call, not a replay
            await asyncio.sleep(pause)
    return rows


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
    parser.add_argument("--pause", type=float, default=3.0, help="seconds between live calls")
    args = parser.parse_args()
    with recording(args.mode, FOLDER) as rec:
        rows = asyncio.run(run(args.pause if args.mode != "replay" else 0))
    out = report(rows)
    out.update(live_calls=rec.calls, replayed=rec.replayed, failed_calls=len(rec.failures))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(out, indent=2))
    print(f"live calls {rec.calls} · replayed {rec.replayed} · failed {len(rec.failures)} · saved {REPORT}")


if __name__ == "__main__":
    main()
