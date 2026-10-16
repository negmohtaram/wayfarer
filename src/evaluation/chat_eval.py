"""Route the labelled chat messages and report how many the router gets right.

python -m src.evaluation.chat_eval --mode record   # live calls, saved
python -m src.evaluation.chat_eval                 # replay, no key needed
"""
import argparse
import asyncio
from pathlib import Path

from src.chat.router import route_message
from src.evaluation.chat_messages import CHAT_MESSAGES
from src.evaluation.recording import recording

FOLDER = Path("eval_results/recordings/chat_v0")


async def run(trip_days: int = 3) -> list[tuple]:
    rows = []
    for text, intent, day in CHAT_MESSAGES:
        route = await route_message(text, trip_days=trip_days)
        rows.append((text, intent, day, route))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["record", "replay"], default="replay")
    args = parser.parse_args()
    with recording(args.mode, FOLDER) as rec:
        rows = asyncio.run(run())
    right = 0
    for text, intent, day, route in rows:
        ok = route.intent.value == intent and route.day == day
        right += ok
        got = f"{route.intent.value} day={route.day}" + (f" ERROR {route.error}" if route.error else "")
        print(f"{'✓' if ok else '✗'} {text[:45]:45} expected {intent} day={day} | got {got}")
    print(f"\n{right}/{len(rows)} correct (intent and day) · live calls {rec.calls} · "
          f"replayed {rec.replayed} · failed {len(rec.failures)}")


if __name__ == "__main__":
    main()
