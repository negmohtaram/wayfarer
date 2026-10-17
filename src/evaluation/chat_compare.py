"""Paired comparison of two chat router runs on the same messages.

Usage: python -m src.evaluation.chat_compare lisbon_v0 lisbon_v1_retrieval
Reads eval_results/chat_router_<name>.json and prints accuracy, the messages that flipped
each way, and an exact McNemar test on the flips.
"""
import json
import sys
from math import comb
from pathlib import Path


def load(name: str) -> dict:
    return {r["text"]: r for r in json.loads(Path(f"eval_results/chat_router_{name}.json").read_text())["rows"]}


def mcnemar_p(fixed: int, broke: int) -> float:
    """Two-sided exact test: if both versions were equally good, each flip would go either way
    with chance 1/2, like a coin."""
    n = fixed + broke
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(fixed, broke) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main() -> None:
    a_name, b_name = sys.argv[1], sys.argv[2]
    a, b = load(a_name), load(b_name)
    if set(a) != set(b):
        raise SystemExit("the two runs used different messages; a paired comparison needs the same ones")
    n = len(a)
    for key, label in (("ok_intent", "intent"), ("ok", "intent and day")):
        acc_a = sum(r[key] for r in a.values()) / n
        acc_b = sum(r[key] for r in b.values()) / n
        fixed = [t for t in a if not a[t][key] and b[t][key]]
        broke = [t for t in a if a[t][key] and not b[t][key]]
        print(f"\n{label}: {a_name} {acc_a:.1%} -> {b_name} {acc_b:.1%} (n={n}) · "
              f"fixed {len(fixed)}, broke {len(broke)}, McNemar p={mcnemar_p(len(fixed), len(broke)):.3f}")
        for title, texts in (("fixed", fixed), ("broke", broke)):
            for t in texts:
                print(f"  {title:5} {t[:58]:58} want {a[t]['intent']}/{a[t]['day']} · "
                      f"{a[t]['got_intent']}/{a[t]['got_day']} -> {b[t]['got_intent']}/{b[t]['got_day']}")


if __name__ == "__main__":
    main()
