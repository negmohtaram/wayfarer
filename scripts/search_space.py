"""Measure the planner's search space for every evaluation scenario.

For each scenario: feasible candidates after hard constraints (n), the pool the
current pipeline keeps, trip days (D) and stops per day (k). The count covers
choosing D*k distinct stops, assigning them to days and ordering each day:
n! / (n - D*k)!. Start times are left out, so the real space is larger.
"""

import math
from collections import Counter

from src.agents.nodes.experience_generator import _ensure_variety, score_experience
from src.agents.nodes.optimizer import _experiences_per_day
from src.evaluation.scenarios.baseline_scenarios import ALL_SCENARIOS
from src.knowledge.rag import get_city_experiences
from src.models.constraints import check_experience


def log10_perm(n: int, r: int) -> float:
    """log10 of n! / (n - r)!, or -inf when r > n (no plan exists)."""
    if r > n:
        return float("-inf")
    return (math.lgamma(n + 1) - math.lgamma(n - r + 1)) / math.log(10)


def main() -> None:
    rows = []
    for s in ALL_SCENARIOS:
        p = s.persona
        allx = get_city_experiences(p.destination.lower())
        feasible = [e for e in allx if not check_experience(e, p)]
        for e in feasible:
            e.persona_score = score_experience(e, p)
        feasible.sort(key=lambda e: e.persona_score or 0, reverse=True)
        pool = _ensure_variety(feasible, p)
        d, k = p.trip_days, _experiences_per_day(p)
        rows.append((s.id, p.destination, len(allx), len(feasible), len(pool), d, k,
                     log10_perm(len(feasible), d * k), log10_perm(len(pool), d * k)))

    print(f"{'scenario':38} {'city':10} {'all':>4} {'feas':>4} {'pool':>4} {'D':>2} {'k':>2} {'log10 feas':>10} {'log10 pool':>10}")
    for r in rows:
        print(f"{r[0]:38} {r[1]:10} {r[2]:4} {r[3]:4} {r[4]:4} {r[5]:2} {r[6]:2} {r[7]:10.1f} {r[8]:10.1f}")
    sizes = sorted(r[7] for r in rows)
    print(f"\nlog10 size over feasible candidates: min {sizes[0]:.1f}, median {sizes[len(sizes)//2]:.1f}, max {sizes[-1]:.1f}")
    print("trip lengths:", dict(sorted(Counter(r[5] for r in rows).items())))
    print("scenarios where D*k > feasible (no full plan possible):", [r[0] for r in rows if r[7] == float('-inf')])


if __name__ == "__main__":
    main()
