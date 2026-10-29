"""Paired comparison of two planners on the same scenarios, with no model calls.

Candidate generation, both planners and the assembler run in code only, so the same scenario gives
the same candidate pool to both planners and the difference between them has no model noise. Critics
and revisions are left out on purpose: they decide nothing about which planner builds better days.

Usage:
    python -m src.evaluation.planner_compare
    python -m src.evaluation.planner_compare --a baseline --b v2a --transit-wait 10
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

import src.agents.nodes.assembler as assembler_module
from src.agents.nodes.assembler import assembler_node
from src.agents.nodes.experience_generator import experience_generator_node
from src.agents.nodes.optimizer import optimizer_node
from src.evaluation.metrics import compute_soft_metrics
from src.evaluation.scenarios.baseline_scenarios import ALL_SCENARIOS
from src.evaluation.stats import cluster_bootstrap_ci
from src.evaluation.validators import run_all_validators

EVAL_DIR = Path("eval_results")
METRICS = ["travel_min", "travel_min_with_wait", "walk_km", "transit_hops", "stops",
           "travel_per_leg", "persona_fit", "planner_ms"]
# Lower is better for these; higher is better for persona_fit and stops.
LOWER_IS_BETTER = {"travel_min", "travel_min_with_wait", "walk_km", "transit_hops", "travel_per_leg",
                   "planner_ms"}


async def plan_once(scenario, planner: str, transit_wait: int) -> dict:
    """Candidate pool, one planner and the assembler for one scenario. Returns its row of metrics."""
    weather = scenario.injected_weather
    state = {"session_id": f"compare_{scenario.id}", "persona": scenario.persona, "log": [],
             "revision_count": 0, "planner": planner}
    state = await experience_generator_node(state)
    started = time.perf_counter()
    state = await optimizer_node(state)
    planner_ms = (time.perf_counter() - started) * 1000
    if state.get("error"):
        return {"scenario_id": scenario.id, "planner": planner, "error": state["error"]}

    original = assembler_module.fetch_weather

    async def fixed_weather(city, start_date, days):
        return list(weather) if weather is not None else [None] * days

    assembler_module.fetch_weather = fixed_weather
    try:
        state = await assembler_node(state)
    finally:
        assembler_module.fetch_weather = original
    itinerary = state["itinerary"]

    results = run_all_validators(itinerary, scenario)
    expected = set(scenario.expected_violation_codes)
    hard_fails = sorted({r.code for r in results if r.severity == "error" and not r.passed})
    soft = compute_soft_metrics(itinerary, scenario)
    legs = [se for day in itinerary.days for se in day.experiences[1:]]
    transit = sum(1 for se in legs if se.travel_mode == "transit")
    return {
        "scenario_id": scenario.id,
        "planner": planner,
        "valid": not [c for c in hard_fails if c not in expected],
        "hard_fails": hard_fails,
        "travel_min": soft.total_travel_time_minutes,
        "travel_min_with_wait": soft.total_travel_time_minutes + transit_wait * transit,
        "walk_km": soft.total_walking_km,
        "transit_hops": transit,
        "stops": sum(len(day.experiences) for day in itinerary.days),
        "travel_per_leg": round(soft.total_travel_time_minutes / len(legs), 2) if legs else 0.0,
        "persona_fit": soft.avg_persona_score,
        "planner_ms": round(planner_ms, 2),
    }


def paired(rows_a: dict, rows_b: dict, metric: str) -> dict:
    """Mean of (b - a) over scenarios both planners solved, its 95% bootstrap interval, and flips."""
    diffs = {sid: [rows_b[sid][metric] - rows_a[sid][metric]] for sid in rows_a
             if sid in rows_b and rows_a[sid].get(metric) is not None and rows_b[sid].get(metric) is not None}
    est, low, high = cluster_bootstrap_ci(diffs)
    better = (lambda d: d < 0) if metric in LOWER_IS_BETTER else (lambda d: d > 0)
    wins = sorted(sid for sid, (d,) in diffs.items() if better(d))
    losses = sorted(sid for sid, (d,) in diffs.items() if d != 0 and not better(d))
    return {"mean_diff": round(est, 3), "ci_low": round(low, 3), "ci_high": round(high, 3),
            "excludes_zero": low > 0 or high < 0, "scenarios": len(diffs),
            "b_better": len(wins), "b_worse": len(losses), "ties": len(diffs) - len(wins) - len(losses),
            "b_worse_ids": losses}


async def compare(a: str, b: str, transit_wait: int) -> dict:
    scenarios = list(ALL_SCENARIOS)
    rows = {p: {} for p in (a, b)}
    for scenario in scenarios:
        for p in (a, b):
            rows[p][scenario.id] = await plan_once(scenario, p, transit_wait)
    solved = {p: {sid: r for sid, r in rows[p].items() if "error" not in r} for p in (a, b)}
    return {
        "a": a, "b": b, "scenarios": len(scenarios), "transit_wait_min": transit_wait,
        "valid": {p: sum(r["valid"] for r in solved[p].values()) for p in (a, b)},
        "errors": {p: {sid: r["error"] for sid, r in rows[p].items() if "error" in r} for p in (a, b)},
        "invalid": {p: {sid: r["hard_fails"] for sid, r in solved[p].items() if not r["valid"]} for p in (a, b)},
        "planner_ms_max": {p: max(r["planner_ms"] for r in solved[p].values()) for p in (a, b)},
        "paired": {m: paired(solved[a], solved[b], m) for m in METRICS},
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description="Paired planner comparison, no model calls")
    parser.add_argument("--a", default="baseline")
    parser.add_argument("--b", default="v2a")
    parser.add_argument("--transit-wait", type=int, default=10, help="Minutes added per transit hop")
    args = parser.parse_args()

    report = asyncio.run(compare(args.a, args.b, args.transit_wait))
    print(f"{report['b']} vs {report['a']} on {report['scenarios']} scenarios (b minus a, 95% CI)")
    print(f"valid: {report['valid']}  errors: { {p: len(e) for p, e in report['errors'].items()} }")
    for m, r in report["paired"].items():
        flag = "*" if r["excludes_zero"] else " "
        print(f"{flag} {m:22s} {r['mean_diff']:>9} [{r['ci_low']}, {r['ci_high']}]  "
              f"better {r['b_better']} / worse {r['b_worse']} / tie {r['ties']}")
    out = EVAL_DIR / f"planner_{args.b}_vs_{args.a}.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
