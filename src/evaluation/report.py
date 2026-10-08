"""
Aggregate baseline_v1 report from a list of EvaluationResult objects.

The report intentionally does NOT hide expected failures (must_do/must_avoid).
They are listed under "known_unsupported_features" so the signal is clear.
"""

import statistics
from collections import Counter, defaultdict

from src.evaluation.models import EvaluationResult


def generate_report(results: list[EvaluationResult], version: str) -> dict:
    total = len(results)
    if total == 0:
        return {"version": version, "total_scenarios": 0}

    planning_failed = [r for r in results if r.planning_error]
    successful = [r for r in results if r.output_itinerary and not r.planning_error]

    # Valid itinerary rate — excluding expected failures
    truly_passing = [
        r for r in successful
        if all(
            v.passed or v.code in r.scenario.expected_violation_codes
            for v in r.validation_results
            if v.severity == "error"
        )
    ]

    # Violation counts by code (unexpected failures only)
    violation_counts: Counter = Counter()
    expected_violation_counts: Counter = Counter()
    for r in successful:
        for v in r.validation_results:
            if not v.passed and v.severity == "error":
                if v.code in r.scenario.expected_violation_codes:
                    expected_violation_counts[v.code] += 1
                else:
                    violation_counts[v.code] += 1

    # Soft metrics aggregates
    travel_times = [r.soft_metrics.total_travel_time_minutes for r in successful if r.soft_metrics]
    persona_scores = [r.soft_metrics.avg_persona_score for r in successful
                      if r.soft_metrics and r.soft_metrics.avg_persona_score is not None]
    budget_utils = [r.soft_metrics.budget_utilization_pct for r in successful if r.soft_metrics]
    energy_mismatches = [r.soft_metrics.energy_mismatch_count for r in successful if r.soft_metrics]

    # Operational metrics
    runtimes = [r.operational_metrics.total_runtime_seconds for r in results if not r.planning_error]
    llm_calls = [r.operational_metrics.llm_call_count for r in successful]
    revision_counts = [r.operational_metrics.revision_count for r in successful]

    def _stats(vals: list[float]) -> dict:
        if not vals:
            return {"count": 0}
        return {
            "count": len(vals),
            "mean": round(statistics.mean(vals), 2),
            "median": round(statistics.median(vals), 2),
            "p95": round(sorted(vals)[int(len(vals) * 0.95)], 2),
            "min": round(min(vals), 2),
            "max": round(max(vals), 2),
        }

    def _distribution(vals: list[int]) -> dict:
        if not vals:
            return {}
        c: Counter = Counter(vals)
        return {str(k): v for k, v in sorted(c.items())}

    return {
        "version": version,
        "total_scenarios": total,
        "planning_failures": len(planning_failed),
        "planning_failure_rate_pct": round(len(planning_failed) / total * 100, 1),
        "valid_itinerary_rate_pct": round(len(truly_passing) / total * 100, 1),
        "unexpected_violation_counts": dict(violation_counts.most_common()),
        "known_unsupported_features": dict(expected_violation_counts.most_common()),
        "top_unexpected_failures": [
            {"code": code, "count": count, "rate_pct": round(count / total * 100, 1)}
            for code, count in violation_counts.most_common(5)
        ],
        "soft_metrics": {
            "avg_persona_score": _stats(persona_scores),
            "total_travel_time_minutes": _stats(travel_times),
            "budget_utilization_pct": _stats(budget_utils),
            "energy_mismatch_count": _stats(energy_mismatches),
        },
        "operational": {
            "runtime_seconds": _stats(runtimes),
            "llm_calls": _stats([float(x) for x in llm_calls]),
            "revision_count_distribution": _distribution(revision_counts),
        },
        "planning_errors": [
            {"scenario_id": r.scenario_id, "error": r.planning_error}
            for r in planning_failed
        ],
    }


def print_report(report: dict) -> None:
    print("\n" + "=" * 60)
    print(f"  WAYFARER EVALUATION REPORT — {report['version']}")
    print("=" * 60)

    total = report["total_scenarios"]
    print(f"\nScenarios run:      {total}")
    print(f"Planning failures:  {report['planning_failures']} ({report['planning_failure_rate_pct']}%)")
    print(f"Valid rate:         {report['valid_itinerary_rate_pct']}% (unexpected failures only)")

    if report["top_unexpected_failures"]:
        print("\nTop unexpected violations:")
        for f in report["top_unexpected_failures"]:
            print(f"  {f['code']:<35} {f['count']:>3} scenarios ({f['rate_pct']}%)")
    else:
        print("\nNo unexpected hard violations.")

    if report["known_unsupported_features"]:
        print("\nKnown unsupported (expected failures):")
        for code, count in report["known_unsupported_features"].items():
            print(f"  {code:<35} {count:>3} scenarios")

    sm = report["soft_metrics"]
    print("\nSoft metrics (successful runs):")
    if sm["avg_persona_score"].get("count"):
        print(f"  avg_persona_score:       mean={sm['avg_persona_score']['mean']:.3f}, "
              f"p95={sm['avg_persona_score']['p95']:.3f}")
    if sm["total_travel_time_minutes"].get("count"):
        print(f"  travel_time_min:         mean={sm['total_travel_time_minutes']['mean']:.0f}, "
              f"p95={sm['total_travel_time_minutes']['p95']:.0f}")
    if sm["budget_utilization_pct"].get("count"):
        print(f"  budget_utilization_pct:  mean={sm['budget_utilization_pct']['mean']:.1f}%, "
              f"p95={sm['budget_utilization_pct']['p95']:.1f}%")

    op = report["operational"]
    if op["runtime_seconds"].get("count"):
        print(f"\nRuntime (seconds):   mean={op['runtime_seconds']['mean']:.1f}, "
              f"p95={op['runtime_seconds']['p95']:.1f}")
    if op["llm_calls"].get("count"):
        print(f"LLM calls/run:       mean={op['llm_calls']['mean']:.1f}")
    if op["revision_count_distribution"]:
        dist = op["revision_count_distribution"]
        print(f"Revision cycles:     {dict(dist)}")

    if report["planning_errors"]:
        print(f"\nPlanning errors ({len(report['planning_errors'])}):")
        for e in report["planning_errors"][:5]:
            print(f"  {e['scenario_id']}: {e['error'][:80]}")

    print("\n" + "=" * 60)
