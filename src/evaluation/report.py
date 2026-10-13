"""
Aggregate baseline_v1 report from a list of EvaluationResult objects.

The report intentionally does NOT hide expected failures (must_do/must_avoid).
They are listed under "known_unsupported_features" so the signal is clear.

Runs marked invalid (a model or weather call failed or had no recording, so a fallback
planned instead) are left out of every number and listed separately. Rates and soft metrics
come with 95% cluster-bootstrap intervals over scenarios; an interval wider than +-5 points
(rates) or +-10% of the mean (soft metrics) is flagged as too wide to compare versions.
"""

import statistics
from collections import Counter, defaultdict

from src.evaluation.models import EvaluationResult
from src.evaluation.stats import cluster_bootstrap_ci

RATE_FLAG_POINTS = 5.0       # half-width in percentage points
SOFT_FLAG_SHARE = 0.10       # half-width as a share of the mean
MAX_INVALID_SHARE = 0.10     # above this the run is not usable as a baseline


def _truly_passing(r: EvaluationResult) -> bool:
    """Planned, and every error-level check passed apart from known unsupported features."""
    if r.planning_error or not r.output_itinerary:
        return False
    return all(
        v.passed or v.code in r.scenario.expected_violation_codes
        for v in r.validation_results
        if v.severity == "error"
    )


def _interval(by_scenario: dict[str, list[float]], scale: float = 1.0) -> dict:
    est, lo, hi = cluster_bootstrap_ci(by_scenario)
    return {"estimate": round(est * scale, 2), "ci_low": round(lo * scale, 2),
            "ci_high": round(hi * scale, 2), "half_width": round((hi - lo) / 2 * scale, 2),
            "scenarios": sum(1 for v in by_scenario.values() if v)}


def _rate_interval(by_scenario: dict[str, list[float]]) -> dict:
    out = _interval(by_scenario, scale=100)
    out["too_wide"] = out["half_width"] > RATE_FLAG_POINTS
    return out


def _soft_interval(by_scenario: dict[str, list[float]]) -> dict:
    if not any(by_scenario.values()):
        return {"scenarios": 0}
    out = _interval(by_scenario)
    out["too_wide"] = out["half_width"] > SOFT_FLAG_SHARE * abs(out["estimate"])
    return out


def _repeats_section(valid: list[EvaluationResult], all_results: list[EvaluationResult]) -> dict:
    """CIs over scenarios, per-scenario pass counts and the scenarios whose verdict flipped."""
    passes: dict[str, list[float]] = defaultdict(list)
    soft: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in valid:
        passes[r.scenario_id].append(1.0 if _truly_passing(r) else 0.0)
        if r.soft_metrics:
            for name in ("avg_persona_score", "total_travel_time_minutes",
                         "budget_utilization_pct", "energy_mismatch_count"):
                value = getattr(r.soft_metrics, name)
                if value is not None:
                    soft[name][r.scenario_id].append(float(value))

    per_scenario = {sid: {"passed": int(sum(v)), "valid_runs": len(v)} for sid, v in passes.items()}
    repeated = [sid for sid, v in passes.items() if len(v) >= 2]
    flips = sorted(sid for sid in repeated if 0 < sum(passes[sid]) < len(passes[sid]))
    invalid = [r for r in all_results if r.eval_error]
    invalid_share = len(invalid) / len(all_results)

    return {
        "total_runs": len(all_results),
        "valid_runs": len(valid),
        "invalid_runs": len(invalid),
        "invalid_share_pct": round(invalid_share * 100, 1),
        "publishable": invalid_share <= MAX_INVALID_SHARE,
        "valid_itinerary_rate": _rate_interval(passes) if passes else {"scenarios": 0},
        "soft_metric_intervals": {name: _soft_interval(by) for name, by in soft.items()},
        "per_scenario": per_scenario,
        "flipped_scenarios": flips,
        "flip_rate_pct": round(len(flips) / len(repeated) * 100, 1) if repeated else None,
        "invalid": [{"scenario_id": r.scenario_id, "run": r.run, "error": r.eval_error}
                    for r in invalid],
    }


def generate_report(results: list[EvaluationResult], version: str) -> dict:
    if not results:
        return {"version": version, "total_scenarios": 0}
    all_results = results
    results = [r for r in all_results if not r.eval_error]
    total = len(results)
    if total == 0:
        return {"version": version, "total_scenarios": 0,
                "repeats": _repeats_section(results, all_results)}

    planning_failed = [r for r in results if r.planning_error]
    successful = [r for r in results if r.output_itinerary and not r.planning_error]

    # Valid itinerary rate — excluding expected failures
    truly_passing = [r for r in successful if _truly_passing(r)]

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
        "repeats": _repeats_section(results, all_results),
    }


def print_report(report: dict) -> None:
    print("\n" + "=" * 60)
    print(f"  WAYFARER EVALUATION REPORT — {report['version']}")
    print("=" * 60)

    total = report["total_scenarios"]
    print(f"\nRuns scored:        {total}")
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

    rep = report.get("repeats")
    if rep:
        print(f"\nValid runs:          {rep['valid_runs']}/{rep['total_runs']} "
              f"({rep['invalid_runs']} invalid, {rep['invalid_share_pct']}%)"
              + ("" if rep["publishable"] else "  NOT PUBLISHABLE: too many invalid runs"))
        rate = rep["valid_itinerary_rate"]
        if rate.get("scenarios"):
            flag = "  [too wide]" if rate["too_wide"] else ""
            print(f"Valid rate, 95% CI:  {rate['estimate']}% [{rate['ci_low']}, {rate['ci_high']}] "
                  f"over {rate['scenarios']} scenarios{flag}")
        for name, ci in rep["soft_metric_intervals"].items():
            if ci.get("scenarios"):
                flag = "  [too wide]" if ci["too_wide"] else ""
                print(f"  {name:<26} {ci['estimate']} [{ci['ci_low']}, {ci['ci_high']}]{flag}")
        if rep["flip_rate_pct"] is not None:
            print(f"Flipped verdicts:    {len(rep['flipped_scenarios'])} scenarios "
                  f"({rep['flip_rate_pct']}% of repeated)")
            for sid in rep["flipped_scenarios"][:10]:
                ps = rep["per_scenario"][sid]
                print(f"  {sid}: passed {ps['passed']}/{ps['valid_runs']}")
        for inv in rep["invalid"][:5]:
            print(f"  invalid: {inv['scenario_id']} run {inv['run']}: {inv['error'][:70]}")

    if report["planning_errors"]:
        print(f"\nPlanning errors ({len(report['planning_errors'])}):")
        for e in report["planning_errors"][:5]:
            print(f"  {e['scenario_id']}: {e['error'][:80]}")

    print("\n" + "=" * 60)
