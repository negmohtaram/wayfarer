"""
Replay an evaluation run by scenario ID.

Loads the latest saved result for the given scenario and re-runs the planning
graph with the same persona and injected weather. The LLM is still live so
soft metrics and critic scores may differ slightly between replays — that
variance is itself informative.

Usage:
    python -m src.evaluation.replay --scenario pace_slow_solo_amsterdam
    python -m src.evaluation.replay --scenario pace_slow_solo_amsterdam --from-file eval_results/runs/pace_slow_solo_amsterdam_20260904_120000.json
"""

import argparse
import asyncio
import json
from pathlib import Path

from src.evaluation.runner import EVAL_DIR, run_scenario, save_result
from src.evaluation.scenarios.baseline_scenarios import SCENARIO_BY_ID
from src.models.persona import TravelerPersona


def load_latest_result(scenario_id: str) -> dict | None:
    index_path = EVAL_DIR / "index.json"
    if not index_path.exists():
        return None
    index = json.loads(index_path.read_text())
    paths = index.get(scenario_id, [])
    if not paths:
        return None
    latest = sorted(paths)[-1]
    p = Path(latest)
    if not p.exists():
        return None
    return json.loads(p.read_text())


async def replay_scenario(scenario_id: str, from_file: str | None = None) -> None:
    if from_file:
        saved = json.loads(Path(from_file).read_text())
        snapshot = saved.get("input_snapshot", {})
        persona = TravelerPersona(**snapshot["persona"])
        injected_weather = snapshot.get("injected_weather")
        scenario = SCENARIO_BY_ID.get(scenario_id)
        if scenario is None:
            print(f"Unknown scenario ID: {scenario_id}")
            return
        # Override persona from saved snapshot for exact replay
        scenario = scenario.model_copy(update={"persona": persona, "injected_weather": injected_weather})
    else:
        scenario = SCENARIO_BY_ID.get(scenario_id)
        if scenario is None:
            print(f"Unknown scenario ID: {scenario_id}")
            print(f"Available: {', '.join(SCENARIO_BY_ID)}")
            return
        saved = load_latest_result(scenario_id)
        if saved is None:
            print(f"No saved result found for '{scenario_id}'. Run the scenario first.")
            return
        snapshot = saved.get("input_snapshot", {})
        if snapshot.get("persona"):
            persona = TravelerPersona(**snapshot["persona"])
            injected_weather = snapshot.get("injected_weather")
            scenario = scenario.model_copy(update={"persona": persona, "injected_weather": injected_weather})

    print(f"Replaying: {scenario.id}")
    print(f"  Persona: {scenario.persona.destination}, {scenario.persona.trip_days} days, "
          f"pace={scenario.persona.pace}, budget={scenario.persona.budget_per_day}")

    result = await run_scenario(scenario)
    path = save_result(result)

    status = "✓ PASSED" if result.passed_all_hard_validators else "✗ FAILED"
    print(f"\n{status} — {result.operational_metrics.total_runtime_seconds:.1f}s")
    print(f"Saved: {path}")

    if result.planning_error:
        print(f"Planning error: {result.planning_error}")
        return

    failures = [r for r in result.validation_results if not r.passed and r.severity == "error"]
    if failures:
        print(f"\nHard violations ({len(failures)}):")
        for v in failures:
            expected_fail = v.code in scenario.expected_violation_codes
            marker = "[expected]" if expected_fail else "[unexpected]"
            print(f"  {marker} {v.code}: {v.message}")

    if result.soft_metrics:
        m = result.soft_metrics
        print("\nSoft metrics:")
        print(f"  avg_persona_score:      {m.avg_persona_score}")
        print(f"  total_travel_time_min:  {m.total_travel_time_minutes}")
        print(f"  experience_diversity:   {m.experience_type_diversity}/6")
        print(f"  localness_delta:        {m.localness_delta}")
        print(f"  budget_utilization_pct: {m.budget_utilization_pct}%")
        print(f"  energy_mismatches:      {m.energy_mismatch_count}")


def main():
    parser = argparse.ArgumentParser(description="Replay a Wayfarer evaluation scenario")
    parser.add_argument("--scenario", required=True, help="Scenario ID to replay")
    parser.add_argument("--from-file", help="Path to specific result file to replay from")
    args = parser.parse_args()
    asyncio.run(replay_scenario(args.scenario, args.from_file))


if __name__ == "__main__":
    main()
