"""
Evaluation runner.

Usage:
    # Run all scenarios and save results
    python -m src.evaluation.runner

    # Run a single scenario by ID
    python -m src.evaluation.runner --scenario pace_slow_solo_amsterdam

    # Run all and write baseline_v1 report
    python -m src.evaluation.runner --report

    # Record 3 live runs of every scenario, then replay them with no network
    python -m src.evaluation.runner --mode record --repeats 3 --report
    python -m src.evaluation.runner --mode replay --repeats 3 --report

Recordings go to eval_results/recordings/<version>/run_<k>/<scenario_id>/, one folder per
scenario and run, so any single run of any scenario can be replayed on its own.

    # After a rate limit: record again, replaying every scenario run that already recorded
    # cleanly and calling the model only for the missing or invalid ones
    python -m src.evaluation.runner --mode record --repeats 3 --resume --report
"""

import argparse
import asyncio
import contextlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from src.evaluation.instrumentation import run_planning, run_replanning
from src.evaluation.metrics import compute_soft_metrics
from src.evaluation.models import EvaluationResult, OperationalMetrics
from src.evaluation.recording import recording
from src.evaluation.scenarios.baseline_scenarios import ALL_SCENARIOS, SCENARIO_BY_ID
from src.evaluation.validators import passed_all_hard, run_all_validators
from src.agents.state import PlanningState

EVAL_DIR = Path("eval_results")
PLANNER_VERSION = "baseline_v1"


async def run_scenario(scenario, version: str = PLANNER_VERSION) -> EvaluationResult:
    run_at = datetime.now(timezone.utc).isoformat()
    session_id = f"eval_{scenario.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    input_snapshot = {
        "scenario_id": scenario.id,
        "persona": scenario.persona.model_dump(),
        "injected_weather": scenario.injected_weather,
    }

    initial_itinerary = None
    initial_itinerary_dict = None
    output_itinerary_dict = None
    validation_results = []
    planning_error = None

    if scenario.disruption:
        # Step 1: run planning to get a base itinerary
        state: PlanningState = {
            "session_id": session_id,
            "persona": scenario.persona,
            "log": [],
            "revision_count": 0,
        }
        initial_itinerary, plan_metrics = await run_planning(
            state, injected_weather=scenario.injected_weather
        )

        if initial_itinerary is None:
            planning_error = plan_metrics.errors[0] if plan_metrics.errors else "Planning failed"
            return EvaluationResult(
                scenario_id=scenario.id,
                planner_version=version,
                run_at=run_at,
                scenario=scenario,
                input_snapshot=input_snapshot,
                operational_metrics=plan_metrics,
                passed_all_hard_validators=False,
                planning_error=planning_error,
            )

        initial_itinerary_dict = initial_itinerary.model_dump()

        # Step 2: run replanning
        output_itinerary, replan_metrics = await run_replanning(
            initial_itinerary=initial_itinerary,
            disruption_message=scenario.disruption.message,
            completed_day_indices=scenario.disruption.completed_day_indices,
            session_id=session_id,
        )

        op_metrics = OperationalMetrics(
            total_runtime_seconds=round(
                plan_metrics.total_runtime_seconds + replan_metrics.total_runtime_seconds, 3
            ),
            node_runtimes={
                **{f"planning/{k}": v for k, v in plan_metrics.node_runtimes.items()},
                **{f"replanning/{k}": v for k, v in replan_metrics.node_runtimes.items()},
            },
            llm_call_count=plan_metrics.llm_call_count + replan_metrics.llm_call_count,
            revision_count=plan_metrics.revision_count,
            errors=plan_metrics.errors + replan_metrics.errors,
        )

        if output_itinerary is None:
            planning_error = op_metrics.errors[-1] if op_metrics.errors else "Replanning failed"
            return EvaluationResult(
                scenario_id=scenario.id,
                planner_version=version,
                run_at=run_at,
                scenario=scenario,
                initial_itinerary=initial_itinerary_dict,
                input_snapshot=input_snapshot,
                operational_metrics=op_metrics,
                passed_all_hard_validators=False,
                planning_error=planning_error,
            )

        output_itinerary_dict = output_itinerary.model_dump()
        validation_results = run_all_validators(output_itinerary, scenario, initial_itinerary)
        soft_metrics = compute_soft_metrics(output_itinerary, scenario)

    else:
        # Standard planning (no disruption)
        state: PlanningState = {
            "session_id": session_id,
            "persona": scenario.persona,
            "log": [],
            "revision_count": 0,
        }
        output_itinerary, op_metrics = await run_planning(
            state, injected_weather=scenario.injected_weather
        )

        if output_itinerary is None:
            planning_error = op_metrics.errors[0] if op_metrics.errors else "Planning failed"
            return EvaluationResult(
                scenario_id=scenario.id,
                planner_version=version,
                run_at=run_at,
                scenario=scenario,
                input_snapshot=input_snapshot,
                operational_metrics=op_metrics,
                passed_all_hard_validators=False,
                planning_error=planning_error,
            )

        output_itinerary_dict = output_itinerary.model_dump()
        validation_results = run_all_validators(output_itinerary, scenario)
        soft_metrics = compute_soft_metrics(output_itinerary, scenario)

    return EvaluationResult(
        scenario_id=scenario.id,
        planner_version=version,
        run_at=run_at,
        scenario=scenario,
        output_itinerary=output_itinerary_dict,
        initial_itinerary=initial_itinerary_dict,
        input_snapshot=input_snapshot,
        validation_results=validation_results,
        soft_metrics=soft_metrics,
        operational_metrics=op_metrics,
        passed_all_hard_validators=passed_all_hard(validation_results),
    )


def save_result(result: EvaluationResult) -> Path:
    EVAL_DIR.mkdir(exist_ok=True)
    runs_dir = EVAL_DIR / "runs"
    runs_dir.mkdir(exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = runs_dir / f"{result.scenario_id}_run{result.run}_{ts}.json"
    out_path.write_text(result.model_dump_json(indent=2))

    # Update index
    index_path = EVAL_DIR / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    index.setdefault(result.scenario_id, []).append(str(out_path))
    index_path.write_text(json.dumps(index, indent=2))

    return out_path


COMPLETE_MARK = "_complete"   # written when a scenario run recorded with no failed call


def recording_folder(root: Path, run: int, scenario_id: str) -> Path:
    return Path(root) / f"run_{run}" / scenario_id


async def run_all(
    version: str = PLANNER_VERSION,
    save: bool = True,
    mode: str = "live",
    repeats: int = 1,
    scenarios=None,
    recordings_root: Path | None = None,
    resume: bool = False,
) -> list[EvaluationResult]:
    """Run every scenario `repeats` times. mode is live, record or replay.

    With resume (record mode only), a scenario run whose recording is complete is replayed
    instead of recorded again; the rest are recorded from scratch.
    """
    if mode not in ("live", "record", "replay"):
        raise ValueError(mode)
    scenarios = ALL_SCENARIOS if scenarios is None else scenarios
    root = recordings_root or EVAL_DIR / "recordings" / version
    results = []
    total = len(scenarios)
    for run in range(repeats):
        for i, scenario in enumerate(scenarios, 1):
            print(f"[run {run}] [{i}/{total}] Running: {scenario.id}")
            folder = recording_folder(root, run, scenario.id)
            run_mode = mode
            if mode == "record":
                if resume and (folder / COMPLETE_MARK).exists():
                    run_mode = "replay"
                else:
                    shutil.rmtree(folder, ignore_errors=True)   # no stale answers from an older recording
            if run_mode == "live":
                ctx = contextlib.nullcontext(None)
            else:
                ctx = recording(run_mode, folder)
            with ctx as rec:
                try:
                    result = await run_scenario(scenario, version)
                except Exception as exc:
                    print(f"  ✗ {scenario.id} — RUNNER EXCEPTION: {exc}")
                    result = EvaluationResult(
                        scenario_id=scenario.id,
                        planner_version=version,
                        run_at=datetime.now(timezone.utc).isoformat(),
                        scenario=scenario,
                        input_snapshot={},
                        operational_metrics=OperationalMetrics(
                            total_runtime_seconds=0,
                            llm_call_count=0,
                            revision_count=0,
                            errors=[str(exc)],
                        ),
                        passed_all_hard_validators=False,
                        planning_error=str(exc),
                    )
            result.run = run
            if rec is not None:
                result.recorded_calls = rec.calls if run_mode == "record" else rec.replayed
                if rec.failures:
                    result.eval_error = (
                        f"{len(rec.failures)} call(s) failed, fallback used: "
                        + "; ".join(rec.failures)
                    )
                elif run_mode == "record":
                    folder.mkdir(parents=True, exist_ok=True)
                    (folder / COMPLETE_MARK).touch()
            if result.eval_error:
                status = "INVALID"
            else:
                status = "✓" if result.passed_all_hard_validators else "✗"
            error = f" — ERROR: {result.planning_error}" if result.planning_error else ""
            invalid = f" — {result.eval_error}" if result.eval_error else ""
            print(f"  {status} {scenario.id} ({result.operational_metrics.total_runtime_seconds:.1f}s){error}{invalid}")
            if save:
                path = save_result(result)
                print(f"  → {path}")
            results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description="Wayfarer evaluation runner")
    parser.add_argument("--scenario", help="Run a single scenario by ID")
    parser.add_argument("--version", default=PLANNER_VERSION, help="Planner version label")
    parser.add_argument("--mode", choices=["live", "record", "replay"], default="live",
                        help="live: call the model; record: call it and save every answer; "
                             "replay: reuse saved answers, no network")
    parser.add_argument("--repeats", type=int, default=1, help="Runs per scenario")
    parser.add_argument("--recordings", type=Path, default=None,
                        help="Recordings folder (default eval_results/recordings/<version>)")
    parser.add_argument("--resume", action="store_true",
                        help="Record mode: replay scenario runs already recorded cleanly, record the rest")
    parser.add_argument("--report", action="store_true", help="Generate baseline report after run")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to disk")
    args = parser.parse_args()

    scenarios = None
    if args.scenario:
        scenario = SCENARIO_BY_ID.get(args.scenario)
        if not scenario:
            print(f"Unknown scenario: {args.scenario}")
            print(f"Available: {', '.join(SCENARIO_BY_ID)}")
            return
        scenarios = [scenario]

    results = asyncio.run(run_all(
        args.version, save=not args.no_save, mode=args.mode, repeats=args.repeats,
        scenarios=scenarios, recordings_root=args.recordings, resume=args.resume,
    ))
    if args.scenario and args.repeats == 1:
        print(results[0].model_dump_json(indent=2))
    if args.report:
        from src.evaluation.report import generate_report, print_report
        report = generate_report(results, args.version)
        print_report(report)
        if not args.no_save:
            report_path = EVAL_DIR / f"{args.version}_report.json"
            report_path.write_text(json.dumps(report, indent=2))
            print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
