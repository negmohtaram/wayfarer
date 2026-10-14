"""The report leaves invalid runs out, gives intervals over scenarios and lists flipped verdicts."""
from src.evaluation.models import EvaluationResult, OperationalMetrics
from src.evaluation.report import generate_report
from src.evaluation.scenarios.baseline_scenarios import ALL_SCENARIOS


def _result(scenario, run, passed, invalid=False):
    return EvaluationResult(
        scenario_id=scenario.id, planner_version="t", run=run, run_at="x", scenario=scenario,
        output_itinerary={"days": []},
        operational_metrics=OperationalMetrics(total_runtime_seconds=1, llm_call_count=1,
                                               revision_count=0),
        passed_all_hard_validators=passed,
        planning_error=None if passed else "failed",
        eval_error="1 call(s) failed, fallback used: experience_critic: RuntimeError" if invalid else None,
    )


def test_report_counts_only_valid_runs_and_lists_flips():
    s = ALL_SCENARIOS[:4]
    results = (
        [_result(s[0], k, True) for k in range(3)]                       # 3/3
        + [_result(s[1], k, False) for k in range(3)]                    # 0/3
        + [_result(s[2], 0, True), _result(s[2], 1, False), _result(s[2], 2, True)]   # flip
        + [_result(s[3], 0, True), _result(s[3], 1, False, invalid=True)]             # 1 valid run
    )
    rep = generate_report(results, "t")["repeats"]

    assert (rep["total_runs"], rep["valid_runs"], rep["invalid_runs"]) == (11, 10, 1)
    assert rep["publishable"] is True                    # 1/11 = 9% invalid
    assert rep["flipped_scenarios"] == [s[2].id]
    assert rep["per_scenario"][s[2].id] == {"passed": 2, "valid_runs": 3}
    rate = rep["valid_itinerary_rate"]
    assert rate["scenarios"] == 4
    assert rate["estimate"] == round((1 + 0 + 2 / 3 + 1) / 4 * 100, 2)
    assert rate["too_wide"] is True                     # 4 scenarios can't give +-5 points


def test_too_many_invalid_runs_is_not_publishable():
    s = ALL_SCENARIOS[0]
    results = [_result(s, 0, True), _result(s, 1, True, invalid=True)]
    assert generate_report(results, "t")["repeats"]["publishable"] is False
