"""Cluster bootstrap over scenarios."""
import pytest

from src.evaluation.stats import cluster_bootstrap_ci


def test_identical_scenarios_give_a_zero_width_interval():
    est, lo, hi = cluster_bootstrap_ci({f"s{i}": [1.0, 1.0, 1.0] for i in range(20)})
    assert (est, lo, hi) == (1.0, 1.0, 1.0)


def test_correlated_runs_give_a_wider_interval_than_treating_runs_as_independent():
    # 30 scenarios, each always passes or always fails: 3 runs add no new information
    by_scenario = {f"s{i}": [float(i % 2)] * 3 for i in range(30)}
    as_if_independent = {f"s{i}-r{k}": [v] for i, runs in enumerate(by_scenario.values())
                         for k, v in enumerate(runs)}
    est_c, lo_c, hi_c = cluster_bootstrap_ci(by_scenario)
    est_n, lo_n, hi_n = cluster_bootstrap_ci(as_if_independent)
    assert est_c == est_n == 0.5
    assert (hi_c - lo_c) > 1.5 * (hi_n - lo_n)      # about sqrt(3) wider


def test_each_scenario_counts_once_whatever_its_number_of_valid_runs():
    est, _, _ = cluster_bootstrap_ci({"a": [1.0, 1.0, 1.0], "b": [0.0]})
    assert est == 0.5


def test_no_values_raises():
    with pytest.raises(ValueError):
        cluster_bootstrap_ci({"a": []})
