from src.evaluation.planner_compare import paired


def _rows(values, metric):
    return {sid: {metric: v} for sid, v in values.items()}


def test_lower_is_better_counts_a_drop_as_a_win():
    a = _rows({"s1": 50, "s2": 40, "s3": 30}, "travel_min")
    b = _rows({"s1": 40, "s2": 45, "s3": 30}, "travel_min")
    r = paired(a, b, "travel_min")
    assert (r["b_better"], r["b_worse"], r["ties"]) == (1, 1, 1)
    assert r["b_worse_ids"] == ["s2"]


def test_higher_is_better_for_persona_fit():
    a = _rows({"s1": 0.80, "s2": 0.80}, "persona_fit")
    b = _rows({"s1": 0.85, "s2": 0.75}, "persona_fit")
    r = paired(a, b, "persona_fit")
    assert (r["b_better"], r["b_worse"]) == (1, 1)


def test_interval_excludes_zero_only_when_every_scenario_moves_one_way():
    a = _rows({f"s{i}": 10 for i in range(20)}, "travel_min")
    b = _rows({f"s{i}": 7 for i in range(20)}, "travel_min")
    assert paired(a, b, "travel_min")["excludes_zero"]
    mixed = _rows({f"s{i}": 10 + (3 if i % 2 else -3) for i in range(20)}, "travel_min")
    assert not paired(a, mixed, "travel_min")["excludes_zero"]


def test_scenarios_missing_from_one_planner_are_left_out():
    a = _rows({"s1": 10, "s2": 10}, "travel_min")
    b = _rows({"s1": 8}, "travel_min")
    assert paired(a, b, "travel_min")["scenarios"] == 1
