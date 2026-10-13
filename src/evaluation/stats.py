"""Confidence intervals for results with repeated runs per scenario.

Runs of the same scenario share the persona and constraints, so they are not independent.
The bootstrap therefore resamples scenarios (the independent unit) and keeps all runs of each
scenario it draws. Resampling single runs would treat 3 x 56 results as 168 independent ones
and give an interval that is too narrow.
"""
from __future__ import annotations

import math
import random
from statistics import mean


def _percentile(sorted_values: list[float], q: float) -> float:
    """Linear interpolation between the two nearest ranks (q in [0, 1])."""
    pos = q * (len(sorted_values) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def cluster_bootstrap_ci(values_by_scenario: dict[str, list[float]], n_boot: int = 2000,
                         seed: int = 0, level: float = 0.95) -> tuple[float, float, float]:
    """Mean over scenarios and its percentile-bootstrap interval: (estimate, low, high).

    Each scenario counts once (the mean of its runs), so a scenario with an extra valid run
    does not weigh more than one that lost a run to an invalid call.
    """
    per_scenario = [mean(v) for v in values_by_scenario.values() if v]
    n = len(per_scenario)
    if n == 0:
        raise ValueError("no scenarios with values")
    rng = random.Random(seed)
    stats = sorted(sum(per_scenario[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(n_boot))
    tail = (1 - level) / 2
    return mean(per_scenario), _percentile(stats, tail), _percentile(stats, 1 - tail)
