"""Planner v2a: a greedy first plan, then local search over which stops are used and on which day.

Hard rules decide whether a plan is allowed. The score ranks allowed plans. A move is kept only
if the new plan passes every hard rule and scores strictly higher, so the search always ends.
"""
import time
from dataclasses import dataclass
from typing import Iterator, Optional

from src.models.experience import Experience
from src.models.persona import TravelerPersona
from src.optimization.route import optimize_route, travel_time_minutes

MIN_STOPS_PER_DAY = 2
# Starting weights, to be tuned only against measured results.
W_TRAVEL = 0.01      # per minute of travel: 30 minutes cost as much as 0.3 of persona fit
W_VARIETY = 0.1      # per distinct experience type in the trip
W_PACE = 0.5         # per stop missing below the pace target
EPS = 1e-6           # smaller gains are rounding noise, not improvements
MAX_MOVES = 5000
MAX_SECONDS = 0.5

Plan = list[list[Experience]]


@dataclass
class SearchResult:
    status: str                  # "feasible" | "infeasible" | "unresolved"
    days: Plan
    score: float = 0.0
    moves_tried: int = 0
    moves_kept: int = 0
    stopped_by: str = ""         # "local_optimum" | "move_cap" | "time_cap"
    shortfall: int = 0           # stops below the pace target, summed over days
    reason: Optional[str] = None


class _Scorer:
    """Scores plans and checks their hard rules. Day travel is cached by the stops in the day."""

    def __init__(self, persona: TravelerPersona, per_day: int):
        self.per_day = per_day
        self.day_ceiling = persona.hard.day_budget_ceiling
        self.trip_ceiling = persona.hard.trip_budget_ceiling
        self._routes: dict[frozenset[str], tuple[list[Experience], int]] = {}

    def route(self, day: list[Experience]) -> tuple[list[Experience], int]:
        key = frozenset(e.id for e in day)
        if key not in self._routes:
            ordered = optimize_route(list(day))
            minutes = sum(travel_time_minutes(a, b) for a, b in zip(ordered, ordered[1:]))
            self._routes[key] = (ordered, minutes)
        return self._routes[key]

    def allowed(self, plan: Plan) -> bool:
        if any(not MIN_STOPS_PER_DAY <= len(day) <= self.per_day for day in plan):
            return False
        day_costs = [sum(e.cost_euros for e in day) for day in plan]
        return max(day_costs) <= self.day_ceiling and sum(day_costs) <= self.trip_ceiling

    def score(self, plan: Plan) -> float:
        fit = sum(e.persona_score or 0.0 for day in plan for e in day)
        travel = sum(self.route(day)[1] for day in plan)
        variety = len({e.experience_type for day in plan for e in day})
        shortfall = sum(max(0, self.per_day - len(day)) for day in plan)
        return fit - W_TRAVEL * travel + W_VARIETY * variety - W_PACE * shortfall


def _greedy_start(candidates: list[Experience], days: int, scorer: _Scorer) -> Plan:
    """Best stops first, each on the day where it adds the least travel and still fits.
    Every day gets its minimum before any day goes past it, so short candidate lists still cover all days."""
    plan: Plan = [[] for _ in range(days)]
    costs = [0.0] * days
    for stop in sorted(candidates, key=lambda e: e.persona_score or 0.0, reverse=True):
        if all(len(day) >= scorer.per_day for day in plan):
            break
        cap = MIN_STOPS_PER_DAY if any(len(day) < MIN_STOPS_PER_DAY for day in plan) else scorer.per_day
        best_day, best_added = None, None
        for d, day in enumerate(plan):
            if len(day) >= cap or costs[d] + stop.cost_euros > scorer.day_ceiling:
                continue
            added = scorer.route(day + [stop])[1] - scorer.route(day)[1] if day else 0
            if best_added is None or added < best_added:
                best_day, best_added = d, added
        if best_day is not None and sum(costs) + stop.cost_euros <= scorer.trip_ceiling:
            plan[best_day].append(stop)
            costs[best_day] += stop.cost_euros
    return plan


def _neighbours(plan: Plan, unused: list[Experience]) -> Iterator[Plan]:
    """Every plan one move away: replace, relocate, then swap across days."""
    for d, day in enumerate(plan):                       # replace a used stop with an unused one
        for i in range(len(day)):
            for new in unused:
                yield [day[:i] + [new] + day[i + 1:] if k == d else other for k, other in enumerate(plan)]
    for a, day in enumerate(plan):                       # relocate one stop to another day
        for i, stop in enumerate(day):
            for b in range(len(plan)):
                if b != a:
                    yield [day[:i] + day[i + 1:] if k == a else other + [stop] if k == b else other
                           for k, other in enumerate(plan)]
    for a in range(len(plan)):                           # swap two stops on different days
        for b in range(a + 1, len(plan)):
            for i, s in enumerate(plan[a]):
                for j, t in enumerate(plan[b]):
                    yield [plan[a][:i] + [t] + plan[a][i + 1:] if k == a
                           else plan[b][:j] + [s] + plan[b][j + 1:] if k == b else other
                           for k, other in enumerate(plan)]


def plan_v2a(candidates: list[Experience], persona: TravelerPersona, per_day: int) -> SearchResult:
    """Choose and place stops for every day. Candidates must already pass the per-stop hard rules."""
    days = persona.trip_days
    if len(candidates) < MIN_STOPS_PER_DAY * days:
        return SearchResult("infeasible", [], reason=(
            f"{len(candidates)} matching stops for {days} days; at least {MIN_STOPS_PER_DAY * days} are needed"))

    scorer = _Scorer(persona, per_day)
    plan = _greedy_start(candidates, days, scorer)
    if not scorer.allowed(plan):
        return SearchResult("unresolved", plan, reason="the greedy start breaks a hard rule")

    best = scorer.score(plan)
    tried = kept = 0
    stopped_by = "local_optimum"
    started = time.perf_counter()
    improved = True
    while improved:
        improved = False
        used = {e.id for day in plan for e in day}
        unused = [e for e in candidates if e.id not in used]
        for candidate in _neighbours(plan, unused):
            tried += 1
            if tried > MAX_MOVES:
                stopped_by = "move_cap"
                break
            if time.perf_counter() - started > MAX_SECONDS:
                stopped_by = "time_cap"
                break
            if scorer.allowed(candidate) and (s := scorer.score(candidate)) > best + EPS:
                plan, best, kept, improved = candidate, s, kept + 1, True
                break
        if stopped_by != "local_optimum":
            break

    return SearchResult(
        "feasible", [scorer.route(day)[0] for day in plan], score=best, moves_tried=min(tried, MAX_MOVES),
        moves_kept=kept, stopped_by=stopped_by,
        shortfall=sum(max(0, per_day - len(day)) for day in plan),
    )
