"""Planner v2a: a greedy first plan, then local search over which stops are used, on which day and in
what order. Each day is timed with the same function the assembler uses.

Hard rules decide whether a plan is allowed. The score ranks allowed plans. A move is kept only
if the new plan passes every hard rule and scores strictly higher, so the search always ends.
"""
import time
from dataclasses import dataclass
from typing import Iterator, Optional

from src.models.experience import Experience
from src.models.persona import TravelerPersona
from src.optimization.energy import build_energy_curve, time_to_minutes
from src.optimization.route import optimize_route
from src.planning.timing import time_day

MIN_STOPS_PER_DAY = 2
DAY_END_LIMIT = 23 * 60  # hard rule: the last stop ends by 23:00
HIGH_ENERGY = 7.0
# Starting weights, to be tuned only against measured results.
W_TRAVEL = 0.01      # per minute of travel: 30 minutes cost as much as 0.3 of persona fit
W_LATE = 0.005       # per minute a stop starts after its best time-of-day window
W_ENERGY = 0.05      # per point a stop needs more energy than the traveler has at that hour
W_TIRING_PAIR = 0.2  # per two high-energy stops in a row
W_VARIETY = 0.1      # per distinct experience type in the trip
W_PACE = 0.5         # per missing stop squared, per day: one day 2 short costs more than two days 1 short
EPS = 1e-6           # smaller gains are rounding noise, not improvements
MAX_MOVES = 5000
MAX_SECONDS = 2.0      # a safety net only; the benchmark holds plans to 1 s

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


@dataclass
class _Day:
    ordered: list[Experience]
    penalty: float           # travel, lateness and energy costs, in score units
    end: int                 # minute the last stop ends


class _Scorer:
    """Scores plans and checks their hard rules. Each day's best order is cached by the stops in it."""

    def __init__(self, persona: TravelerPersona, per_day: int):
        self.persona = persona
        self.per_day = per_day
        self.day_ceiling = persona.hard.day_budget_ceiling
        self.trip_ceiling = persona.hard.trip_budget_ceiling
        self.curve = sorted((time_to_minutes(t), e) for t, e in build_energy_curve(persona.energy_profile).items())
        self._days: dict[frozenset[str], _Day] = {}

    def _energy_at(self, minute: int) -> float:
        """The traveler's energy at this minute, read off their energy curve (linear between points)."""
        points = self.curve
        if minute <= points[0][0]:
            return points[0][1]
        for (t1, e1), (t2, e2) in zip(points, points[1:]):
            if minute <= t2:
                return e1 + (minute - t1) / (t2 - t1) * (e2 - e1)
        return points[-1][1]

    def _timed(self, ordered: list[Experience]) -> _Day:
        slots = time_day(ordered, self.persona)
        penalty = 0.0
        for k, slot in enumerate(slots):
            energy = self._energy_at(slot.start)
            penalty += W_TRAVEL * slot.travel + W_LATE * slot.late
            penalty += W_ENERGY * max(0.0, slot.experience.energy_required - energy)
            if k and min(slot.experience.energy_required, slots[k - 1].experience.energy_required) >= HIGH_ENERGY:
                penalty += W_TIRING_PAIR
        return _Day(ordered, penalty, slots[-1].end if slots else 0)

    def day(self, stops: list[Experience]) -> _Day:
        """The best order found for these stops: 2-opt on the timed day, from two starting orders."""
        key = frozenset(e.id for e in stops)
        if key not in self._days:
            starts = [optimize_route(list(stops)),
                      sorted(stops, key=lambda e: _WINDOW_ORDER.get(e.best_time_of_day, 2))]
            self._days[key] = min((self._two_opt(o) for o in starts), key=lambda d: d.penalty)
        return self._days[key]

    def _two_opt(self, ordered: list[Experience]) -> _Day:
        best = self._timed(ordered)
        improved = True
        while improved:
            improved = False
            for i in range(len(best.ordered) - 1):
                for j in range(i + 1, len(best.ordered)):
                    o = best.ordered
                    trial = self._timed(o[:i] + o[i:j + 1][::-1] + o[j + 1:])
                    if trial.penalty < best.penalty - EPS:
                        best, improved = trial, True
        return best

    def allowed(self, plan: Plan) -> bool:
        if any(not MIN_STOPS_PER_DAY <= len(day) <= self.per_day for day in plan):
            return False
        day_costs = [sum(e.cost_euros for e in day) for day in plan]
        if max(day_costs) > self.day_ceiling or sum(day_costs) > self.trip_ceiling:
            return False
        return all(self.day(day).end <= DAY_END_LIMIT for day in plan)

    def score(self, plan: Plan) -> float:
        fit = sum(e.persona_score or 0.0 for day in plan for e in day)
        penalties = sum(self.day(day).penalty for day in plan)
        variety = len({e.experience_type for day in plan for e in day})
        shortfall = sum(max(0, self.per_day - len(day)) ** 2 for day in plan)
        return fit - penalties + W_VARIETY * variety - W_PACE * shortfall


_WINDOW_ORDER = {"early_morning": 0, "morning": 1, "afternoon": 2, "golden_hour": 3, "evening": 4, "night": 5}


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
            if scorer.day(day + [stop]).end > DAY_END_LIMIT:
                continue
            added = scorer.day(day + [stop]).penalty - scorer.day(day).penalty if day else 0
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
        "feasible", [scorer.day(day).ordered for day in plan], score=best, moves_tried=min(tried, MAX_MOVES),
        moves_kept=kept, stopped_by=stopped_by,
        shortfall=sum(max(0, per_day - len(day)) for day in plan),
    )
