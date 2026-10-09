"""
Deterministic hard validators for Wayfarer itineraries.

Each validator is a pure function:
    (itinerary: Itinerary, scenario: EvaluationScenario) -> list[ValidationResult]

Validators never modify the itinerary. They produce ValidationResult objects
describing exactly what was observed vs. what was expected.

Severity:
  error   — hard constraint violated; the itinerary is unusable regardless of quality
  warning — soft concern or design issue; the itinerary may still be acceptable
"""

from src.evaluation.models import EvaluationScenario, ValidationResult
from src.models.itinerary import Itinerary
from src.optimization.energy import time_to_minutes
from src.optimization.route import haversine, travel_time_minutes


def _ok(code: str, message: str, severity: str = "error") -> ValidationResult:
    return ValidationResult(code=code, passed=True, severity=severity, message=message)


def _fail(
    code: str,
    message: str,
    severity: str = "error",
    day: int | None = None,
    affected_ids: list[str] | None = None,
    observed: object = None,
    expected: object = None,
) -> ValidationResult:
    return ValidationResult(
        code=code,
        passed=False,
        severity=severity,
        day=day,
        affected_experience_ids=affected_ids or [],
        observed=observed,
        expected=expected,
        message=message,
    )


# ── Hard validators ────────────────────────────────────────────────────────────

def validate_day_count(itinerary: Itinerary, scenario: EvaluationScenario) -> list[ValidationResult]:
    expected = scenario.persona.trip_days
    observed = len(itinerary.days)
    if observed == expected:
        return [_ok("DAY_COUNT_MATCHES", f"Itinerary has {observed} days as expected")]
    return [_fail(
        "DAY_COUNT_MATCHES",
        f"Expected {expected} days, got {observed}",
        observed=observed, expected=expected,
    )]


def validate_no_duplicate_experiences(
    itinerary: Itinerary, scenario: EvaluationScenario
) -> list[ValidationResult]:
    seen: dict[str, int] = {}  # id → day_number of first occurrence
    results = []
    for day in itinerary.days:
        for se in day.experiences:
            exp_id = se.experience.id
            if exp_id in seen:
                results.append(_fail(
                    "NO_DUPLICATE_EXPERIENCES",
                    f"Experience '{se.experience.name}' (id={exp_id}) appears on day {seen[exp_id]} and day {day.day_number}",
                    day=day.day_number,
                    affected_ids=[exp_id],
                    observed=f"appears on days {seen[exp_id]} and {day.day_number}",
                    expected="each experience appears at most once",
                ))
            else:
                seen[exp_id] = day.day_number
    if not results:
        return [_ok("NO_DUPLICATE_EXPERIENCES", "No duplicate experiences found")]
    return results


def validate_no_schedule_overlap(
    itinerary: Itinerary, scenario: EvaluationScenario
) -> list[ValidationResult]:
    results = []
    for day in itinerary.days:
        exps = day.experiences
        for i in range(len(exps)):
            for j in range(i + 1, len(exps)):
                a, b = exps[i], exps[j]
                a_start = time_to_minutes(a.start_time)
                a_end = time_to_minutes(a.end_time)
                b_start = time_to_minutes(b.start_time)
                b_end = time_to_minutes(b.end_time)
                if a_start < b_end and a_end > b_start:
                    results.append(_fail(
                        "NO_SCHEDULE_OVERLAP",
                        f"Day {day.day_number}: '{a.experience.name}' ({a.start_time}–{a.end_time}) "
                        f"overlaps with '{b.experience.name}' ({b.start_time}–{b.end_time})",
                        day=day.day_number,
                        affected_ids=[a.experience.id, b.experience.id],
                        observed=f"{a.start_time}–{a.end_time} overlaps {b.start_time}–{b.end_time}",
                        expected="no overlapping time slots",
                    ))
    if not results:
        return [_ok("NO_SCHEDULE_OVERLAP", "No schedule overlaps found")]
    return results


def validate_no_day_overflow(
    itinerary: Itinerary, scenario: EvaluationScenario
) -> list[ValidationResult]:
    results = []
    MIDNIGHT = 24 * 60
    for day in itinerary.days:
        if not day.experiences:
            continue
        last = day.experiences[-1]
        end_mins = time_to_minutes(last.end_time)
        if end_mins > MIDNIGHT:
            results.append(_fail(
                "NO_DAY_OVERFLOW",
                f"Day {day.day_number}: last activity ends at {last.end_time} (past midnight)",
                day=day.day_number,
                affected_ids=[last.experience.id],
                observed=last.end_time,
                expected="end time ≤ 24:00",
            ))
    if not results:
        return [_ok("NO_DAY_OVERFLOW", "No day overflows past midnight")]
    return results


def validate_daily_budget(
    itinerary: Itinerary, scenario: EvaluationScenario
) -> list[ValidationResult]:
    ceiling = scenario.persona.budget_per_day * 1.2  # mirrors budget_critic threshold
    results = []
    for day in itinerary.days:
        if day.total_cost > ceiling:
            results.append(_fail(
                "DAILY_BUDGET_RESPECTED",
                f"Day {day.day_number}: cost €{day.total_cost:.0f} exceeds daily ceiling €{ceiling:.0f}",
                day=day.day_number,
                observed=day.total_cost,
                expected=f"≤ €{ceiling:.0f}",
            ))
    if not results:
        return [_ok("DAILY_BUDGET_RESPECTED", "All days within daily budget ceiling")]
    return results


def validate_trip_budget(
    itinerary: Itinerary, scenario: EvaluationScenario
) -> list[ValidationResult]:
    ceiling = scenario.persona.budget_per_day * scenario.persona.trip_days * 1.3
    if itinerary.total_cost > ceiling:
        return [_fail(
            "TRIP_BUDGET_RESPECTED",
            f"Total trip cost €{itinerary.total_cost:.0f} exceeds trip ceiling €{ceiling:.0f}",
            observed=itinerary.total_cost,
            expected=f"≤ €{ceiling:.0f}",
        )]
    return [_ok("TRIP_BUDGET_RESPECTED", f"Total cost €{itinerary.total_cost:.0f} within trip budget")]


def validate_accessibility(
    itinerary: Itinerary, scenario: EvaluationScenario
) -> list[ValidationResult]:
    constraints = scenario.persona.physical_constraints
    results = []

    if "wheelchair" in constraints:
        for day in itinerary.days:
            for se in day.experiences:
                if se.experience.physical_intensity == "high":
                    results.append(_fail(
                        "ACCESSIBILITY_RESPECTED",
                        f"Day {day.day_number}: '{se.experience.name}' has high physical intensity "
                        f"but persona uses wheelchair",
                        day=day.day_number,
                        affected_ids=[se.experience.id],
                        observed="physical_intensity=high",
                        expected="physical_intensity ∈ {low, medium}",
                    ))

    if not results:
        return [_ok("ACCESSIBILITY_RESPECTED", "Accessibility constraints respected")]
    return results


def validate_no_impossible_values(
    itinerary: Itinerary, scenario: EvaluationScenario
) -> list[ValidationResult]:
    results = []
    for day in itinerary.days:
        if day.total_cost < 0:
            results.append(_fail(
                "NO_IMPOSSIBLE_VALUES",
                f"Day {day.day_number}: negative total_cost ({day.total_cost})",
                day=day.day_number,
                observed=day.total_cost,
                expected="≥ 0",
            ))
        for se in day.experiences:
            if se.experience.duration_minutes <= 0:
                results.append(_fail(
                    "NO_IMPOSSIBLE_VALUES",
                    f"Day {day.day_number}: '{se.experience.name}' has duration_minutes={se.experience.duration_minutes}",
                    day=day.day_number,
                    affected_ids=[se.experience.id],
                    observed=se.experience.duration_minutes,
                    expected="> 0",
                ))
            if se.travel_time_from_prev < 0:
                results.append(_fail(
                    "NO_IMPOSSIBLE_VALUES",
                    f"Day {day.day_number}: '{se.experience.name}' has negative travel_time_from_prev={se.travel_time_from_prev}",
                    day=day.day_number,
                    affected_ids=[se.experience.id],
                    observed=se.travel_time_from_prev,
                    expected="≥ 0",
                ))
            if se.experience.cost_euros < 0:
                results.append(_fail(
                    "NO_IMPOSSIBLE_VALUES",
                    f"Day {day.day_number}: '{se.experience.name}' has negative cost_euros={se.experience.cost_euros}",
                    day=day.day_number,
                    affected_ids=[se.experience.id],
                    observed=se.experience.cost_euros,
                    expected="≥ 0",
                ))
    if not results:
        return [_ok("NO_IMPOSSIBLE_VALUES", "No impossible durations, travel times, or costs")]
    return results


def validate_travel_time_feasible(
    itinerary: Itinerary, scenario: EvaluationScenario
) -> list[ValidationResult]:
    """
    Check that stated travel_time_from_prev is plausible given haversine distance.
    Uses ±60% tolerance — real routes are not straight lines.
    This is a warning, not an error.
    """
    results = []
    TOLERANCE = 0.60

    for day in itinerary.days:
        exps = day.experiences
        for i in range(1, len(exps)):
            prev = exps[i - 1].experience
            curr = exps[i]
            if curr.travel_time_from_prev == 0:
                continue
            expected_mins = travel_time_minutes(prev, curr.experience, curr.travel_mode)
            stated = curr.travel_time_from_prev
            lo = expected_mins * (1 - TOLERANCE)
            hi = expected_mins * (1 + TOLERANCE)
            if not (lo <= stated <= hi):
                results.append(_fail(
                    "TRAVEL_TIME_FEASIBLE",
                    f"Day {day.day_number}: travel to '{curr.experience.name}' stated {stated} min "
                    f"but haversine estimate is {expected_mins} min ({curr.travel_mode})",
                    severity="warning",
                    day=day.day_number,
                    affected_ids=[curr.experience.id],
                    observed=stated,
                    expected=f"{expected_mins} min ± 60%",
                ))

    if not results:
        return [_ok("TRAVEL_TIME_FEASIBLE", "All stated travel times are plausible", severity="warning")]
    return results


def validate_must_do(
    itinerary: Itinerary, scenario: EvaluationScenario
) -> list[ValidationResult]:
    if not scenario.must_do_ids:
        return []
    scheduled_ids = {
        se.experience.id
        for day in itinerary.days
        for se in day.experiences
    }
    results = []
    for req_id in scenario.must_do_ids:
        if req_id not in scheduled_ids:
            results.append(_fail(
                "MUST_DO_SATISFIED",
                f"Required experience '{req_id}' was not scheduled",
                observed="not in itinerary",
                expected=f"experience {req_id} scheduled",
            ))
    if not results:
        return [_ok("MUST_DO_SATISFIED", "All must-do experiences are scheduled")]
    return results


def validate_must_avoid(
    itinerary: Itinerary, scenario: EvaluationScenario
) -> list[ValidationResult]:
    if not scenario.must_avoid_ids:
        return []
    results = []
    for day in itinerary.days:
        for se in day.experiences:
            if se.experience.id in scenario.must_avoid_ids:
                results.append(_fail(
                    "MUST_AVOID_RESPECTED",
                    f"Forbidden experience '{se.experience.name}' (id={se.experience.id}) was scheduled on day {day.day_number}",
                    day=day.day_number,
                    affected_ids=[se.experience.id],
                    observed="scheduled",
                    expected="not in itinerary",
                ))
    if not results:
        return [_ok("MUST_AVOID_RESPECTED", "No forbidden experiences scheduled")]
    return results


def validate_completed_days_preserved(
    itinerary_after: Itinerary,
    scenario: EvaluationScenario,
    itinerary_before: Itinerary,
) -> list[ValidationResult]:
    """Only run for replanning scenarios. Checks that completed days are unchanged."""
    if not scenario.disruption:
        return []

    results = []
    completed_day_numbers = scenario.disruption.expected_stable_day_numbers
    for day_num in completed_day_numbers:
        before_day = next((d for d in itinerary_before.days if d.day_number == day_num), None)
        after_day = next((d for d in itinerary_after.days if d.day_number == day_num), None)

        if before_day is None or after_day is None:
            results.append(_fail(
                "COMPLETED_DAYS_PRESERVED",
                f"Day {day_num} missing from one of the itineraries",
                day=day_num,
            ))
            continue

        before_ids = [se.experience.id for se in before_day.experiences]
        after_ids = [se.experience.id for se in after_day.experiences]

        if before_ids != after_ids:
            results.append(_fail(
                "COMPLETED_DAYS_PRESERVED",
                f"Day {day_num} experiences changed after replanning (was: {before_ids}, now: {after_ids})",
                day=day_num,
                observed=after_ids,
                expected=before_ids,
            ))

    if not results:
        return [_ok("COMPLETED_DAYS_PRESERVED", "All completed days preserved after replanning")]
    return results


# ── Registry ───────────────────────────────────────────────────────────────────

STANDARD_VALIDATORS = [
    validate_day_count,
    validate_no_duplicate_experiences,
    validate_no_schedule_overlap,
    validate_no_day_overflow,
    validate_daily_budget,
    validate_trip_budget,
    validate_accessibility,
    validate_no_impossible_values,
    validate_travel_time_feasible,
    validate_must_do,
    validate_must_avoid,
]


def run_all_validators(
    itinerary: Itinerary,
    scenario: EvaluationScenario,
    itinerary_before: Itinerary | None = None,
) -> list[ValidationResult]:
    results: list[ValidationResult] = []
    for fn in STANDARD_VALIDATORS:
        results.extend(fn(itinerary, scenario))
    if scenario.disruption and itinerary_before:
        results.extend(validate_completed_days_preserved(itinerary, scenario, itinerary_before))
    return results


def passed_all_hard(results: list[ValidationResult]) -> bool:
    return all(r.passed for r in results if r.severity == "error")
