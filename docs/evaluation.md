# Wayfarer Evaluation Framework

## Why this exists

I want to know how good the current planner is, and where it fails, before I change its architecture.
The first output is a `baseline_v1` report. Later versions of the planner are compared against it.

---

## Running the framework

```bash
# Run all ~50 scenarios and generate the baseline report
python -m src.evaluation.runner --report

# Run a single scenario
python -m src.evaluation.runner --scenario pace_slow_solo_amsterdam

# Replay the latest saved run for a scenario
python -m src.evaluation.replay --scenario pace_slow_solo_amsterdam

# Replay from a specific result file
python -m src.evaluation.replay --scenario pace_slow_solo_amsterdam \
    --from-file eval_results/runs/pace_slow_solo_amsterdam_20260904.json
```

Results are saved to `eval_results/` (gitignored).

---

## Hard validators

Each validator is a pure function and produces `ValidationResult` objects.

| Code | Severity | What it checks |
|---|---|---|
| `DAY_COUNT_MATCHES` | error | `len(itinerary.days) == persona.trip_days` |
| `NO_DUPLICATE_EXPERIENCES` | error | Same experience ID not scheduled twice |
| `NO_SCHEDULE_OVERLAP` | error | No two experiences share overlapping time on the same day |
| `NO_DAY_OVERFLOW` | error | No day's last experience ends after 24:00 |
| `DAILY_BUDGET_RESPECTED` | error | `day.total_cost ≤ budget_per_day × 1.2` (mirrors budget_critic threshold) |
| `TRIP_BUDGET_RESPECTED` | error | `total_cost ≤ budget_per_day × trip_days × 1.3` |
| `ACCESSIBILITY_RESPECTED` | error | Wheelchair constraint → no high physical_intensity experiences |
| `NO_IMPOSSIBLE_VALUES` | error | All durations > 0, travel times ≥ 0, costs ≥ 0 |
| `TRAVEL_TIME_FEASIBLE` | warning | Stated travel time is within ±60% of haversine estimate |
| `MUST_DO_SATISFIED` | error | All must_do_ids appear in scheduled experiences |
| `MUST_AVOID_RESPECTED` | error | No must_avoid_id appears in the itinerary |
| `COMPLETED_DAYS_PRESERVED` | error | Replanning scenarios: completed days unchanged |

### Severity distinction

**error**: the itinerary is unacceptable regardless of quality (hard constraint violated).
**warning**: a concern worth tracking but not a blocking failure.

`passed_all_hard_validators` in `EvaluationResult` is `True` only when all `error`-severity validators pass.

### Expected failures

`MUST_DO_SATISFIED` and `MUST_AVOID_RESPECTED` are always expected to fail on `baseline_v1`
because the planner has no enforcement mechanism for these constraints.
Scenarios that include these are tagged with `expected_violation_codes`, and the report lists them separately
from unexpected failures under **known_unsupported_features**.

---

## Soft metrics

Soft metrics are observations. None of them claims to measure quality on its own.

| Metric | What it measures | Caveat |
|---|---|---|
| `avg_persona_score` | Mean `experience.persona_score` across all scheduled experiences | Measures self-consistency with the planner's own formula, not external quality |
| `total_travel_time_minutes` | Sum of all `travel_time_from_prev` | Factual sum; does not distinguish necessary from avoidable travel |
| `total_walking_km` | Sum of `day.total_walking_km` | Walk-mode only; transit km not tracked |
| `energy_mismatch_count` | Experiences where `energy_required − traveler_energy > 3.0` | Uses same threshold as energy_critic |
| `experience_type_diversity` | Count of distinct `experience_type` values (max 6) | Higher = more variety; does not claim optimal variety |
| `localness_delta` | `|mean(local_score) − persona.local_vs_tourist|` | 0 = perfect alignment; larger = more drift from preference |
| `budget_utilization_pct` | `total_cost / (budget_per_day × trip_days) × 100` | 100% = exactly at budget; >130% = trip budget validator fails |
| `avg_experiences_per_day` | Total scheduled experiences / trip_days | Useful for comparing slow vs fast personas |

### What is NOT measured

The following were considered and rejected:

- **Narrative coherence.** It needs an independent LLM judge, and using the same model that wrote the itinerary would be circular.
- **Emotional arc quality.** Same problem.
- **Satisfaction score.** It needs ground truth from real travelers.

A metric I can't define clearly is left out.

---

## Operational metrics

| Metric | How it is collected |
|---|---|
| `total_runtime_seconds` | `time.perf_counter()` around the full graph run |
| `node_runtimes` | Time between successive stream chunks from `planning_graph.astream()` |
| `llm_call_count` | Estimated: 1 per `critic_panel` pass = `1 + revision_count`. Replanning adds 1 for `disruption_analyzer`. |
| `revision_count` | From `itinerary.revision_count` |
| `errors` | Graph-level errors from node state `.error` field |

### Known gap: LLM retry count

`src/agents/llm.py:call_with_retry` does not expose a counter. LLM retries due to transient
failures are not tracked. This means `llm_call_count` is an undercount when retries occur.
If retry frequency is important, add a module-level counter to `call_with_retry`.

### Known gap: token usage

The Groq client does not surface token counts in the LangChain streaming interface.
`estimated_cost_usd` is always `None` in `baseline_v1`.

---

## Storage format

```
eval_results/
├── index.json                          # {scenario_id: [list of result file paths]}
├── baseline_v1_report.json             # aggregate report
└── runs/
    └── {scenario_id}_{timestamp}.json  # one EvaluationResult per run
```

`eval_results/` is gitignored, so results stay local.

---

## Scenario coverage

~56 scenarios across:

| Group | Count | What it covers |
|---|---|---|
| Pace | 6 | slow (9) + fast (2) across 3 cities |
| Budget | 6 | low (€40-50) + high (€300-400) across 3 cities |
| Social context | 4 | solo, couple, family, friends |
| Local vs tourist | 4 | local_vs_tourist extremes (1 and 9) |
| Preference | 5 | food, culture, nature, nightlife, photography |
| Physical constraints | 4 | wheelchair, bad_knee, elderly, combined |
| Energy profile | 3 | morning_person, evening_person, consistent |
| Trip length | 4 | 1, 2, 5, 7 days |
| Budget stress | 3 | ultra-low, budget vs slow pace, typical reference |
| Injected weather | 4 | all rainy, mixed, unavailable, all sunny |
| Disruption | 5 | weather, closed, tired, love_neighborhood, delayed start |
| Must-do/avoid | 2 | expected failures that document planner gaps |
| Edge cases | 6 | combined constraints, extreme personas, reference scenarios |

---

## Replay reproducibility

Replay re-runs the planning graph with the same persona and injected weather
(stored in `input_snapshot`). The LLM experience critic is still live, so soft
metrics and critic scores may differ slightly between runs. That variance is
expected at `temperature=0.7`, and it shows how much the LLM moves the numbers.

To compare two runs of the same scenario, look at hard validator results
(deterministic) separately from soft metrics (may vary).

---

## Baseline report fields

| Field | Meaning |
|---|---|
| `valid_itinerary_rate_pct` | % of scenarios where all hard validators pass (excluding expected failures) |
| `unexpected_violation_counts` | Per-code count of hard failures that are NOT expected |
| `known_unsupported_features` | Per-code count of expected failures (must_do/must_avoid) |
| `top_unexpected_failures` | Top 5 unexpected failure codes with rate |
| `soft_metrics` | mean, median, p95, min, max for each soft metric |
| `operational.runtime_seconds` | p50/p95 planning latency |
| `operational.revision_count_distribution` | How often 0/1/2 revision cycles occur |
| `planning_errors` | Scenarios where the planner itself raised an exception |

---

## Adding scenarios

Add to `src/evaluation/scenarios/baseline_scenarios.py`:

```python
MY_SCENARIO = EvaluationScenario(
    id="my_scenario",          # unique, lowercase, underscores
    description="one sentence",
    tags=["tag1", "tag2"],
    persona=_p(destination="amsterdam", pace=7, budget_per_day=100),
    # Optional:
    must_do_ids=["ams_001"],
    expected_violation_codes=["MUST_DO_SATISFIED"],  # if failure is expected
    injected_weather=[...],    # list of dicts or None per day
    disruption=DisruptionSpec(...),
)

ALL_SCENARIOS.append(MY_SCENARIO)
SCENARIO_BY_ID[MY_SCENARIO.id] = MY_SCENARIO
```

---

## Comparing versions

With 56 scenarios and a pass rate near 93%, the 95% interval on one pass rate is about plus or minus 7 points. Two separate pass rates therefore can't show a 5-point improvement. Changes are compared on the same scenarios instead. We count the scenarios a change fixes and the ones it breaks, and bootstrap the difference over scenarios. A new planner version makes new model calls, so it runs with repeats, which keeps model noise from looking like an improvement. Reaching plus or minus 5 points on a single rate would need roughly twice as many scenarios.

### v0.3_domain: hard constraints before scoring

Wheelchair access and single-stop cost are checked before any stop is scored, and a day over the budget ceiling can no longer be approved by the budget critic. The 52 scenarios this change does not touch replay exactly. The 4 budget scenarios plan differently, so their model calls were recorded again into `eval_results/recordings/v0.3_domain/`, once each.

The result was 4 fixed scenarios and none broken. That alone is not significant (sign test p about 0.13). The case rests on each fix having a known cause and a test that failed before the change. The budget is met by removing the most expensive stop, so those days are shorter. Swapping in a cheaper stop of the same kind is planned. The report is in `docs/v0.3_domain_report.json`.

## Trip chat router

The router reads one traveler message about a finished plan. A single model call with a forced tool returns the intent and the day the message refers to. The intents are `change_setting`, `feedback`, `explain`, `what_if`, `ask`, `out_of_scope` and `none_of_these`. An invalid reply gets one retry, then the router falls back to `none_of_these`.

It is tested on two hand-labelled sets. Amsterdam has 200 messages for a 3-day trip. Lisbon has 105 messages for a 4-day trip, written separately and held out. The v1 prompt adds similar labelled examples picked by embedding similarity from the Amsterdam set. That condition is only measured on Lisbon, so no test message appears in its own examples.

| set | prompt | intent correct (95% CI) | intent and day correct |
|---|---|---|---|
| Amsterdam | v0 | 84.0% (78.5 to 89.0) | 81.5% |
| Lisbon | v0 | 77.1% (68.6 to 84.8) | 69.5% |
| Lisbon | v1 with examples | 97.1% (93.3 to 100) | 96.2% |

On v0, 11 of the 13 messages with the right intent but the wrong day were about "the last day" or "our final day". The v1 prompt returns them as a reference to the last day, and code turns that into a number for the trip at hand. All five prompt injection messages in the Lisbon set were routed as `out_of_scope`.

```bash
python -m src.evaluation.chat_eval
python -m src.evaluation.chat_eval --set lisbon
python -m src.evaluation.chat_eval --set lisbon --prompt v1 --retrieval
```
