# Planner design

## Why the planner changes

Today a trip is built by separate passes. The experience generator ranks stops and `_ensure_variety` trims the pool. The optimizer then takes the top `days × k` stops, clusters them by day and orders each day by route. The assembler sorts each day by time of day, and the critics and the revision step edit days afterwards. Each pass is reasonable alone. Later passes undo earlier ones, though. The time-of-day sort can break a short route, and a revision swap can break both. No single step owns the whole plan.

The new planner owns selection, day assignment, order, start times and travel in one place. Hard constraints decide whether a plan is allowed. Soft terms rank the allowed plans. The LLM does not decide feasibility, schedules or routes.

## Decision variables

For every candidate stop `e`:

| Variable | Meaning |
|---|---|
| `select[e]` | in the trip or not |
| `day[e]` | which day |
| `position[e]` | which stop of that day |
| `start[e]` | start time in minutes |

Persona fields and stop facts such as cost or duration are input data. The planner reads them and does not change them.

## Hard constraints

| # | Rule | Status in code |
|---|---|---|
| 1 | Wheelchair users only get accessible stops | enforced (`check_experience`) |
| 2 | Each day costs at most 1.2 × the daily budget, the trip at most 1.3 × | enforced per stop and by the budget critic |
| 3 | A stop starts after it opens and ends before it closes | **no data**: experiences have `best_time_of_day`, not opening hours |
| 4 | The day fits in waking hours, and arrival or departure times when known | partly: day start comes from the energy profile |
| 5 | Family trips with children get no adults-only stops | **only a score bonus** (`experience_generator.py:48`, `highlight_ranker.py:52`) |
| 6 | Dietary needs are respected | **not checked**: `persona.hard.dietary` exists, experiences have no dietary tags |
| 7 | The next stop starts after the previous one ends plus travel time, with no overlap | handled by the assembler after the fact, not by the planner |
| 8 | Each stop is used at most once, and every day has at least 2 stops | at most once: yes. Minimum per day: no |

For children, the persona questions ask "Are any children coming?" only when `social_context` is `family`. If the traveler skips it, the planner assumes children are present.

## Soft terms

The objective adds these terms over the selected stops and days.

- persona fit (the current `score_experience`, 0 to 1 per stop)
- minus total travel time
- minus a penalty for two high-energy stops in a row and for energy that does not match the traveler's energy profile
- plus variety across experience types (this replaces the `_ensure_variety` pre-cut, so the planner sees every feasible stop)
- minus a penalty for each stop below the pace target `k` (3, 4 or 5 per day)

Weights start from the current scoring and are tuned only against measured results.

## Search space

Counted on the 56 evaluation scenarios with `python -m scripts.search_space`. The count covers which stops, which day and what order (`n! / (n - D·k)!`), and leaves out start times.

| Scenarios | Feasible candidates | Slots (days × k) | Plans |
|---|---|---|---|
| Paris and Tokyo, 3 days | 15 | 12 | about 10^11 |
| London, 3 days | 20 | 12 | about 10^14 |
| Amsterdam, 3 days | 26 to 30 | 9 to 15 | 10^12 to 10^17 |
| Largest (fast pace, Amsterdam) | 30 | 15 | about 10^20 |
| 1-day trips | 26 to 30 | 3 | about 10^4 |

Checking all of them is out of reach. The candidate lists are short, though, so local search that checks the hard rules on every move can try thousands of plans in well under a second.

Two scenarios cannot fill their days at the pace target. The 5-day Tokyo trip needs 20 stops and has 15 candidates. The 7-day London trip needs 28 and has 20. Today both get fewer stops with no warning and still count as valid. With a minimum of 2 stops per day both are feasible, and the plan says that the city has fewer matching stops than the pace asks for.

## Outcomes

Each run ends in one of three states.

| State | Meaning | What the user sees |
|---|---|---|
| feasible | a plan that passes every hard rule (best found, not proven optimal) | the plan, plus any shortfall note |
| infeasible | proven that no plan can pass the hard rules | which rules conflict, and the options to change them |
| unresolved | the search stopped without a plan and without a proof | an error, not reported as infeasible |

The planner does not relax a hard rule on its own. Simple counting checks run before the search and catch the clear cases, such as fewer than `2 × days` feasible stops, or the two cheapest stops per day already over the budget ceiling.

## Algorithm

The first version, `planner_v2a`, builds a plan greedily and then improves it with local search (moving a stop, swapping two stops, reversing part of a day). It checks the hard rules after each move. It runs behind a `planner` setting so the current pipeline stays available for comparison.

A constraint solver (CP-SAT) is built only if the benchmark below shows that v2a falls short. The candidate lists are small, a proof of optimality on hand-set weights adds little, and modelling travel time in a solver is a few days of work.

## Evaluation

v2a runs on the same 56 scenarios as the current planner, paired, with replay for unchanged model calls.

A solver is not needed if v2a meets all four bars below.

| Metric | Bar |
|---|---|
| valid plans | 56 of 56 with no hard violations |
| score | within 2% of the best of 20 random restarts, and equal to the exact optimum on the 1-day trips |
| latency | under 1 second per plan |
| infeasible cases | all caught by the counting checks |

A solver version is built if v2a misses a valid plan that exists, falls more than 5% short on any scenario group, or gives different best plans across restarts.

Also reported: travel time, persona score, stops per day, model calls and revisions, each with a confidence interval, and every regression listed.

## Results

Run with `python -m src.evaluation.planner_compare` on the 56 scenarios. It needs no model calls, so the two planners see the same candidates and the difference has no model noise. Both planners are timed by `time_day()`, which counts a 10-minute wait at the stop on every transit leg.

| Metric | Baseline | v2a | v2a minus baseline (95% CI) |
|---|---|---|---|
| valid plans | 52 of 56 | 56 of 56 | |
| travel per scenario, waits included | 140.0 min | 91.1 min | −49.0 [−59.9, −38.5], better on 53, worse on 1 |
| transit legs per scenario | 4.46 | 2.75 | −1.71 [−2.14, −1.27] |
| walking per scenario | | | −0.09 km [−0.47, 0.27] |
| persona score | | | −0.010 [−0.012, −0.007] |
| slowest plan | 0.5 ms | 630 ms | |

The baseline's four invalid plans break the daily or trip budget.

In the first run of this benchmark, the wait was added only after planning. v2a saved 42 minutes on paper but used about three more transit legs than the baseline. Once the waits were added, the gain fell to 13 minutes and 25 scenarios were worse. The planner could not see the cost it was creating. Moving the wait into `time_day()` let the search pay for each transit leg, and v2a now takes fewer of them than the baseline.

Against the bars above, valid plans and latency are met. The score bar (random restarts and the exact optimum on 1-day trips) was not measured, so the solver question stays open. The persona score drop is small, but it holds on 49 of 56 scenarios.

## Open questions

- Opening hours: add `open` and `close` to the city data, or keep `best_time_of_day` as a soft window for now.
- Dietary tags on food experiences.
- What the critics and the revision step still do once the planner owns hard rules: review only, or also propose edits that go back through the planner.
