# WAYFARER

[![CI](https://github.com/negmohtaram/wayfarer/actions/workflows/ci.yml/badge.svg)](https://github.com/negmohtaram/wayfarer/actions/workflows/ci.yml)

A multi-agent travel planner that cares about *who's traveling*, not just where.

Most travel planners optimize for logistics. WAYFARER builds itineraries around the traveler's personality, energy, and what they'd regret missing. Every plan then goes through a panel of critics before the traveler sees it.

## How it works

The system is a LangGraph pipeline with the following stages:

1. **Persona Builder** - an 8-turn conversation that profiles the traveler (pace, interests, budget, local-vs-tourist preference)
2. **Experience Generator** - scores 30+ city experiences against the persona
3. **Route & Energy Optimizer** - sequences the day using TSP nearest-neighbor, then reorders by energy curve (high-energy activities when the traveler is fresh)
4. **Itinerary Assembler** - builds day-by-day plans with deliberate pacing (strong opener → peak experience → quiet farewell, inspired by Kahneman's Peak-End Rule)
5. **Critic Panel** - five critics check the plan in parallel (`asyncio.gather`). Logistics, energy pacing, budget and authenticity are rule-based checks. Experience fit is judged by the model.
6. **Revision loop** - if a critic flags an issue, the flagged days are rebuilt with the same scheduler and checked again (max 2 cycles)
7. **Highlight Ranker** - ranks the scheduled experiences into must-do / should-do / nice-to-have tiers

A separate **Replan Graph** handles mid-trip disruptions (weather, closures, "I'm exhausted") by re-analyzing remaining days without touching completed ones.

All intermediate outputs stream to the frontend via SSE.

## Run it

```bash
python3 -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env    # add your Groq API key
pytest -q
```

Plan a trip in the browser:

```bash
uvicorn src.api.main:app
```

Then open http://localhost:8000.

Replay the baseline with no API key and no network. The recorded model and weather calls are in `eval_results/recordings/baseline_v1/`:

```bash
python -m src.evaluation.runner --mode replay --report
```

## Evaluation (baseline_v1)

The planner is measured on 56 fixed scenarios (personas, budgets, accessibility needs, weather, disruptions) with hard checks such as schedule overlaps and budgets, plus soft metrics. Every model and weather call is recorded, so any run can be replayed exactly with no network. A run where a model call failed and a fallback planned instead is marked invalid and left out of every number. The full report is in `docs/baseline_v1_report.json` and the method in `docs/evaluation.md`.

| Metric | baseline_v1 (openai/gpt-oss-120b, 1 run per scenario) |
|---|---|
| Valid runs | 56 of 56 |
| Valid itinerary rate | 92.9% (95% CI 85.7 to 98.2, bootstrap over scenarios) |
| Failing scenarios | 3 low-budget trips end over budget, 1 ultra-low-budget trip crashes |
| Revision loop at its cap | 29 of 55 plans |
| Mean travel time per trip | 96 min (95% CI 84 to 110) |

**Comparing versions.** With 56 scenarios and a pass rate near 93%, the interval is about plus or minus 7 points. Two separate pass rates therefore can't show a 5-point improvement. Changes are compared on the same scenarios instead. We count the scenarios a change fixes and the ones it breaks, and bootstrap the difference over scenarios. A new planner version makes new model calls, so it runs with repeats, which keeps model noise from looking like an improvement. Reaching plus or minus 5 points on a single rate would need roughly twice as many scenarios.

### Hard constraints before scoring (v0.3_domain)

Wheelchair access and single-stop cost are now checked before any stop is scored, and a day over the budget ceiling can no longer be approved by the budget critic. The 52 scenarios this change does not touch replay exactly. The 4 budget scenarios plan differently, so their model calls were recorded again into `eval_results/recordings/v0.3_domain/`, once each.

| | baseline_v1 | v0.3_domain |
|---|---|---|
| Valid itinerary rate | 92.9% | 100% (56 of 56) |
| Crashes | 1 | 0 |
| Fixed / broken scenarios | | 4 / 0 |
| Mean persona score | 0.82 | 0.82 |

Four fixes against no breaks is not significant on its own (sign test p about 0.13). The case rests on each fix having a known cause and a test that failed before the change. The budget is met by removing the most expensive stop, so those days are shorter. Swapping in a cheaper stop of the same kind is planned. The report is in `docs/v0.3_domain_report.json`.

## Trip chat (router)

A traveller can talk to a finished plan, for example "day 2 feels rushed" or "add a fado evening to Saturday". The first step is a router. One model call with a forced tool returns the intent and the day the message is about. The intents are `change_setting`, `feedback`, `explain`, `what_if`, `ask`, `out_of_scope` and `none_of_these`. An invalid reply gets one retry, then the router falls back to `none_of_these`. Code in `src/chat/`. The router is measured on its own and is not connected to the app or the API yet.

It is tested on two hand-labelled sets. Amsterdam has 200 messages for a 3-day trip. Lisbon has 105 messages for a 4-day trip, written separately and held out. The v1 prompt can add similar labelled examples, picked by embedding similarity from the Amsterdam set. That condition is only measured on Lisbon, so no test message appears in its own examples.

| set | prompt | intent correct (95% CI) | intent and day correct |
|---|---|---|---|
| Amsterdam | v0 | 84.0% (78.5 to 89.0) | 81.5% |
| Lisbon | v0 | 77.1% (68.6 to 84.8) | 69.5% |
| Lisbon | v1 with examples | 97.1% (93.3 to 100) | 96.2% |

On v0, 11 of the 13 messages with the right intent but the wrong day were about "the last day" or "our final day". The v1 prompt returns them as a reference to the last day, and code turns that into a number for the trip at hand. All five prompt injection messages in the Lisbon set were routed as `out_of_scope`.

Replay all three with no API key:

```bash
python -m src.evaluation.chat_eval
python -m src.evaluation.chat_eval --set lisbon
python -m src.evaluation.chat_eval --set lisbon --prompt v1 --retrieval
```

## Status

In active development. See `docs/evaluation.md` for how the planner is measured.

## Project structure

```
src/
├── api/           # FastAPI endpoints + SSE streaming
├── agents/        # LangGraph graphs + agent nodes
│   └── nodes/critics/  # the 5 parallel critics
├── knowledge/     # city data (Amsterdam, Paris, Tokyo, London)
├── optimization/  # route optimizer + energy sequencer
├── evaluation/    # scenarios, validators, metrics, replay
└── models/        # Pydantic models
frontend/
└── index.html     # single-file UI
```
