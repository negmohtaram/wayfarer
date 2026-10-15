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
