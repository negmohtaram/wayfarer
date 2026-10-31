# WAYFARER

[![CI](https://github.com/negmohtaram/wayfarer/actions/workflows/ci.yml/badge.svg)](https://github.com/negmohtaram/wayfarer/actions/workflows/ci.yml)

A multi-agent travel planner that cares about *who's traveling*, not just where.

Most travel planners optimize for logistics. WAYFARER builds itineraries around the traveler's personality, energy, and what they'd regret missing. Every plan then goes through a panel of critics before the traveler sees it.

## How it works

The system is a LangGraph pipeline with the following stages:

1. **Persona Builder** - an 8-turn conversation that profiles the traveler (pace, interests, budget, local-vs-tourist preference)
2. **Experience Generator** - scores 30+ city experiences against the persona, after hard rules such as wheelchair access and budget have removed the stops that can't work
3. **Route & Energy Optimizer** - sequences each day. The default planner clusters stops and orders them by route and energy. A second planner (`WAYFARER_PLANNER=v2a`) uses greedy construction and local search and is being measured against it.
4. **Itinerary Assembler** - builds day-by-day plans with deliberate pacing (strong opener → peak experience → quiet farewell, inspired by Kahneman's Peak-End Rule)
5. **Critic Panel** - five critics check the plan in parallel (`asyncio.gather`). Logistics, energy pacing, budget and authenticity are rule-based checks. Experience fit is judged by the model.
6. **Revision loop** - if a critic flags an issue, the flagged days are rebuilt with the same scheduler and checked again (max 2 cycles)
7. **Highlight Ranker** - ranks the scheduled experiences into must-do / should-do / nice-to-have tiers

A separate **Replan Graph** handles mid-trip disruptions (weather, closures, "I'm exhausted") by re-analyzing remaining days without touching completed ones. Cities: Amsterdam, Paris, Tokyo and London.

## Run it

You need Docker and a [Groq](https://console.groq.com) API key.

```bash
git clone https://github.com/negmohtaram/wayfarer.git && cd wayfarer
cp .env.example .env    # add your Groq API key
docker compose up --build
```

Open http://localhost:8000. The first build downloads the base image and dependencies, so it takes a few minutes. Postgres starts first, the app waits for it, runs the migrations and adds a finished demo trip to Paris, made without any model calls.

To reset the database, run `docker compose down -v`.

### Without Docker

```bash
python3 -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
pytest -q
uvicorn src.api.main:app
```

Without `DATABASE_URL` in `.env`, jobs and plans are kept in memory and lost on restart.

### Troubleshooting

If the page says the planning model can't be reached, the server is running but Groq refused or timed out. Groq blocks some regions and some VPN exit addresses. If a request with a wrong key also gets 403 "Access denied", the block is on the network, not the key, so try another network or VPN server.

## API

Planning runs as a background job. `POST /api/plans` returns a job id and `GET /api/jobs/{id}` shows its progress. The interactive reference is at http://localhost:8000/docs. [docs/api.md](docs/api.md) explains how jobs behave and lists the error types.

## Results

The planner is measured on 56 fixed scenarios (personas, budgets, accessibility needs, weather, disruptions) with hard checks such as schedule overlaps and budgets, plus soft metrics. Every model and weather call is recorded, so any run can be replayed with no network. A run where a model call failed and a fallback planned instead is marked invalid and left out.

| | baseline_v1 | v0.3_domain |
|---|---|---|
| Valid itinerary rate | 92.9% (95% CI 85.7 to 98.2) | 100% (56 of 56) |
| Crashes | 1 | 0 |
| Fixed / broken scenarios | | 4 / 0 |
| Mean persona score | 0.82 | 0.82 |

The trip chat router sorts a traveler's message about a finished plan into an intent and a day. On a held-out set of 105 messages it gets 97.1% of intents right (95% CI 93.3 to 100).

Replay the baseline with no API key and no network:

```bash
python -m src.evaluation.runner --mode replay --report
```

How the numbers are made, how versions are compared and the full router results are in [docs/evaluation.md](docs/evaluation.md).

## Project structure

```
src/
├── api/           # FastAPI app and routes
├── jobs/          # job store (Postgres or memory) and background worker
├── agents/        # LangGraph graphs + agent nodes
│   └── nodes/critics/  # the 5 parallel critics
├── planning/      # planner v2a (greedy + local search)
├── optimization/  # route optimizer + energy sequencer
├── chat/          # trip chat router
├── knowledge/     # city data (Amsterdam, Paris, Tokyo, London)
├── evaluation/    # scenarios, validators, metrics, replay
└── models/        # Pydantic models
alembic/           # database migrations
scripts/seed.py    # demo trip added on start
frontend/
└── index.html     # single-file UI
compose.yaml       # app + Postgres
```
