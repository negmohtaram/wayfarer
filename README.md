# WAYFARER

A multi-agent travel planner that cares about *who's traveling*, not just where.

Most travel planners optimize for logistics. WAYFARER builds itineraries around the traveler's personality, energy, and what they'd actually regret missing: then pressure-tests every plan through a panel of critic agents before anything ships.

## How it works

The system is a LangGraph pipeline with the following stages:

1. **Persona Builder** - an 8-turn conversation that profiles the traveler (pace, interests, budget, local-vs-tourist preference)
2. **Experience Generator** - scores 30+ city experiences against the persona
3. **Route & Energy Optimizer** - sequences the day using TSP nearest-neighbor, then reorders by energy curve (high-energy activities when the traveler is fresh)
4. **Itinerary Assembler** - builds day-by-day plans with deliberate pacing (strong opener → peak experience → quiet farewell, inspired by Kahneman's Peak-End Rule)
5. **Critic Panel** - 5 LLM agents run in parallel (`asyncio.gather`), each evaluating a different axis: logistics, experience fit, energy pacing, budget, and authenticity
6. **Revision loop** - if any critic flags an issue, a revision agent patches the plan (max 2 cycles)
7. **Regret Minimizer** - ranks everything into must-do / should-do / nice-to-have tiers

A separate **Replan Graph** handles mid-trip disruptions (weather, closures, "I'm exhausted") by re-analyzing remaining days without touching completed ones.

All intermediate outputs stream to the frontend via SSE.

## Status

In active development

## Project structure

```
src/
├── api/           — FastAPI endpoints + SSE streaming
├── agents/        — LangGraph graphs + agent nodes
│   └── nodes/critics/  — the 5 parallel critics
├── knowledge/     — city data (Amsterdam, Paris, Tokyo)
├── optimization/  — route optimizer + energy sequencer
└── models/        — Pydantic models
frontend/
└── index.html     — single-file UI
```
