# WAYFARER

A multi-agent AI travel planning system that models the traveler as deeply as it models the destination.

## What makes it different

Most travel planners optimize for places. WAYFARER optimizes for **experiences** — specifically the right experiences for *this exact traveler*, in the right order, with the right pacing, designed as a deliberate emotional arc.

**Core innovations:**
- **Emotional arc design** (Kahneman's Peak-End Rule) — deliberate Day 1/Peak/Farewell structure
- **Energy curve modeling** — sequences activities by traveler energy, not just geography
- **Local vs tourist axis** — explicit 0–10 dimension controlling entire venue selection
- **Serendipity zones** — designated wander areas with no schedule
- **5 parallel critic agents** (logistics, experience, energy, budget, authenticity)
- **Regret minimizer** — ranks every experience by what this traveler would regret missing
- **Dynamic replanning** — handles real-time disruptions during the trip

## How it works

WAYFARER is built on a **LangGraph multi-agent pipeline** where each stage is a graph node:

```
Persona Builder (8-turn LLM conversation → TravelerPersona)
  ↓
Experience Generator (scores all city experiences against persona)
  ↓
Route + Energy Optimizer (TSP nearest-neighbor + energy curve matching)
  ↓
Assembler (builds day-by-day schedule with arc labels + weather)
  ↓
Critic Panel — 5 LLM agents run in parallel via asyncio.gather
  ↓
[Conditional edge] — if any critic fails, route to Revision Agent (max 2 cycles)
  ↓
Regret Minimizer (must_do / should_do / nice_to_have tiers)
```

A separate **Replan Graph** handles mid-trip disruptions (weather, closures, fatigue) by re-analyzing remaining days.

All intermediate agent outputs are streamed to the frontend in real time via **Server-Sent Events (SSE)**.

## Replanning

During a trip, travelers can report disruptions ("it's raining", "the museum is closed", "I'm exhausted"). WAYFARER's disruption analyzer classifies the issue, filters the experience pool accordingly, and reassembles the remaining days — without touching completed days.

## Quick Start

### Prerequisites
- Python 3.11+
- A free [Groq API key](https://console.groq.com) (llama-3.3-70b-versatile)

### Setup

```bash
# Clone and enter the project
git clone <repo-url>
cd wayfarer

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Configure API key
cp .env.example .env
# Edit .env and add your Groq API key
```

### Run

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Open http://localhost:8000

## Architecture

```
User
  ↓
Persona Builder (8-question conversational quiz)
  ↓
City Intelligence Layer (curated local knowledge for Amsterdam, Paris, Tokyo)
  ↓
Experience Pool Generator (30+ candidates scored against persona)
  ↓
Route + Energy Optimizer (TSP-inspired + energy curve sequencing)
  ↓
Itinerary Assembler (day-by-day with emotional arc design)
  ↓
Critic Panel — 5 agents in parallel (LangGraph)
  ├─ Logistics Critic
  ├─ Experience Critic
  ├─ Energy Critic
  ├─ Budget Critic
  └─ Authenticity Critic
  ↓
Revision Agent (targeted fixes, max 2 cycles)
  ↓
Regret Minimizer (Must Do / Should Do / Nice to Have)
  ↓
Dynamic Replanner (handles disruptions in real time)
```

## Supported Cities

- **Amsterdam** (30 curated experiences, 5 neighborhoods)
- **Paris** (15 curated experiences, 5 neighborhoods)
- **Tokyo** (15 curated experiences, 5 neighborhoods)

## Free APIs Used

- **Groq** (llama-3.3-70b-versatile) — LLM inference
- **Open-Meteo** — weather forecasts (no key required)
- **CartoDB** — dark map tiles (free for reasonable use)
- **Leaflet.js** — map rendering
- **Overpass/Nominatim** — available for future POI enrichment

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + Uvicorn (async Python) |
| Agent orchestration | LangGraph (StateGraph, conditional edges, parallel nodes) |
| LLM | Groq — llama-3.3-70b-versatile |
| Data validation | Pydantic v2 |
| Frontend | Vanilla JS + Leaflet.js (no framework, no build step) |
| Maps | CartoDB dark tiles |
| Weather | Open-Meteo (free, no key needed) |
| Session storage | File-based JSON (no database) |

## File Structure

```
wayfarer/
├── src/
│   ├── api/          — FastAPI endpoints + SSE streaming
│   ├── agents/       — LangGraph graphs + all agent nodes
│   │   └── nodes/
│   │       └── critics/  — 5 parallel critic agents
│   ├── knowledge/    — City knowledge base + retrieval
│   │   └── cities/   — Amsterdam, Paris, Tokyo JSON data
│   ├── optimization/ — TSP route optimizer + energy sequencer
│   └── models/       — Pydantic models for all data types
├── frontend/
│   └── index.html    — Complete UI in a single file
└── sessions/         — JSON session storage (gitignored)
```
