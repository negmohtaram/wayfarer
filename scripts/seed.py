"""Add one finished demo trip (3 days in Paris) so a fresh database has a job and a plan to look at.

The plan is built by the planner alone, with no model calls, so seeding needs no API key and costs nothing.
Safe to run on every start: the fixed idempotency key means a second run finds the first job and stops.

    python -m scripts.seed
"""
import asyncio

import src.agents.nodes.assembler as assembler_module
from src.agents.nodes.assembler import assembler_node
from src.agents.nodes.experience_generator import experience_generator_node
from src.agents.nodes.optimizer import optimizer_node
from src.config import DATABASE_URL
from src.evaluation.scenarios.baseline_scenarios import SCENARIO_BY_ID
from src.jobs.pg_store import PostgresJobStore

SEED_KEY = "seed-demo-paris"
SCENARIO = "balanced_paris_reference"


async def _no_weather(city, start_date, days):
    return [None] * days


async def build_plan(job_id: str, persona) -> dict:
    state = {"session_id": job_id, "persona": persona, "log": [], "revision_count": 0, "planner": "v2a"}
    state = await experience_generator_node(state)
    state = await optimizer_node(state)
    original = assembler_module.fetch_weather
    assembler_module.fetch_weather = _no_weather
    try:
        state = await assembler_node(state)
    finally:
        assembler_module.fetch_weather = original
    return {
        "plan_id": f"plan-{job_id}",
        "version": 1,
        "job_id": job_id,
        "persona": persona.model_dump(mode="json"),
        "itinerary": state["itinerary"].model_dump(mode="json"),
    }


async def main() -> None:
    if not DATABASE_URL:
        raise SystemExit("DATABASE_URL is not set")
    persona = SCENARIO_BY_ID[SCENARIO].persona
    store = PostgresJobStore(DATABASE_URL)
    try:
        job, created = await store.create("plan", {"persona": persona.model_dump(mode="json")},
                                          idempotency_key=SEED_KEY)
        if not created:
            print(f"Seed job already there: {job.job_id} ({job.status.value})")
            return
        await store.finish(job.job_id, await build_plan(job.job_id, persona))
        print(f"Seeded demo job {job.job_id}")
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
