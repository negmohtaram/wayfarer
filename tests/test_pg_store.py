"""PostgresJobStore against a real database. Runs only when TEST_DATABASE_URL points at a disposable one.

    TEST_DATABASE_URL=postgresql+asyncpg://wayfarer:wayfarer-dev@localhost:5432/wayfarer_test pytest tests/test_pg_store.py
"""
import asyncio

import pytest
from sqlalchemy import text

from src.jobs.models import IdempotencyKeyReused, JobStatus
from src.jobs.pg_store import PostgresJobStore
from tests.conftest import TEST_DATABASE_URL as URL

pytestmark = pytest.mark.usefixtures("fresh_pg")


def run(coro_fn):
    async def go():
        store = PostgresJobStore(URL)
        try:
            return await coro_fn(store)
        finally:
            await store.close()
    return asyncio.run(go())


def test_a_job_survives_a_restart():
    async def go(store):
        job, _ = await store.create("plan", {"persona": {"city": "paris"}})
        await store.close()
        fresh = PostgresJobStore(URL)              # new engine, new connections: like a restarted app
        try:
            return job, await fresh.get(job.job_id)
        finally:
            await fresh.close()

    job, again = run(go)
    assert again.status is JobStatus.QUEUED and again.payload == {"persona": {"city": "paris"}}


def test_same_key_at_the_same_moment_makes_one_job():
    async def go(store):
        results = await asyncio.gather(*[store.create("plan", {}, "abc-123") for _ in range(5)])
        async with store.engine.connect() as conn:
            count = (await conn.execute(text("SELECT count(*) FROM jobs"))).scalar()
        return results, count

    results, count = run(go)
    assert count == 1
    assert len({job.job_id for job, _ in results}) == 1
    assert sum(created for _, created in results) == 1


def test_same_key_with_a_different_payload_is_refused():
    async def go(store):
        await store.create("plan", {"persona": {"destination": "paris", "trip_days": 3}}, "k1")
        same = await store.create("plan", {"persona": {"trip_days": 3, "destination": "paris"}}, "k1")
        with pytest.raises(IdempotencyKeyReused):
            await store.create("plan", {"persona": {"destination": "tokyo", "trip_days": 3}}, "k1")
        return same

    _, created = run(go)
    assert created is False            # key order alone does not make it a different request


def test_two_workers_never_claim_the_same_job():
    async def go(store):
        for _ in range(4):
            await store.create("plan", {})
        return await asyncio.gather(*[store.claim_next() for _ in range(6)])

    claimed = [j for j in run(go) if j is not None]
    assert len(claimed) == 4
    assert len({j.job_id for j in claimed}) == 4
    assert all(j.status is JobStatus.RUNNING and j.attempts == 1 for j in claimed)


def test_finish_saves_the_plan_and_marks_done_together():
    async def go(store):
        job, _ = await store.create("plan", {})
        await store.claim_next()
        plan = {"plan_id": f"plan-{job.job_id}", "version": 1, "itinerary": {"days": []}}
        await store.finish(job.job_id, plan)
        await store.finish(job.job_id, plan)        # a rerun overwrites, no second plan
        async with store.engine.connect() as conn:
            plans = (await conn.execute(text("SELECT count(*) FROM plans"))).scalar()
        return await store.get(job.job_id), await store.get_plan(plan["plan_id"]), plans

    job, plan, plans = run(go)
    assert job.status is JobStatus.DONE and job.plan_id == plan["plan_id"] and plans == 1


def test_finish_for_a_missing_job_saves_nothing():
    async def go(store):
        with pytest.raises(Exception):              # foreign key: a plan must belong to a real job
            await store.finish("00000000-0000-0000-0000-000000000000", {"plan_id": "p", "itinerary": {}})
        return await store.get_plan("p")

    assert run(go) is None


def test_stale_job_is_requeued_then_failed():
    async def go(store):
        job, _ = await store.create("plan", {})
        for _ in range(3):
            await store.claim_next()
            async with store.engine.begin() as conn:  # the worker died 2 minutes ago
                await conn.execute(text("UPDATE jobs SET heartbeat_at = now() - interval '2 minutes'"))
            await store.requeue_stale(stale_after=60, max_attempts=3)
        return await store.get(job.job_id)

    job = run(go)
    assert job.status is JobStatus.FAILED and job.reason == "worker_lost" and job.attempts == 3


def test_bad_status_is_refused_by_the_database():
    async def go(store):
        job, _ = await store.create("plan", {})
        async with store.engine.begin() as conn:
            await conn.execute(text("UPDATE jobs SET status = 'donee' WHERE job_id = :id"), {"id": job.job_id})

    with pytest.raises(Exception, match="jobs_status_valid"):
        run(go)
