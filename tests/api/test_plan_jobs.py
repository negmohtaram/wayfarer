"""Plan jobs: 202 + Location, queued → running → done, idempotency, failure reasons, stale requeue."""

import asyncio
import time


from src.evaluation.scenarios.baseline_scenarios import SCENARIO_BY_ID
from src.jobs.models import JobStatus
from src.jobs.store import InMemoryJobStore
from src.jobs.worker import Worker

PERSONA = SCENARIO_BY_ID["balanced_paris_reference"].persona


def _wait_for(client, job_id, statuses, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        body = client.get(f"/api/jobs/{job_id}").json()
        if body["status"] in statuses:
            return body
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} stuck at {body['status']}")


def test_plan_job_runs_to_done_and_points_to_the_plan(client):
    resp = client.post("/api/plans", json={"persona": PERSONA.model_dump(mode="json")})

    assert resp.status_code == 202
    job_id = resp.json()["job_id"]
    assert resp.headers["location"] == f"/api/jobs/{job_id}"
    assert resp.json()["status"] == "queued"

    job = _wait_for(client, job_id, {"done", "failed"})
    assert job["status"] == "done", job
    assert job["attempts"] == 1 and job["started_at"] <= job["finished_at"]
    assert "itinerary" not in job                      # a pointer, not the plan

    plan = client.get(f"/api/plans/{job['plan_id']}").json()
    assert plan["version"] == 1
    assert len(plan["itinerary"]["days"]) == PERSONA.trip_days


def test_bad_persona_is_rejected_before_a_job_exists(client):
    resp = client.post("/api/plans", json={"persona": {"trip_days": "many"}})
    assert resp.status_code == 422


def test_same_idempotency_key_returns_the_same_job(client):
    body = {"persona": PERSONA.model_dump(mode="json")}
    a = client.post("/api/plans", json=body, headers={"Idempotency-Key": "k1"}).json()
    b = client.post("/api/plans", json=body, headers={"Idempotency-Key": "k1"}).json()
    assert a["job_id"] == b["job_id"]


def test_same_key_with_a_different_body_is_rejected(client):
    tokyo = PERSONA.model_copy(update={"destination": "tokyo"})
    first = client.post("/api/plans", json={"persona": PERSONA.model_dump(mode="json")},
                        headers={"Idempotency-Key": "k1"})
    second = client.post("/api/plans", json={"persona": tokyo.model_dump(mode="json")},
                         headers={"Idempotency-Key": "k1"})

    assert first.status_code == 202
    assert second.status_code == 422
    assert second.json()["detail"]["type"] == "idempotency_key_reused"
    assert len(client.app.state.jobs._jobs) == 1          # the Tokyo request created nothing


def test_unknown_job_is_404(client):
    assert client.get("/api/jobs/nope").status_code == 404


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_a_crashing_job_fails_with_its_reason():
    async def boom(job, factory):
        raise ValueError("catalogue missing for city")

    async def go():
        store = InMemoryJobStore()
        job, _ = await store.create("plan", {})
        await Worker(store, None, runner=boom).run_one(await store.claim_next())
        return await store.get(job.job_id)

    job = asyncio.run(go())
    assert job.status is JobStatus.FAILED
    assert job.reason == "ValueError: catalogue missing for city"


def test_stale_running_job_is_requeued_then_fails_after_max_attempts():
    async def go():
        clock = Clock()
        store = InMemoryJobStore(clock=clock)
        job, _ = await store.create("plan", {})
        for attempt in (1, 2, 3):
            claimed = await store.claim_next()             # worker starts it, then dies silently
            assert claimed.attempts == attempt
            clock.now += 30
            assert await store.requeue_stale(stale_after=60, max_attempts=3) == []   # slow, not dead
            clock.now += 31
            await store.requeue_stale(stale_after=60, max_attempts=3)
        return await store.get(job.job_id)

    job = asyncio.run(go())
    assert job.status is JobStatus.FAILED and job.reason == "worker_lost" and job.attempts == 3


def test_heartbeat_keeps_a_slow_job_alive():
    async def go():
        clock = Clock()
        store = InMemoryJobStore(clock=clock)
        job, _ = await store.create("plan", {})
        await store.claim_next()
        for _ in range(5):                                  # 5 minutes, beating every 50 s
            clock.now += 50
            await store.heartbeat(job.job_id)
            assert await store.requeue_stale(stale_after=60, max_attempts=3) == []
        return await store.get(job.job_id)

    assert asyncio.run(go()).status is JobStatus.RUNNING


def test_rerun_of_the_same_job_overwrites_its_plan(monkeypatch):
    from src.jobs import worker as worker_mod

    class Final:
        def model_dump(self, **kw):
            return {"days": []}

    async def fake_graph(state, context):
        return {"itinerary": Final()}

    monkeypatch.setattr(worker_mod.planning_graph, "ainvoke", fake_graph)

    async def go():
        store = InMemoryJobStore()
        job, _ = await store.create("plan", {"persona": PERSONA.model_dump(mode="json")})
        first = await worker_mod.run_plan_job(job, None)      # worker dies after this, before finish()
        second = await worker_mod.run_plan_job(job, None)     # requeued run
        await store.finish(job.job_id, second)
        return first, second, store

    first, second, store = asyncio.run(go())
    assert first["plan_id"] == second["plan_id"]
    assert len(store._plans) == 1
