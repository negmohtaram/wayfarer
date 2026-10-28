"""Plan jobs: 202 + Location, queued → running → done, idempotency, failure reasons, stale requeue."""

import asyncio
import time


from src.evaluation.scenarios.baseline_scenarios import SCENARIO_BY_ID
from src.jobs.models import DEADLINE_PASSED, JobStatus, JobView
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


def test_unknown_plan_is_404(client):
    assert client.get("/api/plans/nope").status_code == 404


def test_body_without_persona_is_422(client):
    resp = client.post("/api/plans", json={})
    assert resp.status_code == 422
    assert isinstance(resp.json()["detail"], list)          # FastAPI's own check, before our code runs


def test_persona_that_is_not_an_object_is_422(client):
    resp = client.post("/api/plans", json={"persona": "paris"})
    assert resp.status_code == 422
    assert isinstance(resp.json()["detail"], list)


def test_body_that_is_not_json_is_422(client):
    resp = client.post("/api/plans", content=b"plan me a trip", headers={"Content-Type": "application/json"})
    assert resp.status_code == 422


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


def test_a_hanging_job_fails_at_its_deadline():
    async def hangs(job, factory):
        await asyncio.Event().wait()               # a model call that never answers

    async def go():
        store = InMemoryJobStore()
        job, _ = await store.create("plan", {}, deadline_s=0.2)
        start = time.monotonic()
        await Worker(store, None, runner=hangs).run_one(await store.claim_next())
        return await store.get(job.job_id), time.monotonic() - start

    job, took = asyncio.run(go())
    assert job.status is JobStatus.FAILED and job.reason == DEADLINE_PASSED
    assert took < 1.0


def test_a_timeout_inside_the_job_keeps_its_own_reason():
    async def slow_call(job, factory):
        raise TimeoutError("weather service did not answer")

    async def go():
        store = InMemoryJobStore()
        job, _ = await store.create("plan", {}, deadline_s=60)
        await Worker(store, None, runner=slow_call).run_one(await store.claim_next())
        return await store.get(job.job_id)

    assert asyncio.run(go()).reason == "TimeoutError: weather service did not answer"


def test_a_job_past_its_deadline_is_failed_without_running():
    ran = []

    async def runner(job, factory):
        ran.append(job.job_id)

    async def go():
        clock = Clock()
        store = InMemoryJobStore(clock=clock)
        job, _ = await store.create("plan", {}, deadline_s=120)
        clock.now += 300                            # five minutes in the queue
        await Worker(store, None, runner=runner, clock=clock).run_one(await store.claim_next())
        return await store.get(job.job_id)

    job = asyncio.run(go())
    assert job.status is JobStatus.FAILED and job.reason == DEADLINE_PASSED and ran == []


def test_a_lost_job_past_its_deadline_is_failed_not_requeued():
    async def go():
        clock = Clock()
        store = InMemoryJobStore(clock=clock)
        job, _ = await store.create("plan", {}, deadline_s=120)
        await store.claim_next()                    # attempt 1 of 3, then the worker dies
        clock.now += 130
        await store.requeue_stale(stale_after=60, max_attempts=3)
        return await store.get(job.job_id)

    job = asyncio.run(go())
    assert job.status is JobStatus.FAILED and job.reason == DEADLINE_PASSED and job.attempts == 1


def test_a_timed_out_job_is_retryable_and_a_bad_request_is_not():
    async def hangs(job, factory):
        await asyncio.Event().wait()

    async def boom(job, factory):
        raise ValueError("catalogue missing for city")

    async def go():
        store = InMemoryJobStore()
        views = []
        for runner in (hangs, boom):
            job, _ = await store.create("plan", {}, deadline_s=0.1)
            await Worker(store, None, runner=runner).run_one(await store.claim_next())
            views.append(JobView.of(await store.get(job.job_id)))
        return views

    timed_out, crashed = asyncio.run(go())
    assert timed_out.retryable is True
    assert crashed.status == "failed" and crashed.retryable is False


def test_retry_after_a_failure_needs_a_new_idempotency_key():
    async def go():
        store = InMemoryJobStore()
        first, _ = await store.create("plan", {"persona": 1}, "k1", deadline_s=120)
        await store.fail(first.job_id, DEADLINE_PASSED)
        again, created_again = await store.create("plan", {"persona": 1}, "k1", deadline_s=120)
        retry, created_retry = await store.create("plan", {"persona": 1}, "k2", deadline_s=120)
        return first, again, created_again, retry, created_retry

    first, again, created_again, retry, created_retry = asyncio.run(go())
    assert again.job_id == first.job_id and not created_again        # one key, one outcome
    assert again.status is JobStatus.FAILED
    assert created_retry and retry.job_id != first.job_id and retry.status is JobStatus.QUEUED

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
