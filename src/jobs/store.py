import asyncio
import time
import uuid
from typing import Any, Callable, Optional

from src.jobs.models import IdempotencyKeyReused, Job, JobStatus


class InMemoryJobStore:
    """Job table kept in memory. The method set mirrors the Postgres repository that replaces it."""

    def __init__(self, clock: Callable[[], float] = time.time):
        self._clock = clock
        self._jobs: dict[str, Job] = {}
        self._by_key: dict[str, str] = {}
        self._plans: dict[str, dict] = {}
        self._lock = asyncio.Lock()

    async def create(self, kind: str, payload: dict[str, Any], idempotency_key: Optional[str] = None) -> tuple[Job, bool]:
        """Return (job, created). A repeated idempotency key returns the first job instead of a new one,
        or raises IdempotencyKeyReused when the request behind it is different."""
        async with self._lock:
            if idempotency_key and idempotency_key in self._by_key:
                first = self._jobs[self._by_key[idempotency_key]]
                if (first.kind, first.payload) != (kind, payload):
                    raise IdempotencyKeyReused(idempotency_key)
                return first, False
            job = Job(job_id=uuid.uuid4().hex, kind=kind, payload=payload,
                      idempotency_key=idempotency_key, created_at=self._clock())
            self._jobs[job.job_id] = job
            if idempotency_key:
                self._by_key[idempotency_key] = job.job_id
            return job, True

    async def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    async def claim_next(self) -> Optional[Job]:
        """Move the oldest queued job to running. Only one caller can win a given job."""
        async with self._lock:
            queued = [j for j in self._jobs.values() if j.status is JobStatus.QUEUED]
            if not queued:
                return None
            job = min(queued, key=lambda j: j.created_at)
            now = self._clock()
            job.status, job.attempts = JobStatus.RUNNING, job.attempts + 1
            job.started_at = job.heartbeat_at = now
            return job

    async def heartbeat(self, job_id: str) -> None:
        job = self._jobs[job_id]
        if job.status is JobStatus.RUNNING:
            job.heartbeat_at = self._clock()

    async def finish(self, job_id: str, plan: dict) -> None:
        """Save the plan and mark the job done together (one transaction in Postgres)."""
        async with self._lock:
            self._plans[plan["plan_id"]] = plan
            job = self._jobs[job_id]
            job.status, job.plan_id, job.finished_at = JobStatus.DONE, plan["plan_id"], self._clock()

    async def get_plan(self, plan_id: str) -> Optional[dict]:
        return self._plans.get(plan_id)

    async def fail(self, job_id: str, reason: str) -> None:
        job = self._jobs[job_id]
        job.status, job.reason, job.finished_at = JobStatus.FAILED, reason, self._clock()

    async def requeue_stale(self, stale_after: float, max_attempts: int) -> list[Job]:
        """Running jobs whose worker stopped sending heartbeats go back to the queue, or fail after max_attempts."""
        async with self._lock:
            now, touched = self._clock(), []
            for job in self._jobs.values():
                if job.status is JobStatus.RUNNING and now - job.heartbeat_at > stale_after:
                    if job.attempts >= max_attempts:
                        job.status, job.reason, job.finished_at = JobStatus.FAILED, "worker_lost", now
                    else:
                        job.status = JobStatus.QUEUED
                    touched.append(job)
            return touched
