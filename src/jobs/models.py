from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel


class IdempotencyKeyReused(Exception):
    """The key was already used for a different request. A retry sends the same body, so this is a client bug."""


DEADLINE_PASSED = "timeout: the plan deadline passed"
# Failures that say nothing about the request itself: the same request may work if sent again.
RETRYABLE_REASONS = {DEADLINE_PASSED, "worker_lost"}


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class Job(BaseModel):
    """One unit of background work. Status is about execution: an infeasible plan is still a done job."""

    job_id: str
    kind: str
    status: JobStatus = JobStatus.QUEUED
    payload: dict[str, Any]
    idempotency_key: Optional[str] = None
    attempts: int = 0
    reason: Optional[str] = None          # set when failed
    plan_id: Optional[str] = None         # set when done
    created_at: float
    started_at: Optional[float] = None
    heartbeat_at: Optional[float] = None
    finished_at: Optional[float] = None
    deadline_at: Optional[float] = None   # past this, the job fails instead of running on


class PlanJobRequest(BaseModel):
    persona: dict


class JobAccepted(BaseModel):
    job_id: str
    status: JobStatus


class JobView(BaseModel):
    """What GET /jobs/{id} returns: a pointer to the plan, never the plan itself."""

    job_id: str
    status: JobStatus
    attempts: int
    reason: Optional[str] = None
    plan_id: Optional[str] = None
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    deadline_at: Optional[float] = None
    retryable: bool = False     # failed, and sending the request again (with a new Idempotency-Key) may work

    @classmethod
    def of(cls, job: Job) -> "JobView":
        return cls(**job.model_dump(), retryable=job.status is JobStatus.FAILED and job.reason in RETRYABLE_REASONS)
