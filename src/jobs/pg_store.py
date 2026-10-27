"""The job store on Postgres: same methods as InMemoryJobStore, so the worker and routes don't change.

Times come from the database clock (now()), so several servers agree on what "stale" means.
"""
import json
import uuid
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from src.jobs.models import IdempotencyKeyReused, Job

COLUMNS = ("job_id, kind, status, payload, idempotency_key, attempts, reason, plan_id, "
           "created_at, started_at, heartbeat_at, finished_at, deadline_at")


def _job(row: Any) -> Job:
    d = dict(row._mapping)
    d["job_id"] = str(d["job_id"])          # the driver returns uuid.UUID; the API speaks strings
    if isinstance(d["payload"], str):
        d["payload"] = json.loads(d["payload"])
    for k in ("created_at", "started_at", "heartbeat_at", "finished_at", "deadline_at"):
        d[k] = d[k].timestamp() if d[k] is not None else None
    return Job(**d)


def _is_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
        return True
    except ValueError:
        return False


class PostgresJobStore:
    def __init__(self, url: str, engine: Optional[AsyncEngine] = None):
        self.engine = engine or create_async_engine(url, pool_pre_ping=True)

    async def close(self) -> None:
        await self.engine.dispose()

    async def create(self, kind: str, payload: dict, idempotency_key: Optional[str] = None,
                     deadline_s: Optional[float] = None) -> tuple[Job, bool]:
        """A repeated idempotency key hits the UNIQUE constraint: nothing is inserted and the first job comes back,
        unless the first job was a different request (IdempotencyKeyReused)."""
        async with self.engine.begin() as conn:
            row = (await conn.execute(text(f"""
                INSERT INTO jobs (job_id, kind, payload, idempotency_key, deadline_at)
                VALUES (:job_id, :kind, CAST(:payload AS JSONB), :key,
                        now() + make_interval(secs => CAST(:deadline AS FLOAT)))
                ON CONFLICT (idempotency_key) DO NOTHING
                RETURNING {COLUMNS}"""),
                {"job_id": uuid.uuid4().hex, "kind": kind, "payload": json.dumps(payload), "key": idempotency_key,
                 "deadline": deadline_s},
            )).first()
            if row is not None:
                return _job(row), True
            row = (await conn.execute(text(f"SELECT {COLUMNS} FROM jobs WHERE idempotency_key = :key"),
                                      {"key": idempotency_key})).one()
            first = _job(row)
            if (first.kind, first.payload) != (kind, payload):     # JSONB ignores key order, dict == does too
                raise IdempotencyKeyReused(idempotency_key)
            return first, False

    async def get(self, job_id: str) -> Optional[Job]:
        if not _is_uuid(job_id):
            return None
        async with self.engine.connect() as conn:
            row = (await conn.execute(text(f"SELECT {COLUMNS} FROM jobs WHERE job_id = :id"), {"id": job_id})).first()
        return _job(row) if row else None

    async def claim_next(self) -> Optional[Job]:
        """Take the oldest queued job. SKIP LOCKED: two workers never get the same row, and never wait for each other."""
        async with self.engine.begin() as conn:
            row = (await conn.execute(text(f"""
                UPDATE jobs
                SET status = 'running', attempts = attempts + 1, started_at = now(), heartbeat_at = now()
                WHERE job_id = (
                    SELECT job_id FROM jobs WHERE status = 'queued'
                    ORDER BY created_at LIMIT 1
                    FOR UPDATE SKIP LOCKED)
                RETURNING {COLUMNS}"""))).first()
        return _job(row) if row else None

    async def heartbeat(self, job_id: str) -> None:
        async with self.engine.begin() as conn:
            await conn.execute(text("UPDATE jobs SET heartbeat_at = now() WHERE job_id = :id AND status = 'running'"),
                               {"id": job_id})

    async def finish(self, job_id: str, plan: dict) -> None:
        """Save the plan and mark the job done in one transaction: a crash between them leaves neither."""
        async with self.engine.begin() as conn:
            await conn.execute(text("""
                INSERT INTO plans (plan_id, job_id, version, body)
                VALUES (:plan_id, :job_id, :version, CAST(:body AS JSONB))
                ON CONFLICT (plan_id) DO UPDATE SET body = EXCLUDED.body, version = EXCLUDED.version"""),
                {"plan_id": plan["plan_id"], "job_id": job_id, "version": plan.get("version", 1),
                 "body": json.dumps(plan)})
            await conn.execute(text("""
                UPDATE jobs SET status = 'done', plan_id = :plan_id, finished_at = now() WHERE job_id = :id"""),
                {"plan_id": plan["plan_id"], "id": job_id})

    async def fail(self, job_id: str, reason: str) -> None:
        async with self.engine.begin() as conn:
            await conn.execute(text("""
                UPDATE jobs SET status = 'failed', reason = :reason, finished_at = now() WHERE job_id = :id"""),
                {"reason": reason, "id": job_id})

    async def requeue_stale(self, stale_after: float, max_attempts: int) -> list[Job]:
        async with self.engine.begin() as conn:
            stale = "status = 'running' AND heartbeat_at < now() - make_interval(secs => :stale)"
            failed = (await conn.execute(text(f"""
                UPDATE jobs SET status = 'failed', reason = 'worker_lost', finished_at = now()
                WHERE {stale} AND attempts >= :max RETURNING {COLUMNS}"""),
                {"stale": stale_after, "max": max_attempts})).all()
            requeued = (await conn.execute(text(f"""
                UPDATE jobs SET status = 'queued' WHERE {stale} RETURNING {COLUMNS}"""),
                {"stale": stale_after})).all()
        return [_job(r) for r in failed + requeued]

    async def get_plan(self, plan_id: str) -> Optional[dict]:
        async with self.engine.connect() as conn:
            row = (await conn.execute(text("SELECT body FROM plans WHERE plan_id = :id"), {"id": plan_id})).first()
        if row is None:
            return None
        return json.loads(row.body) if isinstance(row.body, str) else row.body
