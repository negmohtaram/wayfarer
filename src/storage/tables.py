"""The jobs and plans tables. Migrations in alembic/versions create them; this is what the code reads."""
from sqlalchemy import (CheckConstraint, Column, DateTime, ForeignKey, Index, Integer, MetaData, Table, Text,
                        func, text)
from sqlalchemy.dialects.postgresql import JSONB, UUID

metadata = MetaData()

jobs = Table(
    "jobs", metadata,
    Column("job_id", UUID(as_uuid=False), primary_key=True),
    Column("kind", Text, nullable=False),
    Column("status", Text, nullable=False, server_default="queued"),
    Column("payload", JSONB, nullable=False),
    Column("idempotency_key", Text, unique=True),
    Column("attempts", Integer, nullable=False, server_default="0"),
    Column("reason", Text),
    Column("plan_id", Text),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("started_at", DateTime(timezone=True)),
    Column("heartbeat_at", DateTime(timezone=True)),
    Column("finished_at", DateTime(timezone=True)),
    CheckConstraint("status IN ('queued', 'running', 'done', 'failed')", name="jobs_status_valid"),
)

# Only queued rows are in this index, so it stays small however many jobs have finished.
Index("jobs_queued", jobs.c.created_at, postgresql_where=text("status = 'queued'"))

plans = Table(
    "plans", metadata,
    Column("plan_id", Text, primary_key=True),
    Column("job_id", UUID(as_uuid=False), ForeignKey("jobs.job_id"), nullable=False),
    Column("version", Integer, nullable=False, server_default="1"),
    Column("body", JSONB, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)
