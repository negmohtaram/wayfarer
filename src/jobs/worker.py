import asyncio
import contextlib
import logging
import time
from typing import Any, Callable

from src.agents.context import LLMContext
from src.agents.graph import planning_graph
from src.jobs.models import DEADLINE_PASSED, Job
from src.models.persona import TravelerPersona

log = logging.getLogger(__name__)


async def run_plan_job(job: Job, llm_factory: Callable[..., Any]) -> dict:
    """Run the planning graph for one job and return the plan. The store saves it."""
    persona = TravelerPersona(**job.payload["persona"])
    state = {"session_id": job.job_id, "persona": persona, "log": [], "revision_count": 0}
    final = await planning_graph.ainvoke(state, context=LLMContext(llm_factory))
    itinerary = final.get("itinerary")
    if itinerary is None:
        raise RuntimeError("planning finished without an itinerary")
    return {
        "plan_id": f"plan-{job.job_id}",     # same job, same plan: a rerun overwrites instead of leaving an orphan
        "version": 1,
        "job_id": job.job_id,
        "persona": persona.model_dump(mode="json"),
        "itinerary": itinerary.model_dump(mode="json"),
    }


class Worker:
    """One in-process worker: claims jobs, keeps their heartbeat fresh, records the result."""

    def __init__(self, store, llm_factory: Callable[..., Any], *, poll_s: float = 0.2,
                 heartbeat_s: float = 10.0, stale_after: float = 60.0, max_attempts: int = 3,
                 runner: Callable[[Job, Callable[..., Any]], Any] = run_plan_job,
                 clock: Callable[[], float] = time.time):
        self.store, self.llm_factory, self.runner, self.clock = store, llm_factory, runner, clock
        self.poll_s, self.heartbeat_s = poll_s, heartbeat_s
        self.stale_after, self.max_attempts = stale_after, max_attempts
        self.wake = asyncio.Event()

    async def run_forever(self) -> None:
        while True:
            await self.store.requeue_stale(self.stale_after, self.max_attempts)
            job = await self.store.claim_next()
            if job is None:
                self.wake.clear()
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self.wake.wait(), self.poll_s)
                continue
            await self.run_one(job)

    async def run_one(self, job: Job) -> None:
        # Postgres sets deadline_at with its own clock; a few seconds of skew against ours is fine at 120 s.
        left = job.deadline_at - self.clock() if job.deadline_at is not None else None
        if left is not None and left <= 0:
            await self.store.fail(job.job_id, DEADLINE_PASSED)    # nobody is waiting for this plan any more
            return
        beat = asyncio.create_task(self._beat(job.job_id))
        scope = asyncio.timeout(left)
        try:
            async with scope:                                     # cancels whatever the job is waiting on
                plan = await self.runner(job, self.llm_factory)
        except asyncio.CancelledError:
            raise       # shutdown: the job stays running and the next start requeues it
        except Exception as e:
            reason = DEADLINE_PASSED if scope.expired() else f"{type(e).__name__}: {e}"[:300]
            log.warning("job %s failed: %s", job.job_id, reason)
            await self.store.fail(job.job_id, reason)
        else:
            await self.store.finish(job.job_id, plan)
        finally:
            beat.cancel()

    async def _beat(self, job_id: str) -> None:
        while True:
            await asyncio.sleep(self.heartbeat_s)
            await self.store.heartbeat(job_id)
