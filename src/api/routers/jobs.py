from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Request, Response

from src.config import PLAN_JOB_DEADLINE_S
from src.jobs.models import IdempotencyKeyReused, JobAccepted, JobView, PlanJobRequest
from src.models.persona import TravelerPersona

router = APIRouter(tags=["jobs"])


@router.post("/plans", status_code=202, response_model=JobAccepted)
async def create_plan_job(req: PlanJobRequest, request: Request, response: Response,
                          idempotency_key: Optional[str] = Header(None)):
    try:
        TravelerPersona(**req.persona)          # bad input is rejected now, not inside the job
    except Exception as e:
        raise HTTPException(422, f"Invalid persona: {e}")
    try:
        job, created = await request.app.state.jobs.create("plan", req.model_dump(), idempotency_key,
                                                            deadline_s=PLAN_JOB_DEADLINE_S)
    except IdempotencyKeyReused:
        # 422, as in the IETF Idempotency-Key draft; the type tells it apart from an invalid persona.
        raise HTTPException(422, {"type": "idempotency_key_reused",
                                  "message": f"Idempotency-Key {idempotency_key} was already used with a different request"})
    if created:
        request.app.state.worker.wake.set()
    response.headers["Location"] = f"/api/jobs/{job.job_id}"
    return JobAccepted(job_id=job.job_id, status=job.status)


@router.get("/jobs/{job_id}", response_model=JobView)
async def get_job(job_id: str, request: Request):
    job = await request.app.state.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"Job {job_id} not found")
    return JobView(**job.model_dump())


@router.get("/plans/{plan_id}")
async def get_plan(plan_id: str, request: Request):
    plan = await request.app.state.jobs.get_plan(plan_id)
    if plan is None:
        raise HTTPException(404, f"Plan {plan_id} not found")
    return plan
