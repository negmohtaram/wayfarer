from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Request, Response

from src.jobs.models import JobAccepted, JobView, PlanJobRequest
from src.models.persona import TravelerPersona
from src.services.session import load_session

router = APIRouter(tags=["jobs"])


@router.post("/plans", status_code=202, response_model=JobAccepted)
async def create_plan_job(req: PlanJobRequest, request: Request, response: Response,
                          idempotency_key: Optional[str] = Header(None)):
    try:
        TravelerPersona(**req.persona)          # bad input is rejected now, not inside the job
    except Exception as e:
        raise HTTPException(422, f"Invalid persona: {e}")
    job, created = await request.app.state.jobs.create("plan", req.model_dump(), idempotency_key)
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
async def get_plan(plan_id: str):
    return await load_session(plan_id)
