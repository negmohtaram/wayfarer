"""MCP server for Wayfarer.

The tools call the running API over HTTP. The job queue and its worker live in the API process,
so this server never plans anything itself; it only forwards calls and turns API errors into
messages a model can act on.

Run: WAYFARER_API_URL=http://localhost:8000 python -m src.mcp_server.server
"""

import hashlib
import json
import os
from typing import Annotated, Any, Optional
from urllib.parse import quote

import httpx
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, Field

from src.chat.extract import plan_outline
from src.evaluation.models import ValidationResult
from src.jobs.models import JobStatus
from src.models.persona import TravelerPersona

API_URL = os.getenv("WAYFARER_API_URL", "http://localhost:8000")


class JobStarted(BaseModel):
    job_id: str
    status: JobStatus


class JobState(BaseModel):
    job_id: str
    status: JobStatus
    attempts: int
    reason: Optional[str] = None          # why a failed job failed
    plan_id: Optional[str] = None         # set when the job is done
    outline: Optional[str] = None         # one line per day, set when the job is done


class PlanCheck(BaseModel):
    plan_id: str
    passed: bool                          # no failed check with severity "error"
    failures: list[ValidationResult]      # failed checks only, errors and warnings
    checks_run: list[str]


def idempotency_key(persona: TravelerPersona, attempt: int) -> str:
    """Same persona and attempt, same key: a repeated call finds the first job instead of starting another."""
    body = json.dumps(persona.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(f"{attempt}:{body}".encode()).hexdigest()


def _detail(resp: httpx.Response) -> str:
    try:
        detail = resp.json().get("detail", resp.text)
    except ValueError:
        return resp.text
    return detail.get("message", str(detail)) if isinstance(detail, dict) else str(detail)


async def _request(http: httpx.AsyncClient, method: str, path: str, **kwargs: Any) -> dict:
    try:
        resp = await http.request(method, path, **kwargs)
    except httpx.TransportError:
        raise ToolError(f"The Wayfarer API is not reachable at {http.base_url}. Start it, then try again.")
    if resp.status_code < 400:
        return resp.json()
    if resp.status_code == 404:
        raise ToolError(f"Not found: {_detail(resp)}")
    if resp.status_code == 422:
        raise ToolError(f"Rejected input: {_detail(resp)}")
    if resp.status_code == 503:
        wait = resp.headers.get("Retry-After", "30")
        raise ToolError(f"The planning model is unavailable right now. Try again in {wait} seconds.")
    raise ToolError(f"Wayfarer API error {resp.status_code}.")


def build_server(http: httpx.AsyncClient) -> MCPServer:
    mcp = MCPServer("wayfarer", instructions=(
        "Plan city trips. plan_trip starts a background job and returns at once; poll get_job_status "
        "until the status is done or failed. Changes to a plan are only proposed, never applied."))

    @mcp.tool()
    async def plan_trip(
        persona: TravelerPersona,
        attempt: Annotated[int, Field(ge=1, description=(
            "1 for a new request. Only after get_job_status reports failed, call again with the next number."))] = 1,
    ) -> JobStarted:
        """Start planning a trip for this traveler. Returns a job id at once; the plan is made in the background.
        Calling again with the same persona and attempt returns the same job."""
        data = await _request(http, "POST", "/api/plans", json={"persona": persona.model_dump(mode="json")},
                              headers={"Idempotency-Key": idempotency_key(persona, attempt)})
        return JobStarted(**data)

    @mcp.tool()
    async def get_job_status(job_id: Annotated[str, Field(description="Id returned by plan_trip")]) -> JobState:
        """Report a planning job's status. When it is done, also returns the plan id and a short outline."""
        job = await _request(http, "GET", f"/api/jobs/{quote(job_id, safe='')}")
        state = JobState(job_id=job["job_id"], status=job["status"], attempts=job["attempts"],
                         reason=job.get("reason"), plan_id=job.get("plan_id"))
        if state.status == JobStatus.DONE and state.plan_id:
            plan = await _request(http, "GET", f"/api/plans/{quote(state.plan_id, safe='')}")
            state.outline = plan_outline(plan)
        return state

    @mcp.tool()
    async def validate_itinerary(plan_id: Annotated[str, Field(description="Id from get_job_status")]) -> PlanCheck:
        """Run the hard checks (budget, timing, accessibility, duplicates and more) on a saved plan."""
        report = await _request(http, "GET", f"/api/plans/{quote(plan_id, safe='')}/validation")
        results = [ValidationResult(**r) for r in report["results"]]
        return PlanCheck(plan_id=report["plan_id"], passed=report["passed"],
                         failures=[r for r in results if not r.passed],
                         checks_run=sorted({r.code for r in results}))

    return mcp


def main() -> None:
    build_server(httpx.AsyncClient(base_url=API_URL, timeout=30)).run(transport="stdio")


if __name__ == "__main__":
    main()
