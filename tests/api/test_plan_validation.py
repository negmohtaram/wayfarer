"""GET /api/plans/{id}/validation: the hard checks on a saved plan."""

import copy

from tests.api.test_plan_jobs import PERSONA, _wait_for


def _finished_plan(client) -> dict:
    job_id = client.post("/api/plans", json={"persona": PERSONA.model_dump(mode="json")}).json()["job_id"]
    job = _wait_for(client, job_id, {"done", "failed"})
    assert job["status"] == "done", job
    return client.get(f"/api/plans/{job['plan_id']}").json()


def test_saved_plan_gets_a_report_with_every_check(client):
    plan = _finished_plan(client)

    resp = client.get(f"/api/plans/{plan['plan_id']}/validation")

    assert resp.status_code == 200
    report = resp.json()
    assert report["plan_id"] == plan["plan_id"]
    codes = {r["code"] for r in report["results"]}
    assert {"DAY_COUNT_MATCHES", "NO_DUPLICATE_EXPERIENCES", "NO_SCHEDULE_OVERLAP"} <= codes, codes
    assert "MUST_DO_SATISFIED" not in codes          # personas have no must-do list, so the check is skipped
    hard_failures = [r for r in report["results"] if r["severity"] == "error" and not r["passed"]]
    assert report["passed"] == (not hard_failures)


def test_a_broken_plan_fails_and_says_where(client):
    plan = _finished_plan(client)
    broken = copy.deepcopy(plan)
    day = broken["itinerary"]["days"][0]
    day["experiences"].append(copy.deepcopy(day["experiences"][0]))   # same place twice
    broken["plan_id"] = "broken"
    client.app.state.jobs._plans["broken"] = broken

    report = client.get("/api/plans/broken/validation").json()

    assert report["passed"] is False
    dup = [r for r in report["results"] if r["code"] == "NO_DUPLICATE_EXPERIENCES" and not r["passed"]]
    assert dup and dup[0]["severity"] == "error", report["results"]


def test_validation_does_not_change_the_plan(client):
    plan = _finished_plan(client)
    client.get(f"/api/plans/{plan['plan_id']}/validation")
    assert client.get(f"/api/plans/{plan['plan_id']}").json() == plan


def test_unknown_plan_is_404(client):
    assert client.get("/api/plans/nope/validation").status_code == 404
