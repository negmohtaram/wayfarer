"""The whole API on Postgres: a finished plan job is still there after the app restarts."""
import time

from fastapi.testclient import TestClient

from src.agents.nodes import assembler
from src.api import main
from src.evaluation.scenarios.baseline_scenarios import SCENARIO_BY_ID
from tests.api.conftest import FakeModel

def test_finished_job_and_plan_survive_an_app_restart(fresh_pg, monkeypatch):
    URL = fresh_pg
    monkeypatch.setattr(main, "GROQ_API_KEY", "")

    async def no_weather(city, start_date, days):
        return [None] * days

    monkeypatch.setattr(assembler, "_fetch_weather_live", no_weather)
    persona = SCENARIO_BY_ID["balanced_paris_reference"].persona

    def app():
        return TestClient(main.create_app(lambda temperature=0.7: FakeModel([]), require_key=False, database_url=URL))

    with app() as c:
        job_id = c.post("/api/plans", json={"persona": persona.model_dump(mode="json")}).json()["job_id"]
        for _ in range(200):
            job = c.get(f"/api/jobs/{job_id}").json()
            if job["status"] in ("done", "failed"):
                break
            time.sleep(0.05)
        assert job["status"] == "done", job

    with app() as c:                                   # a new app: nothing left in memory
        again = c.get(f"/api/jobs/{job_id}").json()
        plan = c.get(f"/api/plans/{again['plan_id']}").json()
    assert again["status"] == "done"
    assert len(plan["itinerary"]["days"]) == persona.trip_days


def test_job_id_that_is_not_a_uuid_is_404_on_postgres(fresh_pg, monkeypatch):
    monkeypatch.setattr(main, "GROQ_API_KEY", "")
    app = main.create_app(lambda temperature=0.7: FakeModel([]), require_key=False, database_url=fresh_pg)
    with TestClient(app) as c:
        assert c.get("/api/jobs/not-a-uuid").status_code == 404     # Postgres would reject the cast with a 500
