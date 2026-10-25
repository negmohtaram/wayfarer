import asyncio
import contextlib
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from src.agents.llm import get_llm
from src.api.routers import chat, itineraries, jobs, meta, persona, planning
from src.config import GROQ_API_KEY
from src.jobs.store import InMemoryJobStore
from src.jobs.worker import Worker
from src.knowledge.rag import validate_city_knowledge

FRONTEND_DIR = Path(__file__).parent.parent.parent / "frontend"


def create_app(llm_factory: Callable[..., Any] = get_llm, require_key: bool = True) -> FastAPI:
    """Build the app around one model factory. Tests pass a fake one and require_key=False."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # On by default: a wrapped or custom factory must opt out on purpose, not by accident.
        if require_key and not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY environment variable is not set")
        validate_city_knowledge()
        app.state.llm_factory = llm_factory
        app.state.jobs = InMemoryJobStore()
        app.state.worker = Worker(app.state.jobs, llm_factory)
        worker_task = asyncio.create_task(app.state.worker.run_forever())
        yield
        worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker_task

    app = FastAPI(title="WAYFARER", description="Multi-agent AI travel planning system",
                  version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    for module in (persona, planning, jobs, chat, itineraries, meta):
        app.include_router(module.router, prefix="/api")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(FRONTEND_DIR / "index.html")

    return app


app = create_app()
