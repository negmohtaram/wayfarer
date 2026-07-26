from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path

from src.api.routes import router
from src.config import GROQ_API_KEY
from src.knowledge.rag import validate_city_knowledge


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY environment variable is not set")
    validate_city_knowledge()
    yield


app = FastAPI(title="WAYFARER", description="Multi-agent AI travel planning system", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

# Serve frontend
FRONTEND_DIR = Path(__file__).parent.parent.parent / "frontend"

@app.get("/")
async def serve_frontend():
    return FileResponse(FRONTEND_DIR / "index.html")
