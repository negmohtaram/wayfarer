import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
SESSION_DIR = Path(os.getenv("SESSION_DIR", "./sessions"))
SESSION_DIR.mkdir(exist_ok=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# postgresql+asyncpg://user:password@host:5432/db. Empty: jobs and plans stay in memory.
DATABASE_URL = os.getenv("DATABASE_URL", "")
# Which planner builds the days: "baseline" (cluster + route) or "v2a" (greedy + local search).
PLANNER = os.getenv("WAYFARER_PLANNER", "baseline")
GROQ_MODEL = "openai/gpt-oss-120b"
# Provider limits per model: (requests per minute, tokens per minute). Set them to your account's limits.
RATE_LIMITS = {
    "openai/gpt-oss-120b": (30, 8_000),
    "default": (30, 6_000),
}

KNOWLEDGE_DIR = BASE_DIR / "src" / "knowledge" / "cities"

SUPPORTED_CITIES = ["amsterdam", "paris", "tokyo", "london"]

MAX_REVISION_CYCLES = 2
EXPERIENCE_POOL_MULTIPLIER = 3  # generate 3x more experiences than needed
