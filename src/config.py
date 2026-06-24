import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
SESSION_DIR = Path(os.getenv("SESSION_DIR", "./sessions"))
SESSION_DIR.mkdir(exist_ok=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"

KNOWLEDGE_DIR = BASE_DIR / "src" / "knowledge" / "cities"
DATA_DIR = BASE_DIR / "data" / "city_knowledge"

SUPPORTED_CITIES = ["amsterdam", "paris", "tokyo"]

MAX_REVISION_CYCLES = 2
EXPERIENCE_POOL_MULTIPLIER = 3  # generate 3x more experiences than needed
