import json
import logging
from src.config import KNOWLEDGE_DIR
from src.models.experience import Experience

logger = logging.getLogger(__name__)


_city_cache: dict[str, dict] = {}


def _load_city(city: str) -> dict:
    if city not in _city_cache:
        path = KNOWLEDGE_DIR / f"{city}.json"
        if not path.exists():
            raise ValueError(f"No knowledge for city: {city}")
        with open(path) as f:
            _city_cache[city] = json.load(f)
    return _city_cache[city]


def get_city_experiences(city: str) -> list[Experience]:
    data = _load_city(city)
    experiences = []
    for exp_data in data.get("experiences", []):
        try:
            exp = Experience(**exp_data)
            experiences.append(exp)
        except Exception as e:
            logger.warning("Skipping invalid experience in %s: %s", city, e)
            continue
    return experiences


def get_city_neighborhoods(city: str) -> list[dict]:
    return _load_city(city).get("neighborhoods", [])


def get_city_food_scene(city: str) -> list[dict]:
    return _load_city(city).get("food_scene", [])


def get_city_practical_tips(city: str) -> dict:
    return _load_city(city).get("practical_tips", {})


def get_city_weather_month(city: str, month: str) -> dict:
    data = _load_city(city)
    return data.get("weather_by_month", {}).get(month.lower(), {})


def get_city_coordinates(city: str) -> list[float]:
    return _load_city(city).get("coordinates", [0.0, 0.0])


def get_city_photography_spots(city: str) -> list[dict]:
    return _load_city(city).get("photography_spots", [])


def get_city_summary(city: str) -> str:
    data = _load_city(city)
    neighborhoods = data.get("neighborhoods", [])
    nb_summary = "\n".join(
        f"- {n['name']}: {n['character'][:120]}..." for n in neighborhoods[:5]
    )
    tips = data.get("practical_tips", {})
    tip_text = "\n".join(f"- {k}: {v}" for k, v in list(tips.items())[:4])
    return f"""City: {data['city'].title()}, {data['country']}
Currency: {data['currency']} | Language: {data['language']}

Neighborhoods:
{nb_summary}

Key Tips:
{tip_text}
"""
