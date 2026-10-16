from src.models.experience import Experience
from src.models.persona import TravelerPersona


def wheelchair_ok(experience: Experience, persona: TravelerPersona) -> bool:
    """One wheelchair rule for the planner, the replanner and the validator."""
    return "wheelchair" not in persona.physical_constraints or experience.physical_intensity == "low"
