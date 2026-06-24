from pydantic import BaseModel, Field
from typing import Optional


class TravelerPersona(BaseModel):
    pace: float = Field(5.0, ge=0, le=10, description="0=rushed, 10=very slow")
    budget_per_day: int = Field(150, description="Daily budget in euros/dollars")
    food_importance: float = Field(5.0, ge=0, le=10)
    nature_importance: float = Field(5.0, ge=0, le=10)
    culture_importance: float = Field(5.0, ge=0, le=10)
    nightlife_importance: float = Field(3.0, ge=0, le=10)
    walking_tolerance: float = Field(5.0, ge=0, le=10)
    crowd_tolerance: float = Field(5.0, ge=0, le=10)
    local_vs_tourist: float = Field(5.0, ge=0, le=10, description="0=pure tourist, 10=pure local")
    energy_profile: str = Field("consistent", description="morning_person | evening_person | consistent")
    social_context: str = Field("solo", description="solo | couple | family | friends")
    physical_constraints: list[str] = Field(default_factory=list, description="bad_knee, wheelchair, elderly, etc.")
    dietary_requirements: list[str] = Field(default_factory=list)
    photography_interest: float = Field(5.0, ge=0, le=10, description="Affects golden hour scheduling")
    trip_days: int = Field(3, ge=1, le=14)
    destination: str = Field("amsterdam")
    trip_start_date: Optional[str] = Field(None, description="ISO date string YYYY-MM-DD")

    def persona_tags(self) -> list[str]:
        tags = []
        if self.pace >= 7:
            tags.append("Slow explorer")
        elif self.pace <= 3:
            tags.append("Fast mover")
        if self.food_importance >= 7:
            tags.append("Food-first")
        if self.local_vs_tourist >= 7:
            tags.append("Goes local")
        elif self.local_vs_tourist <= 3:
            tags.append("Iconic spots")
        if self.crowd_tolerance <= 3:
            tags.append("Avoids crowds")
        if self.energy_profile == "morning_person":
            tags.append("Morning person")
        elif self.energy_profile == "evening_person":
            tags.append("Night owl")
        if self.photography_interest >= 7:
            tags.append("Photographer")
        if self.culture_importance >= 7:
            tags.append("Culture lover")
        if self.nature_importance >= 7:
            tags.append("Nature seeker")
        if self.physical_constraints:
            tags.append("Accessibility needs")
        return tags

