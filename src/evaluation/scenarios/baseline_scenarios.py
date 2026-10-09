"""
~50 baseline evaluation scenarios for baseline_v1.

Coverage:
  - pace extremes (slow/fast)
  - budget extremes (low/high)
  - social context (solo/couple/family/friends)
  - local vs tourist extremes
  - preference extremes (food/culture/nature/nightlife/photography)
  - physical constraints (wheelchair/bad_knee/elderly/combined)
  - energy profile (morning/evening/consistent)
  - trip length (1/2/5/7 days)
  - budget stress
  - injected weather (rainy/mixed/unavailable/sunny)
  - disruption/replanning
  - must-do/must-avoid (expected to fail on baseline — planner does not enforce them)

must_do/must_avoid scenarios are tagged expected_violation_codes so the report
correctly labels them as known unsupported features, not bugs.
"""

from src.evaluation.models import DisruptionSpec, EvaluationScenario
from src.models.persona import TravelerPersona

_START = "2026-09-10"

_RAINY = {"condition": "rainy", "temp_high": 14.0, "temp_low": 9.0, "rain_probability": 90.0}
_SUNNY = {"condition": "sunny", "temp_high": 22.0, "temp_low": 14.0, "rain_probability": 5.0}


def _p(**kwargs) -> TravelerPersona:
    defaults = dict(
        pace=5, budget_per_day=120, food_importance=5, nature_importance=5,
        culture_importance=5, nightlife_importance=3, walking_tolerance=6,
        crowd_tolerance=5, local_vs_tourist=5, energy_profile="consistent",
        social_context="solo", physical_constraints=[], dietary_requirements=[],
        photography_interest=5, trip_days=3, destination="amsterdam",
        trip_start_date=_START,
    )
    defaults.update(kwargs)
    return TravelerPersona(**defaults)


# ── Group 1: Pace extremes ─────────────────────────────────────────────────────

PACE_SLOW_SOLO_AMSTERDAM = EvaluationScenario(
    id="pace_slow_solo_amsterdam",
    description="Very slow solo traveler, amsterdam, 3 days",
    tags=["pace", "slow", "solo", "amsterdam"],
    persona=_p(pace=9, social_context="solo", destination="amsterdam"),
)

PACE_FAST_SOLO_AMSTERDAM = EvaluationScenario(
    id="pace_fast_solo_amsterdam",
    description="Fast-moving solo traveler, amsterdam, 3 days",
    tags=["pace", "fast", "solo", "amsterdam"],
    persona=_p(pace=2, social_context="solo", destination="amsterdam"),
)

PACE_SLOW_COUPLE_PARIS = EvaluationScenario(
    id="pace_slow_couple_paris",
    description="Very slow couple, paris, 3 days",
    tags=["pace", "slow", "couple", "paris"],
    persona=_p(pace=9, social_context="couple", destination="paris"),
)

PACE_FAST_COUPLE_PARIS = EvaluationScenario(
    id="pace_fast_couple_paris",
    description="Fast-moving couple, paris, 3 days",
    tags=["pace", "fast", "couple", "paris"],
    persona=_p(pace=2, social_context="couple", destination="paris"),
)

PACE_SLOW_FAMILY_TOKYO = EvaluationScenario(
    id="pace_slow_family_tokyo",
    description="Slow family with kids, tokyo, 3 days",
    tags=["pace", "slow", "family", "tokyo"],
    persona=_p(pace=8, social_context="family", destination="tokyo", walking_tolerance=4),
)

PACE_FAST_FRIENDS_LONDON = EvaluationScenario(
    id="pace_fast_friends_london",
    description="Fast-moving friends group, london, 3 days",
    tags=["pace", "fast", "friends", "london"],
    persona=_p(pace=2, social_context="friends", destination="london", nightlife_importance=7),
)

# ── Group 2: Budget extremes ───────────────────────────────────────────────────

BUDGET_LOW_SOLO_AMSTERDAM = EvaluationScenario(
    id="budget_low_solo_amsterdam",
    description="Very low budget solo, amsterdam, 3 days — may exceed daily ceiling",
    tags=["budget", "low", "solo", "amsterdam"],
    persona=_p(budget_per_day=40, social_context="solo", destination="amsterdam"),
)

BUDGET_HIGH_COUPLE_AMSTERDAM = EvaluationScenario(
    id="budget_high_couple_amsterdam",
    description="High budget couple, amsterdam, 3 days",
    tags=["budget", "high", "couple", "amsterdam"],
    persona=_p(budget_per_day=350, social_context="couple", destination="amsterdam"),
)

BUDGET_LOW_SOLO_PARIS = EvaluationScenario(
    id="budget_low_solo_paris",
    description="Very low budget solo, paris, 3 days",
    tags=["budget", "low", "solo", "paris"],
    persona=_p(budget_per_day=45, social_context="solo", destination="paris"),
)

BUDGET_HIGH_COUPLE_PARIS = EvaluationScenario(
    id="budget_high_couple_paris",
    description="High budget couple, paris, 3 days",
    tags=["budget", "high", "couple", "paris"],
    persona=_p(budget_per_day=300, social_context="couple", destination="paris"),
)

BUDGET_LOW_SOLO_TOKYO = EvaluationScenario(
    id="budget_low_solo_tokyo",
    description="Very low budget solo, tokyo, 3 days",
    tags=["budget", "low", "solo", "tokyo"],
    persona=_p(budget_per_day=50, social_context="solo", destination="tokyo"),
)

BUDGET_HIGH_COUPLE_LONDON = EvaluationScenario(
    id="budget_high_couple_london",
    description="High budget couple, london, 2 days",
    tags=["budget", "high", "couple", "london"],
    persona=_p(budget_per_day=400, social_context="couple", destination="london", trip_days=2),
)

# ── Group 3: Social context ────────────────────────────────────────────────────

SOCIAL_SOLO_AMSTERDAM = EvaluationScenario(
    id="social_solo_amsterdam",
    description="Baseline solo traveler, amsterdam, 3 days",
    tags=["social", "solo", "amsterdam"],
    persona=_p(social_context="solo", destination="amsterdam", local_vs_tourist=6),
)

SOCIAL_COUPLE_PARIS = EvaluationScenario(
    id="social_couple_paris",
    description="Romantic couple, paris, 3 days",
    tags=["social", "couple", "paris", "romantic"],
    persona=_p(social_context="couple", destination="paris", culture_importance=7, food_importance=7),
)

SOCIAL_FAMILY_TOKYO = EvaluationScenario(
    id="social_family_tokyo",
    description="Family with children, tokyo, 3 days — family_friendly filter expected",
    tags=["social", "family", "tokyo"],
    persona=_p(social_context="family", destination="tokyo", walking_tolerance=4, pace=7),
)

SOCIAL_FRIENDS_LONDON = EvaluationScenario(
    id="social_friends_london",
    description="Friend group, london, 2 days, nightlife-oriented",
    tags=["social", "friends", "london", "nightlife"],
    persona=_p(
        social_context="friends", destination="london", trip_days=2,
        nightlife_importance=8, energy_profile="evening_person",
    ),
)

# ── Group 4: Local vs tourist ──────────────────────────────────────────────────

LOCAL_HYPER_AMSTERDAM = EvaluationScenario(
    id="local_hyper_amsterdam",
    description="Hyper-local traveler, amsterdam — expects high local_score experiences",
    tags=["local", "amsterdam"],
    persona=_p(local_vs_tourist=9, destination="amsterdam", food_importance=8),
)

TOURIST_HYPER_PARIS = EvaluationScenario(
    id="tourist_hyper_paris",
    description="Classic tourist, paris — expects iconic high-crowd experiences",
    tags=["tourist", "paris"],
    persona=_p(local_vs_tourist=1, destination="paris", culture_importance=8, crowd_tolerance=8),
)

LOCAL_LEANING_TOKYO = EvaluationScenario(
    id="local_leaning_tokyo",
    description="Local-leaning traveler, tokyo",
    tags=["local", "tokyo"],
    persona=_p(local_vs_tourist=7, destination="tokyo"),
)

TOURIST_LEANING_LONDON = EvaluationScenario(
    id="tourist_leaning_london",
    description="Slightly tourist-leaning traveler, london",
    tags=["tourist", "london"],
    persona=_p(local_vs_tourist=3, destination="london", culture_importance=7),
)

# ── Group 5: Preference extremes ──────────────────────────────────────────────

PREF_FOOD_PARIS = EvaluationScenario(
    id="pref_food_paris",
    description="Food-obsessed local traveler, paris — food_importance=9",
    tags=["preference", "food", "paris"],
    persona=_p(food_importance=9, local_vs_tourist=8, destination="paris", culture_importance=3),
)

PREF_CULTURE_PARIS = EvaluationScenario(
    id="pref_culture_paris",
    description="Culture-first traveler, paris — culture_importance=9",
    tags=["preference", "culture", "paris"],
    persona=_p(culture_importance=9, destination="paris", local_vs_tourist=4),
)

PREF_NATURE_AMSTERDAM = EvaluationScenario(
    id="pref_nature_amsterdam",
    description="Nature-seeking traveler, amsterdam — nature_importance=9",
    tags=["preference", "nature", "amsterdam"],
    persona=_p(nature_importance=9, destination="amsterdam", crowd_tolerance=3),
)

PREF_NIGHTLIFE_AMSTERDAM = EvaluationScenario(
    id="pref_nightlife_amsterdam",
    description="Nightlife-first evening person, amsterdam, 2 days",
    tags=["preference", "nightlife", "amsterdam"],
    persona=_p(
        nightlife_importance=9, energy_profile="evening_person",
        destination="amsterdam", trip_days=2, crowd_tolerance=7,
    ),
)

PREF_PHOTOGRAPHY_PARIS = EvaluationScenario(
    id="pref_photography_paris",
    description="Photographer traveler, paris — golden hour focus",
    tags=["preference", "photography", "paris"],
    persona=_p(
        photography_interest=9, destination="paris",
        energy_profile="morning_person", local_vs_tourist=6,
    ),
)

# ── Group 6: Physical constraints ─────────────────────────────────────────────

CONSTRAINT_WHEELCHAIR_AMSTERDAM = EvaluationScenario(
    id="constraint_wheelchair_amsterdam",
    description="Wheelchair user, amsterdam — no high-intensity experiences allowed",
    tags=["accessibility", "wheelchair", "amsterdam"],
    persona=_p(
        physical_constraints=["wheelchair"], pace=9,
        destination="amsterdam", walking_tolerance=3,
    ),
)

CONSTRAINT_BAD_KNEE_PARIS = EvaluationScenario(
    id="constraint_bad_knee_paris",
    description="Traveler with bad knee, paris",
    tags=["accessibility", "bad_knee", "paris"],
    persona=_p(
        physical_constraints=["bad_knee"], pace=7,
        destination="paris", walking_tolerance=4,
    ),
)

CONSTRAINT_ELDERLY_TOKYO = EvaluationScenario(
    id="constraint_elderly_tokyo",
    description="Elderly traveler, tokyo — slow pace, low energy expected",
    tags=["accessibility", "elderly", "tokyo"],
    persona=_p(
        physical_constraints=["elderly"], pace=9,
        destination="tokyo", walking_tolerance=3, crowd_tolerance=3,
    ),
)

CONSTRAINT_MULTI_LONDON = EvaluationScenario(
    id="constraint_multi_london",
    description="Multiple constraints: wheelchair + bad_knee, london",
    tags=["accessibility", "wheelchair", "bad_knee", "london"],
    persona=_p(
        physical_constraints=["wheelchair", "bad_knee"], pace=9,
        destination="london", walking_tolerance=2, trip_days=2,
    ),
)

# ── Group 7: Energy profile ────────────────────────────────────────────────────

ENERGY_MORNING_AMSTERDAM = EvaluationScenario(
    id="energy_morning_amsterdam",
    description="Morning person, amsterdam — early starts, photography interest",
    tags=["energy", "morning_person", "amsterdam"],
    persona=_p(
        energy_profile="morning_person", destination="amsterdam",
        photography_interest=8, pace=6,
    ),
)

ENERGY_EVENING_PARIS = EvaluationScenario(
    id="energy_evening_paris",
    description="Evening person, paris — late starts, nightlife preference",
    tags=["energy", "evening_person", "paris"],
    persona=_p(
        energy_profile="evening_person", destination="paris",
        nightlife_importance=7, food_importance=7,
    ),
)

ENERGY_CONSISTENT_TOKYO = EvaluationScenario(
    id="energy_consistent_tokyo",
    description="Consistent energy profile, tokyo, balanced preferences",
    tags=["energy", "consistent", "tokyo"],
    persona=_p(energy_profile="consistent", destination="tokyo"),
)

# ── Group 8: Trip length ───────────────────────────────────────────────────────

LENGTH_ONEDAY_AMSTERDAM = EvaluationScenario(
    id="length_oneday_amsterdam",
    description="Single-day trip, amsterdam — stress tests clustering and scheduling",
    tags=["length", "1day", "amsterdam"],
    persona=_p(trip_days=1, destination="amsterdam", pace=4),
)

LENGTH_TWODAY_PARIS = EvaluationScenario(
    id="length_twoday_paris",
    description="Two-day trip, paris",
    tags=["length", "2day", "paris"],
    persona=_p(trip_days=2, destination="paris"),
)

LENGTH_FIVEDAY_TOKYO = EvaluationScenario(
    id="length_fiveday_tokyo",
    description="Five-day trip, tokyo — tests experience pool depth",
    tags=["length", "5day", "tokyo"],
    persona=_p(trip_days=5, destination="tokyo", pace=5),
)

LENGTH_SEVENDAY_LONDON = EvaluationScenario(
    id="length_sevenday_london",
    description="Seven-day trip, london — longest supported trip",
    tags=["length", "7day", "london"],
    persona=_p(trip_days=7, destination="london", pace=6),
)

# ── Group 9: Budget stress ─────────────────────────────────────────────────────

STRESS_ULTRA_LOW_BUDGET = EvaluationScenario(
    id="stress_ultra_low_budget",
    description="Ultra-low budget €25/day, amsterdam — below most experience costs",
    tags=["budget_stress", "amsterdam"],
    persona=_p(budget_per_day=25, destination="amsterdam"),
)

STRESS_HIGH_BUDGET_SLOW = EvaluationScenario(
    id="stress_high_budget_slow",
    description="High budget but very slow pace (few experiences), amsterdam",
    tags=["budget_stress", "slow", "amsterdam"],
    persona=_p(budget_per_day=200, pace=9, destination="amsterdam", trip_days=1),
)

STRESS_TYPICAL_BALANCED = EvaluationScenario(
    id="stress_typical_balanced",
    description="Typical traveler, balanced preferences, amsterdam — reference scenario",
    tags=["balanced", "amsterdam"],
    persona=_p(budget_per_day=120, pace=5, destination="amsterdam"),
)

# ── Group 10: Injected weather ────────────────────────────────────────────────

WEATHER_RAINY_ALL_AMSTERDAM = EvaluationScenario(
    id="weather_rainy_all_amsterdam",
    description="3 rainy days, amsterdam — weather_dependent experiences should score lower",
    tags=["weather", "rainy", "amsterdam"],
    persona=_p(destination="amsterdam"),
    injected_weather=[
        {**_RAINY, "date": "2026-09-10"},
        {**_RAINY, "date": "2026-09-11"},
        {**_RAINY, "date": "2026-09-12"},
    ],
)

WEATHER_MIXED_PARIS = EvaluationScenario(
    id="weather_mixed_paris",
    description="Day 1 sunny, day 2 rainy, day 3 sunny, paris",
    tags=["weather", "mixed", "paris"],
    persona=_p(destination="paris"),
    injected_weather=[
        {**_SUNNY, "date": "2026-09-10"},
        {**_RAINY, "date": "2026-09-11"},
        {**_SUNNY, "date": "2026-09-12"},
    ],
)

WEATHER_UNAVAILABLE_TOKYO = EvaluationScenario(
    id="weather_unavailable_tokyo",
    description="Weather API unavailable for all days, tokyo — tests available=False handling",
    tags=["weather", "unavailable", "tokyo"],
    persona=_p(destination="tokyo"),
    injected_weather=[None, None, None],
)

WEATHER_SUNNY_LONDON = EvaluationScenario(
    id="weather_sunny_london",
    description="All sunny days, london — baseline for weather comparison",
    tags=["weather", "sunny", "london"],
    persona=_p(destination="london"),
    injected_weather=[
        {**_SUNNY, "date": "2026-09-10"},
        {**_SUNNY, "date": "2026-09-11"},
        {**_SUNNY, "date": "2026-09-12"},
    ],
)

# ── Group 11: Disruption / replanning ─────────────────────────────────────────

DISRUPTION_WEATHER_AMSTERDAM = EvaluationScenario(
    id="disruption_weather_amsterdam",
    description="Weather disruption after day 1, amsterdam — outdoor experiences on day 2 should be replaced",
    tags=["disruption", "weather", "amsterdam"],
    persona=_p(destination="amsterdam", trip_days=3),
    disruption=DisruptionSpec(
        message="It's raining heavily today and the weather looks terrible. We need to change our plans.",
        completed_day_indices=[0],
        expected_stable_day_numbers=[1],
    ),
)

DISRUPTION_CLOSED_PARIS = EvaluationScenario(
    id="disruption_closed_paris",
    description="Landmark closed unexpectedly after day 1, paris",
    tags=["disruption", "closed", "paris"],
    persona=_p(destination="paris", trip_days=3),
    disruption=DisruptionSpec(
        message="The museum we planned to visit today is unexpectedly closed. We need to rearrange the day.",
        completed_day_indices=[0],
        expected_stable_day_numbers=[1],
    ),
)

DISRUPTION_TIRED_TOKYO = EvaluationScenario(
    id="disruption_tired_tokyo",
    description="Fatigue disruption after day 1, tokyo — high-energy activities should be removed",
    tags=["disruption", "tired", "tokyo"],
    persona=_p(destination="tokyo", trip_days=3),
    disruption=DisruptionSpec(
        message="We're completely exhausted from yesterday. We need a much slower day with easier activities.",
        completed_day_indices=[0],
        expected_stable_day_numbers=[1],
    ),
)

DISRUPTION_LOVE_NEIGHBORHOOD_LONDON = EvaluationScenario(
    id="disruption_love_neighborhood_london",
    description="Fell in love with a neighborhood after day 1, london — stay nearby",
    tags=["disruption", "love_neighborhood", "london"],
    persona=_p(destination="london", trip_days=3),
    disruption=DisruptionSpec(
        message="We absolutely love this neighborhood and want to spend more time exploring it.",
        completed_day_indices=[0],
        expected_stable_day_numbers=[1],
    ),
)

DISRUPTION_DELAYED_START_AMSTERDAM = EvaluationScenario(
    id="disruption_delayed_start_amsterdam",
    description="Late flight arrival, no completed days, amsterdam — first day needs rescheduling",
    tags=["disruption", "delay", "amsterdam"],
    persona=_p(destination="amsterdam", trip_days=3),
    disruption=DisruptionSpec(
        message="Our flight was delayed and we just arrived. We only have a few hours left today.",
        completed_day_indices=[],
        expected_stable_day_numbers=[],
    ),
)

# ── Group 12: Must-do / must-avoid ────────────────────────────────────────────
# These are expected to fail on baseline_v1 because the planner does not enforce them.
# expected_violation_codes documents this as a known limitation, not a bug.

MUST_DO_AMSTERDAM = EvaluationScenario(
    id="must_do_amsterdam",
    description="Must include ams_001 (dawn bike ride) — EXPECTED to fail; planner has no must-do enforcement",
    tags=["must_do", "amsterdam"],
    persona=_p(
        destination="amsterdam", energy_profile="morning_person",
        local_vs_tourist=8, nature_importance=7,
    ),
    must_do_ids=["ams_001"],
    expected_violation_codes=["MUST_DO_SATISFIED"],
)

MUST_AVOID_PARIS = EvaluationScenario(
    id="must_avoid_paris",
    description="Must avoid par_001 — EXPECTED to fail; planner has no must-avoid enforcement",
    tags=["must_avoid", "paris"],
    persona=_p(destination="paris", culture_importance=8),
    must_avoid_ids=["par_001"],
    expected_violation_codes=["MUST_AVOID_RESPECTED"],
)

# ── Additional edge cases to reach 50 ─────────────────────────────────────────

EDGE_ALL_CONSTRAINTS = EvaluationScenario(
    id="edge_all_constraints",
    description="Combined: slow + low budget + wheelchair + 1 day — extreme constraint stack",
    tags=["edge", "accessibility", "budget", "amsterdam"],
    persona=_p(
        pace=9, budget_per_day=50, physical_constraints=["wheelchair"],
        destination="amsterdam", trip_days=1, walking_tolerance=2,
    ),
)

EDGE_MAX_PACE_MAX_BUDGET = EvaluationScenario(
    id="edge_max_pace_max_budget",
    description="Maximum pace + maximum budget + 3 days, paris",
    tags=["edge", "pace", "budget", "paris"],
    persona=_p(
        pace=2, budget_per_day=500, destination="paris",
        culture_importance=8, food_importance=8,
    ),
)

EDGE_ZERO_CROWD_TOLERANCE = EvaluationScenario(
    id="edge_zero_crowd_tolerance",
    description="Extreme crowd aversion (crowd_tolerance=0), amsterdam",
    tags=["edge", "crowd", "amsterdam"],
    persona=_p(crowd_tolerance=0, destination="amsterdam", local_vs_tourist=9),
)

BALANCED_PARIS_REFERENCE = EvaluationScenario(
    id="balanced_paris_reference",
    description="Balanced reference scenario, paris — mirrors stress_typical_balanced for cross-city comparison",
    tags=["balanced", "paris"],
    persona=_p(budget_per_day=120, pace=5, destination="paris"),
)

BALANCED_TOKYO_REFERENCE = EvaluationScenario(
    id="balanced_tokyo_reference",
    description="Balanced reference scenario, tokyo",
    tags=["balanced", "tokyo"],
    persona=_p(budget_per_day=120, pace=5, destination="tokyo"),
)

BALANCED_LONDON_REFERENCE = EvaluationScenario(
    id="balanced_london_reference",
    description="Balanced reference scenario, london",
    tags=["balanced", "london"],
    persona=_p(budget_per_day=120, pace=5, destination="london"),
)

# ── All scenarios ──────────────────────────────────────────────────────────────

ALL_SCENARIOS: list[EvaluationScenario] = [
    PACE_SLOW_SOLO_AMSTERDAM,
    PACE_FAST_SOLO_AMSTERDAM,
    PACE_SLOW_COUPLE_PARIS,
    PACE_FAST_COUPLE_PARIS,
    PACE_SLOW_FAMILY_TOKYO,
    PACE_FAST_FRIENDS_LONDON,
    BUDGET_LOW_SOLO_AMSTERDAM,
    BUDGET_HIGH_COUPLE_AMSTERDAM,
    BUDGET_LOW_SOLO_PARIS,
    BUDGET_HIGH_COUPLE_PARIS,
    BUDGET_LOW_SOLO_TOKYO,
    BUDGET_HIGH_COUPLE_LONDON,
    SOCIAL_SOLO_AMSTERDAM,
    SOCIAL_COUPLE_PARIS,
    SOCIAL_FAMILY_TOKYO,
    SOCIAL_FRIENDS_LONDON,
    LOCAL_HYPER_AMSTERDAM,
    TOURIST_HYPER_PARIS,
    LOCAL_LEANING_TOKYO,
    TOURIST_LEANING_LONDON,
    PREF_FOOD_PARIS,
    PREF_CULTURE_PARIS,
    PREF_NATURE_AMSTERDAM,
    PREF_NIGHTLIFE_AMSTERDAM,
    PREF_PHOTOGRAPHY_PARIS,
    CONSTRAINT_WHEELCHAIR_AMSTERDAM,
    CONSTRAINT_BAD_KNEE_PARIS,
    CONSTRAINT_ELDERLY_TOKYO,
    CONSTRAINT_MULTI_LONDON,
    ENERGY_MORNING_AMSTERDAM,
    ENERGY_EVENING_PARIS,
    ENERGY_CONSISTENT_TOKYO,
    LENGTH_ONEDAY_AMSTERDAM,
    LENGTH_TWODAY_PARIS,
    LENGTH_FIVEDAY_TOKYO,
    LENGTH_SEVENDAY_LONDON,
    STRESS_ULTRA_LOW_BUDGET,
    STRESS_HIGH_BUDGET_SLOW,
    STRESS_TYPICAL_BALANCED,
    WEATHER_RAINY_ALL_AMSTERDAM,
    WEATHER_MIXED_PARIS,
    WEATHER_UNAVAILABLE_TOKYO,
    WEATHER_SUNNY_LONDON,
    DISRUPTION_WEATHER_AMSTERDAM,
    DISRUPTION_CLOSED_PARIS,
    DISRUPTION_TIRED_TOKYO,
    DISRUPTION_LOVE_NEIGHBORHOOD_LONDON,
    DISRUPTION_DELAYED_START_AMSTERDAM,
    MUST_DO_AMSTERDAM,
    MUST_AVOID_PARIS,
    EDGE_ALL_CONSTRAINTS,
    EDGE_MAX_PACE_MAX_BUDGET,
    EDGE_ZERO_CROWD_TOLERANCE,
    BALANCED_PARIS_REFERENCE,
    BALANCED_TOKYO_REFERENCE,
    BALANCED_LONDON_REFERENCE,
]

SCENARIO_BY_ID: dict[str, EvaluationScenario] = {s.id: s for s in ALL_SCENARIOS}
