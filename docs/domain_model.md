# Domain model

Wayfarer plans a trip from facts about the traveler and the city, and from what the traveler would like. These need different handling. A rule a plan must satisfy is checked and can reject a plan. A preference only ranks plans that already pass every rule. If both end up in one weighted score, a large bonus can hide a broken rule. In `budget_critic`, two days over budget on a 3-day trip still score 7.0, which is enough to approve.

Each field below gets two labels. The first says what it means.

| Label | Meaning | How the planner may use it |
|---|---|---|
| F | fact about the trip, the traveler or a place | as input; may feed a rule |
| H | hard constraint | rejects a plan that breaks it, never traded for points |
| S | soft preference | ranks valid plans only |
| D | derived, computed by code from other fields | recomputed, never edited directly |
| J | judgment, someone's rating | ranks valid plans, never decides validity |
| P | presentation text | shown to the traveler, never read by the planner |

The second says where the value comes from. That is the city catalogue in `src/knowledge/cities/*.json` (written by hand), the traveler through the persona builder (an LLM call), or code.

## TravelerPersona

| Field | Meaning | Source | Note |
|---|---|---|---|
| `pace` | S | persona builder | |
| `budget_per_day` | H | persona builder | costs are estimates, so the ceiling needs an explicit tolerance |
| `food_importance`, `nature_importance`, `culture_importance`, `nightlife_importance` | S | persona builder | |
| `walking_tolerance` | S, sometimes H | persona builder | a preference for most travelers and a limit for some, so it should become two fields |
| `crowd_tolerance` | S | persona builder | |
| `local_vs_tourist` | S | persona builder | |
| `energy_profile` | S | persona builder | |
| `social_context` | F | persona builder | creates rules (families) and preferences |
| `physical_constraints` | H | persona builder | free text today; needs fixed values to be checkable |
| `dietary_requirements` | H and S | persona builder | allergies and halal are rules, tastes are preferences; one list holds both |
| `photography_interest` | S | persona builder | |
| `trip_days` | F, H | traveler | never plan more days; fewer only with a stated reason |
| `destination` | F, H | traveler | every stop must be in this city |
| `trip_start_date` | F | traveler | drives opening days and weather |
| `persona_tags()` | D, P | code | labels for the UI |

Hard constraints taken from the traveler's own words go through an LLM. A misread preference costs a slightly worse plan. A misread rule can strand someone or exclude half the city, so hard constraints should be confirmed with the traveler before they bind.

## Experience

| Field | Meaning | Source | Note |
|---|---|---|---|
| `id`, `name`, `experience_type` | F | catalogue | |
| `location`, `neighborhood`, `city` | F | catalogue | `location` feeds travel time, `city` the destination rule |
| `duration_minutes` | F, estimate | catalogue | the day must fit |
| `best_time_of_day` | F and J | catalogue | holds both "only open at night" and "nicest at golden hour" |
| `cost_euros` | F, estimate | catalogue | what the budget rule is checked against |
| `weather_dependent` | F | catalogue | |
| `family_friendly` | F | catalogue | a yes/no where the real fact is often an age limit |
| `energy_required`, `crowd_level`, `local_score`, `romantic_score` | J | catalogue | ratings of the place |
| `physical_intensity` | J | catalogue | used today as a stand-in for wheelchair access |
| `description`, `why_this_matters`, `practical_tips` | P | catalogue | |
| `persona_score`, `highlight_score` | D | code | depend on the traveler, yet stored on the shared place object |
| `highlight_tier` | D, P | code | |

## Gaps found

1. Wheelchair access is not represented. The planner filters on `physical_intensity == "low"`, and a low-intensity museum can still have stairs. The plan is a `wheelchair_accessible` field that can be true, false or unknown, with unknown excluded for wheelchair users and the source of each value kept.
2. There are no opening hours, so a stop can be scheduled when it is closed.
3. Traveler-specific scores are written onto catalogue objects. The plan is to keep places unchanged and score them in a separate object with a breakdown of each term.
