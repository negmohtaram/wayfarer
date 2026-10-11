import json
import re
from src.agents.llm import call_with_retry, get_llm
from src.config import GROQ_MODEL
from src.agents.state import PlanningState
from src.models.persona import TravelerPersona


PERSONA_SYSTEM = """You are a warm, perceptive travel personality interviewer for WAYFARER — a premium AI travel planner.

Your job is to build a detailed traveler profile through a conversational 8-question quiz. It should feel like a lifestyle quiz from a premium magazine, not a form.

You will be given the traveler's destination, trip duration, and their previous answers. Ask ONE question at a time, thoughtfully building on what they've shared.

The 8 questions should cover (adapt based on flow):
1. Pace and energy (do they rush or savour?)
2. Food importance and preferences (dietary needs, local vs touristy)
3. Culture vs nature vs nightlife balance
4. Social context (solo, couple, family, friends) and physical constraints
5. Morning person vs evening person and crowd tolerance
6. Budget range per day
7. Local vs tourist experience preference
8. Photography interest and any special interests

After all 8 questions, output a JSON block (and ONLY this JSON, no text after it) exactly in this format:
```json
{
  "complete": true,
  "persona": {
    "pace": <0-10>,
    "budget_per_day": <integer euros>,
    "food_importance": <0-10>,
    "nature_importance": <0-10>,
    "culture_importance": <0-10>,
    "nightlife_importance": <0-10>,
    "walking_tolerance": <0-10>,
    "crowd_tolerance": <0-10>,
    "local_vs_tourist": <0-10>,
    "energy_profile": "<morning_person|evening_person|consistent>",
    "social_context": "<solo|couple|family|friends>",
    "physical_constraints": [],
    "dietary_requirements": [],
    "photography_interest": <0-10>,
    "trip_days": <integer>,
    "destination": "<city>"
  },
  "persona_narrative": "<2-3 sentence poetic description of this traveler>"
}
```

For each question (not the final JSON), respond in this format:
```json
{
  "complete": false,
  "question": "<the next question, warm and conversational>",
  "question_number": <1-8>,
  "hint": "<optional brief example answers to help them>"
}
```

Be warm, specific, and show you listened. Reference their previous answers. Make each question feel natural.
"""


async def persona_builder_node(state: PlanningState) -> PlanningState:
    """Processes a single turn of the persona-building conversation."""
    state.setdefault("log", [])
    state["log"].append("🧠 Building traveler profile...")
    return state


async def run_persona_turn(
    destination: str,
    trip_days: int,
    conversation_history: list[dict],
    user_message: str,
) -> dict:
    """
    Run one turn of the persona-building conversation.
    Returns either a question or the completed persona JSON.
    """
    llm = get_llm(temperature=0.8)

    messages = [{"role": "system", "content": PERSONA_SYSTEM}]

    context_msg = f"Trip details: destination={destination}, trip_days={trip_days}\n\n"
    if conversation_history:
        context_msg += "Conversation so far:\n"
        for msg in conversation_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            context_msg += f"{role.title()}: {content}\n"

    if user_message:
        context_msg += f"\nUser's latest answer: {user_message}"

    messages.append({"role": "user", "content": context_msg})

    response = await call_with_retry(
        llm.ainvoke, messages,
        meta={"site": "persona_builder", "model": GROQ_MODEL, "temperature": 0.8, "schema": None},
    )
    content = response.content

    # Extract JSON from response
    json_match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try parsing entire response as JSON
    try:
        return json.loads(content.strip())
    except json.JSONDecodeError:
        pass

    # Fallback: assume it's a question
    return {
        "complete": False,
        "question": content.strip(),
        "question_number": len(conversation_history) // 2 + 1,
        "hint": ""
    }


def build_persona_from_result(result: dict, destination: str, trip_days: int) -> TravelerPersona:
    persona_data = result.get("persona", {})
    persona_data["destination"] = destination
    persona_data["trip_days"] = trip_days
    return TravelerPersona(**persona_data)
