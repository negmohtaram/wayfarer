from typing import Any, Callable, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.api.deps import get_llm_factory
from src.chat.card import ProposalCard, build_card
from src.chat.extract import extract_changes
from src.chat.handler import NEEDS_CHANGE, TurnSteps, handle_turn
from src.chat.router import Intent, route_message
from src.services.session import load_session

router = APIRouter(tags=["chat"])


class MessageIn(BaseModel):
    text: str


class MessageOut(BaseModel):
    intent: Intent
    day: Optional[int] = None
    card: Optional[ProposalCard] = None


def turn_steps(llm_factory: Callable[..., Any], loaded: dict) -> TurnSteps:
    async def load(plan_id):
        plan = loaded["plan"] = await load_session(plan_id)
        it = plan["itinerary"]
        return {"trip_days": len(it["days"]), "destination": it["destination"], "plan": plan}

    async def route(text, trip_days):
        return await route_message(text, trip_days, llm=llm_factory(temperature=0))

    async def extract(text, route, trip):
        return await extract_changes(text, trip["plan"], llm=llm_factory(temperature=0))

    async def scope(route, trip):
        return {"day": route.day}

    async def weather(city, scope):
        return None     # the saved plan already carries each day's forecast

    return TurnSteps(load_session=load, route=route, extract=extract, scope=scope, weather=weather)


@router.post("/trips/{plan_id}/messages", response_model=MessageOut)
async def post_message(plan_id: str, msg: MessageIn, llm_factory: Callable[..., Any] = Depends(get_llm_factory)):
    loaded: dict = {}
    result = await handle_turn(msg.text, plan_id, turn_steps(llm_factory, loaded))
    card = None
    if result.route.intent in NEEDS_CHANGE and result.change is not None:
        card = build_card(result.route.intent, result.change, loaded["plan"])
    return MessageOut(intent=result.route.intent, day=result.route.day, card=card)
