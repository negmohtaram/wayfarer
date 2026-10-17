import asyncio

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from src.chat.router import Intent, Route, RouterError, parse_route, route_message


def tool_reply(**args) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": "route_message", "args": args, "id": "c1"}])


class FakeModel:
    """Returns the given replies in order and keeps every message list it was sent."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.sent = []

    def bind_tools(self, tools, tool_choice=None):
        self.tools, self.tool_choice = tools, tool_choice
        return self

    async def ainvoke(self, messages):
        self.sent.append(messages)
        return self.replies.pop(0)


ALL = list(Intent)


def test_valid_call_becomes_a_route():
    route = parse_route(tool_reply(intent="feedback", day=2), ALL, trip_days=3)
    assert route == Route(intent=Intent.FEEDBACK, day=2)


@pytest.mark.parametrize("args, problem", [
    ({"intent": "change_budget"}, "intent"),             # made-up intent
    ({"intent": "explain", "day": "two"}, "day"),        # wrong type
    ({"intent": "explain", "day": 5}, "outside"),        # day beyond the trip
    ({}, "intent"),                                      # nothing at all
])
def test_bad_arguments_raise_router_error(args, problem):
    with pytest.raises(RouterError, match=problem):
        parse_route(tool_reply(**args), ALL, trip_days=3)


def test_no_tool_call_raises_router_error():
    with pytest.raises(RouterError, match="expected one"):
        parse_route(AIMessage(content="Sure, I lowered your budget!"), ALL, trip_days=3)


def test_intent_not_allowed_at_this_stage():
    with pytest.raises(RouterError, match="not allowed"):
        parse_route(tool_reply(intent="what_if"), [Intent.ASK, Intent.NONE_OF_THESE], trip_days=3)


def test_router_retries_once_with_the_error_fed_back():
    model = FakeModel(tool_reply(intent="change_budget"), tool_reply(intent="change_setting"))
    route = asyncio.run(route_message("lower the budget", trip_days=3, llm=model))
    assert route.intent is Intent.CHANGE_SETTING and route.error is None
    retry_messages = model.sent[1]
    assert isinstance(retry_messages[-1], ToolMessage)
    assert "change_budget" in retry_messages[-1].content
    assert retry_messages[1].content == "lower the budget"      # traveler text untouched


def test_router_falls_back_after_two_bad_calls():
    model = FakeModel(tool_reply(intent="nope"), tool_reply(intent="still_nope"))
    route = asyncio.run(route_message("lower the budget", trip_days=3, llm=model))
    assert route.intent is Intent.NONE_OF_THESE
    assert route.error


def test_router_forces_the_one_tool_and_offers_only_allowed_intents():
    model = FakeModel(tool_reply(intent="ask"))
    asyncio.run(route_message("is it far?", trip_days=3, allowed=[Intent.ASK], llm=model))
    assert model.tool_choice == "route_message"
    enum = model.tools[0]["function"]["parameters"]["properties"]["intent"]["enum"]
    assert enum == ["ask", "none_of_these"]


def test_broken_json_call_is_answered_with_its_id_on_retry():
    broken = AIMessage(content="", invalid_tool_calls=[
        {"name": "route_message", "args": '{"intent": "ask"', "id": "bad1", "error": "bad json"}])
    model = FakeModel(broken, tool_reply(intent="ask"))
    route = asyncio.run(route_message("is it far?", trip_days=3, llm=model))
    assert route.intent is Intent.ASK
    answer = model.sent[1][-1]
    assert isinstance(answer, ToolMessage) and answer.tool_call_id == "bad1"


@pytest.mark.parametrize("ref, trip_days, day", [
    ("last", 4, 4), ("last", 3, 3), ("first", 4, 1), ("middle", 5, 3), ("middle", 4, None),
])
def test_day_ref_is_resolved_by_code_from_the_trip_length(ref, trip_days, day):
    route = parse_route(tool_reply(intent="change_setting", day_ref=ref), ALL, trip_days=trip_days)
    assert route.day == day and route.day_ref.value == ref


def test_day_and_day_ref_that_disagree_are_rejected():
    with pytest.raises(RouterError, match="does not match"):
        parse_route(tool_reply(intent="feedback", day=3, day_ref="last"), ALL, trip_days=4)


def test_prompt_states_the_trip_length_and_v0_stays_unchanged():
    from src.chat.router import SYSTEM_V0, system_prompt
    assert "The trip has 4 days." in system_prompt(4)
    assert system_prompt(4, "v0") == SYSTEM_V0
    model = FakeModel(tool_reply(intent="ask"))
    asyncio.run(route_message("is it far?", trip_days=3, llm=model, prompt="v0"))
    assert "day_ref" not in model.tools[0]["function"]["parameters"]["properties"]
