"""POST /api/trips/{id}/messages: route, extract a list of changes, build the card in code."""

from langchain_core.messages import AIMessage

from src.chat.card import build_card
from src.chat.extract import Change, ChangeList, Op, Slot
from src.chat.router import DayRef, Intent


def plan(days):
    return {"version": 3, "itinerary": {"destination": "amsterdam", "days": [
        {"day_number": i + 1, "experiences": [{"experience": {"id": f"d{i + 1}s{j}", "name": n}} for j, n in enumerate(names)]}
        for i, names in enumerate(days)]}}


PLAN = plan([["Rijksmuseum", "Vondelpark", "Canal boat tour"],
             ["Anne Frank House", "Jordaan walk"],
             ["Van Gogh Museum", "Albert Cuyp market", "Heineken Experience"]])


class ChatModel:
    """Routes every message as change_setting and returns the changes it was given."""

    def __init__(self, changes):
        self.changes, self.seen = changes, []

    def bind_tools(self, tools, tool_choice=None):
        return self

    def with_structured_output(self, schema, **kw):
        assert schema is ChangeList
        return self

    async def ainvoke(self, messages, **kw):
        self.seen.append(messages[-1].content)
        if "Current plan" in messages[-1].content:
            return ChangeList(changes=self.changes)
        return AIMessage(content="", tool_calls=[{"name": "route_message", "args": {"intent": "change_setting"}, "id": "c1"}])


def test_message_returns_intent_and_a_filled_card(client):
    client.app.state.jobs._plans["p1"] = PLAN
    model = ChatModel([Change(op=Op.MOVE, item="the Rijksmuseum", to_day=2, slot=Slot.MORNING)])
    client.app.state.llm_factory = lambda temperature=0.7: model

    body = client.post("/api/trips/p1/messages",
                       json={"text": "move the museum to day 2\nand put it in the morning"}).json()

    assert body["intent"] == "change_setting"
    card = body["card"]
    assert card["changes"][0]["item_id"] == "d1s0" and card["changes"][0]["to_day"] == 2
    assert card["summary"] == "Move Rijksmuseum from day 1 to day 2 (morning)"
    assert {e["day"]: (e["stops_before"], e["stops_after"]) for e in card["effects"]} == {1: (3, 2), 2: (2, 3)}
    assert card["can_confirm"] and card["plan_version"] == 3
    assert "Day 1: Rijksmuseum, Vondelpark" in model.seen[-1]      # the extractor saw the plan


def test_ambiguous_stop_asks_back_instead_of_guessing():
    card = build_card(Intent.CHANGE_SETTING, [Change(op=Op.REMOVE, item="the museum")], PLAN)
    assert not card.can_confirm and card.changes == []
    assert card.ask == "Which one do you mean: Rijksmuseum or Van Gogh Museum?"


def test_a_move_that_empties_a_day_below_the_minimum_is_blocked():
    card = build_card(Intent.CHANGE_SETTING, [Change(op=Op.MOVE, item="Jordaan", to_day=1)], PLAN)
    assert card.checks == ["Day 2 would have 1 stop; the minimum is 2."]
    assert not card.can_confirm


def test_several_changes_and_positions_are_resolved_by_code():
    card = build_card(Intent.CHANGE_SETTING, [
        Change(op=Op.REMOVE, item="boat tour"),
        Change(op=Op.ADD, item="a coffee place near the hotel", to_day_ref=DayRef.LAST),
    ], PLAN)
    assert [c.op for c in card.changes] == [Op.REMOVE, Op.ADD]
    assert card.changes[1].to_day == 3
    assert card.can_confirm


def test_a_day_past_the_trip_is_a_check_not_a_crash():
    card = build_card(Intent.CHANGE_SETTING, [Change(op=Op.MOVE, item="Vondelpark", to_day=7)], PLAN)
    assert "there is no day 7" in card.checks[0]


def test_unknown_plan_is_404(client):
    assert client.post("/api/trips/nope/messages", json={"text": "hi"}).status_code == 404
