"""Build the proposal card for a list of changes. Plain code: nothing here calls a model.

The card is what the traveler confirms. It carries the plan version it was built against,
so a confirm that arrives after the plan changed can be refused instead of applied to the
wrong plan.
"""
import uuid
from typing import Optional

from pydantic import BaseModel

from src.chat.extract import Change, Op, Slot
from src.chat.router import Intent, resolve_day

MIN_STOPS_PER_DAY = 2      # the planner's hard minimum (docs/planner_design.md)


class CardChange(BaseModel):
    op: Op
    item_id: Optional[str] = None
    item_name: Optional[str] = None
    from_day: Optional[int] = None
    to_day: Optional[int] = None
    slot: Optional[Slot] = None
    setting: Optional[str] = None
    value: Optional[str] = None


class Effect(BaseModel):
    day: int
    stops_before: int
    stops_after: int


class ProposalCard(BaseModel):
    card_id: str
    intent: Intent
    changes: list[CardChange]
    summary: str
    effects: list[Effect]
    checks: list[str]                # hard problems; any entry blocks confirm
    ask: Optional[str] = None        # question back when the message can't be resolved
    can_confirm: bool
    plan_version: int


def _stops(plan: dict) -> list[tuple[int, dict]]:
    return [(d["day_number"], s["experience"]) for d in plan["itinerary"]["days"] for s in d["experiences"]]


def find_stop(name: Optional[str], plan: dict) -> tuple[Optional[tuple[int, dict]], list[str]]:
    """Match the traveler's words to one stop. Returns (match, candidates when ambiguous)."""
    if not name:
        return None, []
    words = [w for w in name.lower().replace("the ", " ").split() if len(w) > 2]
    hits = [(day, e) for day, e in _stops(plan) if words and all(w in e["name"].lower() for w in words)]
    if len(hits) == 1:
        return hits[0], []
    return None, [e["name"] for _, e in hits]


def build_card(intent: Intent, changes: list[Change], plan: dict) -> ProposalCard:
    days = plan["itinerary"]["days"]
    n_days = len(days)
    counts = {d["day_number"]: len(d["experiences"]) for d in days}
    after = dict(counts)
    out, checks, asks, lines = [], [], [], []

    for ch in changes:
        to_day = ch.to_day or (resolve_day(ch.to_day_ref, n_days) if ch.to_day_ref else None)
        if to_day is not None and to_day > n_days:
            checks.append(f"The trip has {n_days} days, so there is no day {to_day}.")
            to_day = None
        if ch.op is Op.SETTING:
            out.append(CardChange(op=ch.op, setting=ch.setting, value=ch.value))
            lines.append(f"Set {ch.setting} to {ch.value}")
            continue
        if ch.op is Op.ADD:
            out.append(CardChange(op=ch.op, item_name=ch.item, to_day=to_day, slot=ch.slot))
            if to_day:
                after[to_day] += 1
            lines.append(f"Add {ch.item}" + (f" on day {to_day}" if to_day else ""))
            continue
        match, candidates = find_stop(ch.item, plan)
        if match is None:
            asks.append(f"Which one do you mean: {' or '.join(candidates)}?" if candidates
                        else f"I couldn't find {ch.item!r} in your plan. Which stop do you mean?")
            continue
        from_day, exp = match
        out.append(CardChange(op=ch.op, item_id=exp["id"], item_name=exp["name"],
                              from_day=from_day, to_day=to_day, slot=ch.slot))
        if ch.op is Op.REMOVE:
            after[from_day] -= 1
            lines.append(f"Remove {exp['name']} from day {from_day}")
        elif ch.op is Op.MOVE and to_day:
            after[from_day] -= 1
            after[to_day] += 1
            when = f" ({ch.slot.value})" if ch.slot else ""
            lines.append(f"Move {exp['name']} from day {from_day} to day {to_day}{when}")
        elif ch.op is Op.MOVE and ch.slot:
            lines.append(f"Move {exp['name']} to the {ch.slot.value} of day {from_day}")
        else:
            lines.append(f"{ch.op.value.title()} {exp['name']}")

    for day, n in after.items():
        if n < MIN_STOPS_PER_DAY and n < counts[day]:
            checks.append(f"Day {day} would have {n} stop{'s' if n != 1 else ''}; the minimum is {MIN_STOPS_PER_DAY}.")
    if not out and not asks:
        asks.append("What would you like to change?")

    return ProposalCard(
        card_id=uuid.uuid4().hex, intent=intent, changes=out, summary="; ".join(lines),
        effects=[Effect(day=d, stops_before=counts[d], stops_after=after[d]) for d in counts if after[d] != counts[d]],
        checks=checks, ask=" ".join(asks) or None,
        can_confirm=bool(out) and not checks and not asks,
        plan_version=plan.get("version", 1),
    )
