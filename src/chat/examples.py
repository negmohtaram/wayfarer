"""Labelled example messages the router sees next to a new message.

The bank is the hand-labelled Amsterdam set (a 3-day trip). For each new message the most similar
examples are picked by embedding similarity, at most two per intent, so the router sees close
neighbours from more than one intent instead of six copies of the likeliest one.

Days in the bank belong to a 3-day trip. "The last day" is shown as day_ref=last, never as day 3,
so an example cannot teach the wrong number for a trip of another length.
"""
import re
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from src.evaluation.chat_messages import CHAT_MESSAGES

EMBEDDING_MODEL = "minishlab/potion-base-8M"
BANK_TRIP_DAYS = 3

Encoder = Callable[[list[str]], np.ndarray]


@dataclass(frozen=True)
class Example:
    text: str
    intent: str
    day: int | None
    day_ref: str | None

    def render(self) -> str:
        where = f", day_ref={self.day_ref}" if self.day_ref else (f", day={self.day}" if self.day else "")
        return f'"{self.text}" -> {self.intent}{where}'


def _example(text: str, intent: str, day: int | None) -> Example:
    if day == BANK_TRIP_DAYS and re.search(r"\b(last|final)\b", text, re.I):
        return Example(text, intent, None, "last")
    if day == 1 and re.search(r"\bfirst day\b", text, re.I):
        return Example(text, intent, None, "first")
    return Example(text, intent, day, None)


def default_encoder() -> Encoder:
    from model2vec import StaticModel
    model = StaticModel.from_pretrained(EMBEDDING_MODEL)
    return lambda texts: np.asarray(model.encode(texts))


class ExampleBank:
    def __init__(self, rows: Sequence[tuple] = CHAT_MESSAGES, encode: Encoder | None = None):
        self.examples = [_example(text, intent, day) for text, intent, day, *_ in rows]
        self.encode = encode or default_encoder()
        self.vectors = _unit(self.encode([e.text for e in self.examples]))

    def similarities(self, text: str) -> np.ndarray:
        return self.vectors @ _unit(self.encode([text]))[0]

    def retrieve(self, text: str, k: int = 6, per_intent: int = 2) -> list[Example]:
        picked, count = [], {}
        for i in np.argsort(-self.similarities(text)):
            ex = self.examples[i]
            if count.get(ex.intent, 0) < per_intent:
                picked.append(ex)
                count[ex.intent] = count.get(ex.intent, 0) + 1
            if len(picked) == k:
                break
        return picked

    def closest(self, text: str) -> tuple[float, str]:
        """Highest similarity to any bank message: used to prove a test set does not leak."""
        sims = self.similarities(text)
        i = int(np.argmax(sims))
        return float(sims[i]), self.examples[i].text


def render_examples(examples: list[Example]) -> str:
    lines = "\n".join(f"- {e.render()}" for e in examples)
    return f"Labelled examples from another trip, closest to this message first:\n{lines}"


def _unit(m: np.ndarray) -> np.ndarray:
    return m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-12, None)
