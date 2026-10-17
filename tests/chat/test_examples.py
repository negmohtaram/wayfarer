import numpy as np

from src.chat.examples import ExampleBank, render_examples

ROWS = [
    ("Keep the last day light", "change_setting", 3, ""),
    ("The last day is a bit empty", "feedback", 3, ""),
    ("Why does the last day end early?", "explain", 3, ""),
    ("Is the last day far from the station?", "ask", 3, ""),
    ("Lower the budget", "change_setting", None, ""),
    ("Cut the budget to 50", "change_setting", None, ""),
    ("Can you pack more into the first day?", "change_setting", 1, ""),
]


def word_encoder(texts):
    """Bag of words over a fixed vocabulary: deterministic, no model download."""
    vocab = sorted({w for t, *_ in ROWS for w in t.lower().split()} | {"final", "day", "budget"})
    return np.array([[t.lower().split().count(w) for w in vocab] for t in texts], dtype=float)


def test_last_day_is_shown_as_a_position_never_as_day_3():
    bank = ExampleBank(ROWS, encode=word_encoder)
    rendered = render_examples(bank.retrieve("make the last day lighter", k=3))
    assert "day_ref=last" in rendered and "day=3" not in rendered


def test_first_day_becomes_a_position_too():
    bank = ExampleBank(ROWS, encode=word_encoder)
    first = [e for e in bank.examples if "first day" in e.text][0]
    assert first.day_ref == "first" and first.day is None


def test_no_more_than_two_examples_per_intent():
    bank = ExampleBank(ROWS, encode=word_encoder)
    picked = bank.retrieve("change the budget on the last day", k=5)
    intents = [e.intent for e in picked]
    assert max(intents.count(i) for i in set(intents)) <= 2
    assert len(set(intents)) >= 3                          # the top intent cannot fill the list


def test_a_test_message_copied_into_the_bank_is_caught():
    bank = ExampleBank(ROWS, encode=word_encoder)
    score, text = bank.closest("Lower the budget")
    assert score > 0.99 and text == "Lower the budget"
