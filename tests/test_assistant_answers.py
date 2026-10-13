"""Deterministic answer pieces: aggregation cards, languages, citations."""

from datetime import date

from banko_ai.assistant import answers as A
from banko_ai.utils.aggregations import AggregationResult
from tests._chat import ScriptedChat


def _agg(**kw):
    base = dict(operation="sum", category=None, window_start=date(2026, 9, 1), window_end=date(2026, 10, 1),
                total=42.5, count=3, average=14.17, first_date=None, last_date=None,
                rows=[{"date": "2026-09-03", "merchant": "Blue Bottle", "amount": 14.5}])
    base.update(kw)
    return AggregationResult(**base)


def test_card_names_the_merchant_when_filtered():
    text = A.render_aggregation_markdown(_agg(merchant="Blue Bottle"))
    assert "**$42.50** at Blue Bottle" in text
    assert "| 2026-09-03 | Blue Bottle | $14.50 |" in text


def test_card_for_categories_and_empty_windows():
    assert "on Restaurant" in A.render_aggregation_markdown(_agg(category="Restaurant"))
    assert "No matching transactions" in A.render_aggregation_markdown(_agg(count=0, total=0.0, rows=[]))


def test_language_names():
    assert A.language_name("es-ES") == "Spanish"
    assert A.language_name(None) == "English"
    assert A.language_name("xx-XX") == "English"
    assert len(A.LANGUAGES) == 10


def test_translate_skips_english_and_survives_failure():
    chat = ScriptedChat("Hola")
    assert A.translate("Hello", "English", chat) == "Hello"
    assert chat.calls == []
    assert A.translate("Hello", "Spanish", chat) == "Hola"

    class Boom:
        def complete(self, messages, *, temperature=0.3):
            raise RuntimeError("down")

    assert A.translate("Hello", "Spanish", Boom()) == "Hello"


def test_citations():
    a, b = "11111111-1111-1111-1111-111111111111", "22222222-2222-2222-2222-222222222222"
    text = f"Coffee at Blue Bottle [txn:{a}] and again [txn:{b}] [txn:{a}]."
    assert A.citation_ids(text) == [a, b]
    assert A.strip_citations(text) == "Coffee at Blue Bottle and again."


def test_insights_use_only_given_figures():
    chat = ScriptedChat("- You visit Blue Bottle often.")
    out = A.aggregation_insights(_agg(), "how much on coffee", chat)
    assert out == "- You visit Blue Bottle often."
    assert "$42.50" in chat.prompts()[-1]
