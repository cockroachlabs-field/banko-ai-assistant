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


# What granite3.3:8b really wrote in the Ollama smoke (task-13-smoke-report.md).
RECEIPT_EXPENSE = "20818c6c-4aa5-461a-a786-827173c43206"
MEMORY_ID = "37e68312-0b8a-4c43-9f0e-5a3c1d2e4f60"
GRANITE_RECEIPT = f"The amount of this receipt from Trader Joe's was $25.94 [txn:<{RECEIPT_EXPENSE}>]."
GRANITE_PLACEHOLDER = ("However, generally, dining out can be reduced by cooking at home, which could save "
                       "money [txn:<expense_id>]. Consider setting a budget.")
GRANITE_MEMORY = f"You are saving for a trip to Japan. * Goal: Saving for Japan [txn:<{MEMORY_ID}>]"
GRANITE_PAGE = "- [txn:<expense_id>] de83b07c-f3cf-4d62-8415-2eed569fcc88\n- Pizza [txn:<uuid>]"


def test_citation_ids_accept_angle_brackets_and_spaces():
    assert A.citation_ids(GRANITE_RECEIPT) == [RECEIPT_EXPENSE]
    assert A.citation_ids(f"wine [ txn: < {RECEIPT_EXPENSE.upper()} > ]") == [RECEIPT_EXPENSE]
    assert A.citation_ids(GRANITE_PLACEHOLDER) == []


def test_resolve_citations_keeps_known_ids_and_drops_everything_else():
    text = f"{GRANITE_RECEIPT} {GRANITE_PLACEHOLDER} {GRANITE_MEMORY}"
    out = A.resolve_citations(text, {RECEIPT_EXPENSE})
    assert f"[txn:{RECEIPT_EXPENSE}]" in out
    assert out.count("[txn:") == 1 and "<" not in out and MEMORY_ID not in out
    assert "could save money. Consider" in out and out.endswith("Saving for Japan")
    assert A.resolve_citations(GRANITE_PAGE, set()) == "- de83b07c-f3cf-4d62-8415-2eed569fcc88\n- Pizza"


def test_strip_citations_removes_every_variant():
    for text in (GRANITE_RECEIPT, GRANITE_PLACEHOLDER, GRANITE_MEMORY, GRANITE_PAGE,
                 f"x [txn:{RECEIPT_EXPENSE}]", "y [ TXN: whatever ]"):
        assert "txn" not in A.strip_citations(text).lower()


def test_insights_use_only_given_figures():
    chat = ScriptedChat("- You visit Blue Bottle often.")
    out = A.aggregation_insights(_agg(), "how much on coffee", chat)
    assert out == "- You visit Blue Bottle often."
    assert "$42.50" in chat.prompts()[-1]
