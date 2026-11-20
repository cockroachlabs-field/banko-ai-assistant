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


def test_card_shows_the_last_day_counted_not_the_day_after():
    """The window end is exclusive; "between 2026-09-13 to 2026-10-14" for
    the last 30 days, asked on the 13th, showed tomorrow."""
    text = A.render_aggregation_markdown(_agg(category="Restaurant"))
    assert "on Restaurant from 2026-09-01 to 2026-09-30 (3 transactions)" in text
    assert "2026-10-01" not in text and "between" not in text
    empty = A.render_aggregation_markdown(_agg(count=0, total=0.0, rows=[]))
    assert "from 2026-09-01 to 2026-09-30" in empty and "between" not in empty
    for op in ("count", "average"):
        assert "from 2026-09-01 to 2026-09-30" in A.render_aggregation_markdown(_agg(operation=op))


def test_insights_see_the_same_window_as_the_card():
    chat = ScriptedChat("- ok")
    A.aggregation_insights(_agg(), "how much on coffee", chat)
    assert "from 2026-09-01 to 2026-09-30" in chat.prompts()[-1] and "2026-10-01" not in chat.prompts()[-1]


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


def test_second_person_flips_the_common_cases():
    assert A.second_person("I'm saving for Japan") == "you're saving for Japan"
    assert A.second_person("my rent is due on the 1st") == "your rent is due on the 1st"
    assert A.second_person("My goal is an emergency fund") == "your goal is an emergency fund"
    assert A.second_person("I prefer paying with my card") == "you prefer paying with your card"
    assert A.second_person("I am done with I've and I'll") == "you are done with you've and you'll"
    assert A.second_person("I was paid by my landlord") == "you were paid by your landlord"


def test_insights_use_only_given_figures():
    chat = ScriptedChat("- You visit Blue Bottle often.")
    out = A.aggregation_insights(_agg(), "how much on coffee", chat)
    assert out == "- You visit Blue Bottle often."
    assert "$42.50" in chat.prompts()[-1]


def test_a_visits_card_says_how_many_it_lists():
    from banko_ai.assistant.answers import render_visits_markdown
    recent = [{"expense_id": f"e{i}", "date": f"2026-09-{20 - i:02d}", "amount": 12.5, "merchant": "IKEA",
               "category": "Shopping", "description": ""} for i in range(10)]
    card = render_visits_markdown({"merchant": "IKEA", "visits": 12, "total": 150.0, "average": 12.5,
                                   "first": "2026-07-01", "last": "2026-09-20", "recent": recent})
    assert card.startswith("You went to **IKEA** 12 times since 2026-07-01 and spent **$150.00** there")
    assert "The last visit was on 2026-09-20 ($12.50)." in card
    assert card.count("| $12.50 | Shopping |") == 10 and card.endswith("The latest 10 of 12 visits.")


# What watsonx (openai/gpt-oss-120b) wrote: it copied the tool result's "cite" key into the marker.
WATSONX_CITE = f"$34.83 at Pizza Hut [cite:txn:{RECEIPT_EXPENSE}] and $69.59 at Domino's [ cite : txn: <{MEMORY_ID}> ]."


def test_a_marker_that_names_its_cite_key_is_still_a_citation():
    assert A.citation_ids(WATSONX_CITE) == [RECEIPT_EXPENSE, MEMORY_ID]
    assert A.strip_citations(WATSONX_CITE) == "$34.83 at Pizza Hut and $69.59 at Domino's."
    assert A.resolve_citations(WATSONX_CITE, {RECEIPT_EXPENSE}) == \
        f"$34.83 at Pizza Hut [txn:{RECEIPT_EXPENSE}] and $69.59 at Domino's."


def test_a_translation_wrapped_in_html_paragraphs_comes_back_as_markdown():
    """granite3.3:8b returned '<p>Gastaste **$959.67** ...</p>': Markdown inside
    an HTML block is not parsed, so the page showed the asterisks."""
    assert A.unwrap_html("<p>Gastaste **$959.67** en Restaurante.</p>") == "Gastaste **$959.67** en Restaurante."
    assert A.unwrap_html("<p>Uno</p>\n<p>Dos<br>tres</p>") == "Uno\n\nDos\ntres"
    assert A.unwrap_html("Plain **text**, where a < b and c > d.") == "Plain **text**, where a < b and c > d."
    # What granite also wrote: a closed first paragraph, then an HTML table (which the page renders).
    table = "<table>\n  <tr><td>Pizza Hut</td><td>$34.83</td></tr>\n</table>"
    assert A.unwrap_html(f"<p>Gastaste **$1**.</p>\n\n{table}") == f"Gastaste **$1**.\n\n{table}"
    assert A.unwrap_html(f"Gastaste **$1**.</p>\n\n{table}") == f"Gastaste **$1**.\n\n{table}"

    class Wraps:
        def complete(self, messages, *, temperature=0.3):
            return "<p>Hola **$1**</p>"

    assert A.translate("Hello **$1**", "Spanish", Wraps()) == "Hola **$1**"


def test_a_translation_keeps_category_names_as_the_page_shows_them():
    """granite translated "Groceries" as "Leche en el supermercado"; the page
    shows category names as written, so the answer does too."""
    prompt = A.translation_messages("You spent $1 on Groceries.", "Spanish")[0]["content"]
    assert "category name" in prompt and "merchant name" in prompt


_TOP = [{"date": "2026-09-01", "merchant": "Parkside Apartments", "amount": 1500.0, "category": "Housing"},
        {"date": "2026-09-14", "merchant": "Delta", "amount": 412.18, "category": "Travel"},
        {"date": "2026-09-09", "merchant": "Costco", "amount": 268.4, "category": "Groceries"}]


def test_the_largest_purchase_card_names_it_and_what_comes_next():
    text = A.render_aggregation_markdown(_agg(operation="largest", count=96, total=4210.55, rows=_TOP))
    assert text.splitlines()[0] == ("Your biggest purchase from 2026-09-01 to 2026-09-30 was **$1,500.00** at "
                                    "**Parkside Apartments** (Housing) on 2026-09-01, out of 96 purchases.")
    assert "The next largest:" in text
    assert "| 2026-09-14 | Delta | Travel | $412.18 |" in text and "| 2026-09-09 | Costco | Groceries | $268.40 |" in text
    assert "Parkside Apartments | Housing" not in text       # the headline is not repeated in the table


def test_the_card_for_several_purchases_lists_them_in_order():
    text = A.render_aggregation_markdown(_agg(operation="largest", count=96, total=4210.55, rows=_TOP, limit=3))
    assert text.splitlines()[0] == "Your 3 biggest purchases from 2026-09-01 to 2026-09-30, out of 96:"
    rows = [line for line in text.splitlines() if line.startswith("| 2026")]
    assert rows == ["| 2026-09-01 | Parkside Apartments | Housing | $1,500.00 |",
                    "| 2026-09-14 | Delta | Travel | $412.18 |", "| 2026-09-09 | Costco | Groceries | $268.40 |"]
    asked_five = A.render_aggregation_markdown(_agg(operation="largest", count=3, total=2180.58, rows=_TOP, limit=5))
    assert asked_five.splitlines()[0] == "Your 3 biggest purchases from 2026-09-01 to 2026-09-30, out of 3:"


def test_the_card_for_the_smallest_a_category_a_merchant_and_an_only_purchase():
    small = A.render_aggregation_markdown(_agg(operation="smallest", count=96, rows=list(reversed(_TOP))))
    assert small.startswith("Your smallest purchase from 2026-09-01 to 2026-09-30 was **$268.40** at **Costco**")
    assert "The next smallest:" in small
    one = [{"date": "2026-09-12", "merchant": "Nobu", "amount": 84.2, "category": "Restaurant"}]
    category = A.render_aggregation_markdown(_agg(operation="largest", category="Restaurant", count=1, rows=one))
    assert category == ("Your biggest Restaurant purchase from 2026-09-01 to 2026-09-30 was **$84.20** at **Nobu** "
                        "on 2026-09-12. It was the only one.")
    merchant = A.render_aggregation_markdown(_agg(operation="largest", merchant="Costco", count=4, rows=_TOP[2:]))
    assert merchant.startswith("Your biggest purchase at Costco from 2026-09-01 to 2026-09-30 was **$268.40** "
                               "(Groceries) on 2026-09-09, out of 4 purchases.")
    assert "No matching transactions" in A.render_aggregation_markdown(
        _agg(operation="largest", count=0, total=0.0, rows=[]))
