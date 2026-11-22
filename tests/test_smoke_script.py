"""The release smoke's own logic: reading the assistant's event stream,
telling a Spanish answer from an English one, and failing a check when a
tool raised behind an answer."""

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "smoke.py"
_spec = importlib.util.spec_from_file_location("smoke", _SCRIPT)
smoke = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(smoke)


def test_the_event_stream_parses_into_events_with_their_data():
    lines = ["event: step", 'data: {"label": "Intent routing"}', "", "event: answer",
             'data: {"answer": "You spent $42.10", "route": "aggregate"}', "", "event: done", "data: {}"]
    assert smoke.parse_sse(lines) == [("step", {"label": "Intent routing"}),
                                      ("answer", {"answer": "You spent $42.10", "route": "aggregate"}),
                                      ("done", {})]


def test_spanish_is_told_apart_from_english():
    assert smoke.reads_spanish("Gastaste **$312.40** en restaurantes el mes pasado, en 9 compras.")
    assert not smoke.reads_spanish("You spent **$312.40** on restaurants last month, across 9 purchases.")


def test_the_generated_receipt_is_a_png():
    assert smoke.receipt_png("12.34")[:8] == b"\x89PNG\r\n\x1a\n"


def _answer(trace=(), search=None, route="plan", answer="Here are your late night orders.", cached=False):
    meta = {"trace": list(trace), **({"search": search} if search else {})}
    return {"answer": answer, "route": route, "meta": meta, "cached": cached}


# Trace steps as banko_ai/assistant/tools.py writes them; tests/test_assistant_api.py
# feeds the smoke the ones the API really sends.
_RAISED = {"kind": "tool", "label": "search_transactions", "detail": "failed; details are in the server log",
           "status": "error"}
_WORKED = {"kind": "tool", "label": "search_transactions", "detail": "10 rows", "status": "success"}
_EMPTY = {"kind": "tool", "label": "search_transactions", "detail": "0 rows", "status": "success"}
# Arguments the planner got wrong: its own mistake, not a tool that broke.
_DECLINED = {"kind": "tool", "label": "search_transactions", "detail": "bad arguments, skipped", "status": "warning"}


def test_a_tool_that_raised_is_read_from_the_answer_trace():
    assert smoke.failed_tools(_answer([_WORKED, _RAISED])) == ["search_transactions"]
    assert smoke.failed_tools(_answer([_RAISED, _RAISED])) == ["search_transactions"]  # named once
    assert smoke.failed_tools(_answer([_WORKED, _DECLINED])) == []
    assert smoke.failed_tools({"answer": "hi"}) == []


class _Stream:
    def __init__(self, data, event="answer"):
        self.lines = [f"event: {event}", "data: " + smoke.json.dumps(data), "", "event: done", "data: {}"]

    def raise_for_status(self):
        pass

    def iter_lines(self, decode_unicode=True):
        return iter(self.lines)


def _smoke_answering(monkeypatch, *answers):
    """A Smoke whose assistant replies with these answer events, in order."""
    s = smoke.Smoke("http://banko.test", "maya", None)
    queue = list(answers)
    monkeypatch.setattr(s.http, "post", lambda *a, **kw: _Stream(queue.pop(0)))
    monkeypatch.setattr(s, "thread", lambda: "t-1")
    return s


def test_an_answer_put_together_after_a_tool_raised_fails_the_check(monkeypatch):
    """The assistant still answers when a tool raises, without that tool's
    facts, so an answer alone proves nothing about the tool."""
    s = _smoke_answering(monkeypatch, _answer([_RAISED], answer="I don't have any information about that."))
    with pytest.raises(AssertionError, match="search_transactions failed"):
        s.ask("t-1", smoke.SEARCH_QUESTION)


def test_the_search_check_needs_a_search_to_have_run(monkeypatch):
    s = _smoke_answering(monkeypatch, _answer([], answer="You like coffee."))
    with pytest.raises(AssertionError, match="no search ran"):
        s.search_and_more()


def test_the_search_check_passes_on_one_page_and_on_two(monkeypatch):
    # The answer names its search only when there is a next page to ask for.
    one = _smoke_answering(monkeypatch, _answer([_WORKED]))
    assert "no second page" in one.search_and_more()
    more = _answer([], route="more", search={"q": "coffee", "next": None, "items": [{"expense_id": "e1"}]})
    two = _smoke_answering(monkeypatch, _answer([_WORKED], search={"q": "coffee", "next": "c1"}), more)
    assert two.search_and_more() == "answered, then 1 more from the database"


def test_the_search_check_fails_on_an_answer_from_the_cache(monkeypatch):
    """A cached answer ran no search, so it proves nothing about search. The
    receipt check posts a new purchase first, which retires earlier answers."""
    s = _smoke_answering(monkeypatch, _answer([], cached=True))
    with pytest.raises(AssertionError, match="answer cache"):
        s.search_and_more()


def test_the_search_check_fails_when_the_search_found_nothing(monkeypatch):
    # A search by meaning brings the nearest purchases, whatever they are: none means it did not look.
    s = _smoke_answering(monkeypatch, _answer([_EMPTY], answer="You have no such purchases."))
    with pytest.raises(AssertionError, match="found no purchases"):
        s.search_and_more()


def _card(text="You spent **$41.20** on Coffee since 2026-05-23 (9 transactions).", calls=0, route="aggregate",
          plan="• group (scalar)", search=None, cached=False):
    """A category's card as the assistant sends it: from SQL, with its plan and no model call."""
    answer = _answer([], route=route, answer=text, cached=cached, search=search)
    answer["meta"].update({"model_calls": calls, **({"explain": plan} if plan else {})})
    return answer


def test_a_category_in_three_wordings_must_be_one_card_from_sql(monkeypatch):
    """ "coffee", "my coffee expenses", "show me my coffee purchases": the
    database's answer, so the same one each time, with no model call."""
    ok = _smoke_answering(monkeypatch, _card(), _card(), _card())
    assert ok.named_alone() == "You spent **$41.20** on Coffee since 2026-05-23 (9 transactions). | one page"
    planned = _smoke_answering(monkeypatch, _card(), _card(route="plan", calls=3), _card())
    with pytest.raises(AssertionError, match="route 'plan'"):
        planned.named_alone()
    narrated = _smoke_answering(monkeypatch, _card(calls=1), _card(), _card())
    with pytest.raises(AssertionError, match="1 model call"):
        narrated.named_alone()
    replayed = _smoke_answering(monkeypatch, _card(cached=True), _card(), _card())
    with pytest.raises(AssertionError, match="from a cache"):
        replayed.named_alone()
    no_plan = _smoke_answering(monkeypatch, _card(), _card(plan=None), _card())
    with pytest.raises(AssertionError, match="no query plan"):
        no_plan.named_alone()
    other = _smoke_answering(monkeypatch, _card(), _card(), _card(text="You spent **$12.00** on Coffee since 2026-10-23."))
    with pytest.raises(AssertionError, match="another answer"):
        other.named_alone()


def test_a_categorys_list_goes_on_from_the_database(monkeypatch):
    listed = {"q": "", "category": "Coffee", "next": "c1"}
    more = _answer([], route="more", search={**listed, "next": None, "items": [{"expense_id": "e1"}, {"expense_id": "e2"}]})
    more["meta"]["model_calls"] = 0
    two = _smoke_answering(monkeypatch, _card(search=listed), _card(search=listed), _card(search=listed), more)
    assert two.named_alone().endswith("| then 2 more from the database")
    empty = _answer([], route="more", search={**listed, "next": None, "items": []})
    none = _smoke_answering(monkeypatch, _card(search=listed), _card(search=listed), _card(search=listed), empty)
    with pytest.raises(AssertionError, match="no purchases"):
        none.named_alone()


class _Json:
    def __init__(self, data):
        self.data = data

    def json(self):
        return self.data


def test_a_merchant_by_its_name_is_the_visits_card_again(monkeypatch):
    """ "did I go to Domino's?" and "Domino's purchases" are one answer from SQL."""
    card = "You went to **Domino's** 29 times since 2026-05-28 and spent **$1,434.62** there ($49.47 a visit)."
    visits = _answer([], route="visits", answer=card)
    visits["meta"]["model_calls"] = 0
    s = _smoke_answering(monkeypatch, visits, visits)
    monkeypatch.setattr(s, "api", lambda method, path, **kw: _Json({"items": [{"merchant": "Domino's"}] * 3}))
    assert s.visits().startswith("You went to **Domino's** 29 times")
    planned = _answer([], route="plan", answer="You like Domino's.")
    s = _smoke_answering(monkeypatch, visits, planned)
    monkeypatch.setattr(s, "api", lambda method, path, **kw: _Json({"items": [{"merchant": "Domino's"}] * 3}))
    with pytest.raises(AssertionError, match="route 'plan'"):
        s.visits()
    other = _answer([], route="visits", answer=card.replace("29 times", "3 times"))
    s = _smoke_answering(monkeypatch, visits, other)
    monkeypatch.setattr(s, "api", lambda method, path, **kw: _Json({"items": [{"merchant": "Domino's"}] * 3}))
    with pytest.raises(AssertionError, match="another answer"):
        s.visits()


def test_the_subject_check_wants_the_total_of_what_was_named(monkeypatch):
    """ "groceries total last month" answered with the total of everything
    is a failure, however well it reads, and so is any other total than the
    one "on groceries last month" got."""
    card = "You spent **$502.72** on Groceries from 2026-10-01 to 2026-10-31 (8 transactions)."
    kept = _answer([], route="aggregate", answer=card + "\n\n- Steady.")
    s = _smoke_answering(monkeypatch, kept, kept, kept)
    assert "on Groceries" in s.totals() and s.groceries == card
    assert "on Groceries" in s.subject_kept()
    s.groceries = card.replace("502.72", "552.80")         # asked outright, it was another figure
    with pytest.raises(AssertionError, match="not the total 'on groceries last month' got"):
        s.subject_kept()
    dropped = _answer([], route="aggregate", answer="You spent **$7,631.76** on all categories from 2026-10-01 to 2026-10-31.")
    with pytest.raises(AssertionError, match="not a total of groceries"):
        _smoke_answering(monkeypatch, dropped).subject_kept()
    planned = _answer([], route="plan", answer="Your groceries came to about $500.")
    with pytest.raises(AssertionError, match="route 'plan'"):
        _smoke_answering(monkeypatch, planned).subject_kept()


def _smoke_uploading(monkeypatch, *statuses):
    """A Smoke whose receipt uploads end with these statuses, in order; every question is answered."""
    s = smoke.Smoke("http://banko.test", "maya", None)
    queue, uploads = list(statuses), []

    def post(url, **kw):
        if url.endswith("/api/v1/receipts"):
            uploads.append(s.total)
            return _Stream({"status": queue.pop(0), "merchant": "BLUE BOTTLE COFFEE", "amount": s.total,
                            "thread_id": "t-1"}, event="receipt")
        return _Stream(_answer(route="receipt"))

    monkeypatch.setattr(s.http, "post", post)
    return s, uploads


def test_a_receipt_that_comes_back_a_duplicate_is_uploaded_again_with_a_new_total(monkeypatch):
    """A total an earlier run posted in the last few days posts nothing, and
    the checks after this one count on a new purchase having retired earlier
    runs' cached answers."""
    s, uploads = _smoke_uploading(monkeypatch, "duplicate", "posted")
    assert "posted" in s.receipt()
    assert len(uploads) == 2 and uploads[0] != uploads[1]


def test_receipts_that_only_come_back_duplicates_fail_the_check(monkeypatch):
    s, uploads = _smoke_uploading(monkeypatch, "duplicate", "duplicate", "duplicate")
    with pytest.raises(AssertionError, match="duplicate"):
        s.receipt()
    assert len(uploads) == 3
