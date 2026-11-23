"""A statement about one user's purchases stays inside that user's rows.

Purchases are kept apart by user, and on a multi-region database by region:
idx_expenses_user_date leads with the user, as the vector index does with
its prefix column. The table used to carry two more indexes that held
every user's purchases, by merchant and by category. Left to pick,
CockroachDB read a category "since ever" or a merchant's visits through
those, for any user it had no statistics on yet: every user's coffee, to
answer for one. And a total about something that is no category read every
label in the table once per process.

Those two indexes are gone, and so is the vector index that had no user in
front of it (tests/test_schema_registry.py). Every statement on one user's
purchases names the index that leads with the user, so the next index
someone adds cannot do the same, and so the table is never scanned for
one user: with statistics that say a user holds most of a small table,
which the three personas of a new demo database do, CockroachDB scans it
for any statement that names no index. The fraud check's duplicate and
outlier lookups did, on every receipt.

These tests run what the assistant, the pages, the coach, a receipt's
checks and the 1.x search run, in both states (no statistics, and fresh
ones for one user who holds every row), and read the plan of every
statement on purchases they sent."""

import json
import os
import re
import time
import uuid
from datetime import date

import psycopg
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import make_url

DB = os.getenv("DATABASE_URL")
pytestmark = pytest.mark.skipif(not DB, reason="DATABASE_URL not set")
NAME = "banko_per_user_reads_test"
# What one user's purchases may be read through: the index that leads with the user, the vector index that
# does, and the primary key for the rows those found.
OWN = {"idx_expenses_user_date", "idx_expenses_user_embedding", "expenses_pkey"}


@pytest.fixture(scope="module")
def nobody_counted():
    """A database of its own where the purchases table has no statistics,
    and a diner with a month of purchases in it. Returns (url, user id):
    everything asked of it names the url, so nothing a process keeps about
    the suite's own database (the role requests connect as, for one) sends
    a read elsewhere."""
    from banko_ai.db.schema import ensure_schema
    from banko_ai.services import ledger
    from banko_ai.services.accounts import list_accounts
    from banko_ai.services.seeding import seed_user
    from banko_ai.utils.db_retry import normalize_db_url
    url = normalize_db_url(str(make_url(DB).set(database=NAME).render_as_string(hide_password=False)))
    admin = create_engine(DB, isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(text(f"DROP DATABASE IF EXISTS {NAME} CASCADE"))
        conn.execute(text(f"CREATE DATABASE {NAME}"))
    own = create_engine(url, isolation_level="AUTOCOMMIT")
    try:
        ensure_schema(url)
        member = str(uuid.uuid4())
        with own.connect() as conn:
            conn.execute(text("INSERT INTO users (user_id, username, spending_style, demo_user) "
                              "VALUES (CAST(:u AS UUID), :name, 'diner', true)"),
                         {"u": member, "name": f"t-{member[:8]}"})
        seed_user(member, months=1, today=date.today(), url=url)
        card = next(a for a in list_accounts(member, url=url)["accounts"] if a["kind"] == "credit")
        ledger.post_purchase(member, card["account_id"], amount="6.25", merchant="Blue Bottle Coffee",
                             category="Coffee", url=url)
        with own.connect() as conn:
            # Nobody collects them again, and the ones collected so far are gone.
            conn.execute(text("ALTER TABLE expenses SET (sql_stats_automatic_collection_enabled = false)"))
            conn.execute(text("ALTER TABLE expenses INJECT STATISTICS '[]'"))
            assert conn.execute(text("SELECT count(*) FROM [SHOW STATISTICS FOR TABLE expenses]")).scalar() == 0
        yield url, member
    finally:
        own.dispose()
        with admin.connect() as conn:
            conn.execute(text(f"DROP DATABASE IF EXISTS {NAME} CASCADE"))
        admin.dispose()


@pytest.fixture(params=["no statistics", "statistics for one user who holds every row"])
def either_way(request, nobody_counted):
    """The two states a plan has been seen to leave one user's rows in.
    With no statistics CockroachDB read a category through an index that
    held everyone's purchases. With statistics, on a small table one user
    holds most of, it scans the table for a statement that names no index."""
    url, member = nobody_counted
    own = create_engine(url, isolation_level="AUTOCOMMIT")
    with own.connect() as conn:                 # a card kept in the other state would be read back, and nothing asked
        for kept in ("search_cache", "answer_cache"):
            conn.execute(text(f"DELETE FROM {kept} WHERE user_id = CAST(:u AS UUID)"), {"u": member})
    if request.param == "no statistics":
        own.dispose()
        yield url, member
        return
    try:
        with own.connect() as conn:
            conn.execute(text("ANALYZE expenses"))
            assert conn.execute(text("SELECT count(*) FROM [SHOW STATISTICS FOR TABLE expenses]")).scalar() > 0
        yield url, member
    finally:
        with own.connect() as conn:
            conn.execute(text("ALTER TABLE expenses INJECT STATISTICS '[]'"))
        own.dispose()


# Statements on the purchases table that name no index, and why they need none.
BY_ITS_ID = re.compile(r"expense_id\s*(?:=|IN\b)")        # one purchase, or a few, by primary key
EVERY_USERS = {
    "banko_ai/services/platform.py": "the Platform page counts every user's purchases, in the background",
    "banko_ai/services/users.py": "deleting users: their purchases are named by a list of users",
}
# Functions that ask about every user's purchases and are reached by no route: each is named here.
NO_ONE_USER = {
    ("banko_ai/agents/fraud_agent.py", "scan_recent_expenses"): "the 1.x orchestrator's audit of everyone's recent purchases",
    ("banko_ai/vector_search/search.py", "_build_general_query"): "a 1.x search with no user: every route names the signed-in one",
}
# The code a signed-in user's pages, answers, nudges and receipts run, and the 1.x search behind /api/search.
# The 1.x provider classes' own search (banko_ai/ai_providers) is called by no route and is not held to this.
HELD = ("banko_ai/services", "banko_ai/assistant", "banko_ai/coach", "banko_ai/utils/aggregations.py",
        "banko_ai/agents", "banko_ai/vector_search")


def test_every_statement_on_one_users_purchases_names_the_index_that_leads_with_the_user():
    """The rule, wherever a statement is written: FROM expenses@idx_expenses_user_date
    (or the vector index that leads with the user). A statement that names
    none leaves the choice to the planner, and for a user it has no
    statistics on it has picked an index that holds everyone's purchases."""
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    files = [f for held in HELD for f in ([root / held] if held.endswith(".py") else sorted((root / held).rglob("*.py")))]
    assert len(files) > 20
    unnamed = []
    for path in files:
        name = str(path.relative_to(root))
        source = path.read_text()
        for found in re.finditer(r"\b(?:FROM|JOIN)\s+expenses\b(?!@)", source):
            statement = source[found.start():found.start() + 320]
            inside = re.findall(r"^\s*def (\w+)", source[:found.start()], re.M)
            if BY_ITS_ID.search(statement) or name in EVERY_USERS or (name, inside[-1] if inside else "") in NO_ONE_USER:
                continue
            unnamed.append(f"{name}:{source.count(chr(10), 0, found.start()) + 1}")
    assert unnamed == []
    for (name, function), why in NO_ONE_USER.items():        # each exception is still there to need one
        assert why and re.search(rf"^\s*def {function}\(", (root / name).read_text(), re.M), (name, function)


class _Sent:
    """Every read of the purchases table the app sends, with its parameters."""

    def __init__(self) -> None:
        self.reads: list[tuple[str, object]] = []

    def note(self, conn, cursor, statement, parameters, context, executemany) -> None:
        first = statement.lstrip().split(None, 1)[0].upper()
        if re.search(r"\bexpenses\b", statement) and first in ("SELECT", "WITH"):
            self.reads.append((statement, parameters))

    def __enter__(self) -> "_Sent":
        event.listen(Engine, "before_cursor_execute", self.note)
        return self

    def __exit__(self, *exc) -> None:
        event.remove(Engine, "before_cursor_execute", self.note)


def _scans(plan: str) -> list[tuple[str, str]]:
    """(index, spans) for every read of the purchases table in a plan."""
    found, index = [], None
    for line in plan.splitlines():
        line = line.strip(" │├└─")
        if line.startswith("•"):
            index = None
        elif line.startswith("table: expenses@"):
            index = line.split("@", 1)[1].strip()
            found.append((index, ""))
        elif line.startswith("spans:") and index is not None:
            found[-1] = (index, line.split(":", 1)[1].strip())
    return found


def _strays(url: str, reads: list[tuple[str, object]]) -> list[str]:
    """The statements whose plan leaves the user's rows, each with the index it read."""
    out, seen = [], set()
    plain = make_url(url).set(drivername="postgresql").render_as_string(hide_password=False)
    with psycopg.connect(plain, autocommit=True) as conn:
        for statement, parameters in reads:
            with psycopg.ClientCursor(conn) as cur:     # EXPLAIN takes no placeholders bound by the server
                cur.execute("EXPLAIN " + statement, parameters)
                plan = "\n".join(str(r[0]) for r in cur.fetchall())
            for index, spans in _scans(plan):
                if index not in OWN or "FULL SCAN" in spans:
                    line = f"{index} {spans}".strip() + " <- " + " ".join(statement.split())[:150]
                    if line not in seen:
                        seen.add(line)
                        out.append(line)
    return out


class _Chat:
    """A model for the turns that have one write under a card: it plans nothing and says little."""

    def complete(self, messages, *, temperature=0.3):
        return '{"steps": []}'

    def stream(self, messages, *, temperature=0.3):
        yield "- Steady."


QUESTIONS = (
    "restaurants",                                              # a category, since ever
    "my coffee expenses this year",                             # a category over many days
    "blue bottle coffee",                                       # one of the user's merchants
    "did I go to Blue Bottle Coffee this year?",
    "how much did I spend on restaurants this year?",
    "how much did I spend at blue bottle coffee this year?",    # no category: every label in the table was read
    "what was my biggest purchase this year?",
    "top 3 restaurant purchases this year",
    "how much did I spend in total this month?",
)


def test_what_the_assistant_reads_for_one_user_stays_in_that_users_rows(either_way):
    from banko_ai.assistant import background
    from banko_ai.assistant import graph as G
    from banko_ai.services import caches, threads
    url, member = either_way
    caches.forget_merchants()
    g = G.build_graph(InMemorySaver())
    with _Sent() as sent:
        for question in QUESTIONS:
            thread = threads.create_thread(member, url=url)["thread_id"]
            out = G.run_turn(member, thread, question, chat=_Chat(), graph=g, url=url)
            assert out["route"]["kind"] in ("aggregate", "visits"), question
            assert out["answer"], question
        background.drain()
    assert len(sent.reads) >= 12, len(sent.reads)               # the statements were caught
    assert _strays(url, sent.reads) == []
    assert not [s for s, _ in sent.reads if " ".join(s.split()).startswith("SELECT DISTINCT shopping_type")]


def test_what_the_pages_and_the_coach_read_for_one_user_stays_in_that_users_rows(either_way):
    from banko_ai.coach import insights as coach
    from banko_ai.coach import tools as coach_tools
    from banko_ai.services import budgets, caches, recurring, reports
    from banko_ai.services import transactions as txns
    from banko_ai.utils import aggregations
    url, member = either_way
    today = date.today()
    with _Sent() as sent:
        txns.list_transactions(member, limit=25, url=url)
        txns.list_transactions(member, limit=25, category="Restaurant", url=url)
        reports.dashboard(member, url=url)
        reports.cashflow(member, url=url)
        reports.spending_pace(member, url=url)
        budgets.budget_progress(member, url=url)
        recurring.detect_recurring(member, url=url)
        coach.get_monthly_summary(member, today.year, today.month, url)
        coach.get_spending_velocity(member, "Restaurant", 300.0, url)
        coach.get_top_merchants(member, url)
        coach.get_top_merchants(member, url, category="Restaurant", days=400)
        coach.detect_subscriptions(member, url)
        coach_tools.get_recent_transactions(member, url)
        coach_tools.get_recent_transactions(member, url, category="Restaurant", days=400)
        caches.merchant_names(member, uuid.uuid4().int, url)    # a state of the user's money nobody has seen
        aggregations.resolve_merchant("blue bottle", member, url)
    assert len(sent.reads) >= 14, len(sent.reads)
    assert _strays(url, sent.reads) == []


def test_a_name_no_category_carries_is_matched_without_reading_anyones_purchases(nobody_counted):
    """ "how much did I spend at starbucks" asks whether starbucks is a
    category first. The near match was against every label in the purchases
    table, every user's in every region, kept for the life of the process."""
    from banko_ai.utils.aggregations import resolve_category
    url, _ = nobody_counted
    with _Sent() as sent:
        assert resolve_category("starbucks", url) is None
        assert resolve_category("restaurants", url) == "Restaurant"
        assert resolve_category("electronic", url) == "Electronics"        # a label, but for its last letter
        assert resolve_category("entertainmant", url) == "Entertainment"   # a near miss of one
        assert resolve_category("zzzunknownzzz", url) is None
    assert sent.reads == []


class _NoModel:
    """The checks ask a model only to explain a purchase that tripped a signal. Here it has nothing to say."""

    def invoke(self, *args, **kwargs):
        raise RuntimeError("no model behind this check")


def test_what_a_receipts_checks_read_for_one_user_stays_in_that_users_rows(either_way):
    """Every receipt's purchase goes through the 1.x fraud and budget
    agents (services/receipts.py): the purchase by its id, the user's
    same-day charges and outlier amounts, the month's spending, and the
    purchases most like this one, by vector. Their statements named no
    index, and with statistics the duplicate and outlier lookups scanned
    the whole table."""
    from banko_ai.agents.budget_agent import BudgetAgent
    from banko_ai.agents.fraud_agent import FraudAgent
    from banko_ai.agents.llm_factory import get_embedding_model
    from banko_ai.db.engine import get_engine
    url, member = either_way
    with get_engine(url).connect() as conn:
        expense = str(conn.execute(text(
            "SELECT expense_id FROM expenses@idx_expenses_user_date WHERE user_id = CAST(:u AS UUID) "
            "ORDER BY expense_date DESC LIMIT 1"), {"u": member}).scalar())
    with _Sent() as sent:
        fraud = FraudAgent(region="local", llm=_NoModel(), database_url=url, embedding_model=get_embedding_model(),
                           fraud_threshold=0.7, duplicate_window_days=3)
        checked = fraud.analyze_expense(expense)
        budget = BudgetAgent(region="local", llm=_NoModel(), database_url=url, alert_threshold=0.8)
        status = budget.check_budget_status(user_id=member, monthly_budget=3000.0)
    assert checked.get("expense_id") == expense and "error" not in checked, checked
    assert status.get("user_id") == member, status
    assert len(sent.reads) >= 8, len(sent.reads)
    assert [s for s, _ in sent.reads if "<=>" in s]                # the purchases most like this one were asked for
    assert _strays(url, sent.reads) == []


def _old_enough_to_be_read_in_the_past(url: str) -> None:
    """The 1.x summary reads a few seconds back (follower_read_timestamp),
    and a read in the past cannot see a database made after the moment it
    reads at. This one was made seconds ago: wait until that moment is past
    its making."""
    plain = make_url(url).set(drivername="postgresql").render_as_string(hide_password=False)
    until = time.monotonic() + 60
    with psycopg.connect(plain, autocommit=True) as conn:
        while True:
            try:
                conn.execute("SELECT count(*) FROM expenses AS OF SYSTEM TIME follower_read_timestamp()").fetchone()
                return
            except psycopg.Error:
                if time.monotonic() > until:
                    raise
                time.sleep(0.5)


def test_what_the_1x_search_reads_for_one_user_stays_in_that_users_rows(either_way):
    """POST /api/search and GET /api/user-summary answer for the signed-in
    user through the 1.x search engine."""
    from banko_ai.vector_search.search import VectorSearchEngine
    url, member = either_way
    _old_enough_to_be_read_in_the_past(url)
    engine = VectorSearchEngine(url, None)
    with _Sent() as sent:
        found = engine.search_expenses("coffee", user_id=member, limit=5)
        engine.get_user_spending_summary(member)
    assert len(found) == 5 and {str(r.user_id) for r in found} == {member}
    assert len(sent.reads) >= 2 and [s for s, _ in sent.reads if "<=>" in s]
    assert _strays(url, sent.reads) == []


def test_the_fraud_agents_search_tools_search_one_users_purchases_or_none(nobody_counted):
    """Both tools took a user or did without, and without one they searched
    every user's purchases. The fraud check always names the user, and no
    route asks for everyone's: a search with no user is declined, and
    nothing is read."""
    from banko_ai.agents.llm_factory import get_embedding_model
    from banko_ai.agents.tools.search_tools import create_search_tools
    url, member = nobody_counted
    tools = {t.name: t.func for t in create_search_tools(url, get_embedding_model())}
    with _Sent() as sent:
        for declined in (tools["vector_search_expenses"]("coffee"), tools["sql_search_expenses"]({"category": "Coffee"})):
            out = json.loads(declined)
            assert out["success"] is False and "user" in out["error"], out
    assert sent.reads == []
    with _Sent() as sent:
        alike = json.loads(tools["vector_search_expenses"]("coffee", member, 3))
        listed = json.loads(tools["sql_search_expenses"]({"user_id": member, "category": "Coffee"}, 5))
    assert alike["success"] and alike["count"] == 3 and {r["user_id"] for r in alike["results"]} == {member}
    assert listed["success"] and listed["count"] >= 1 and {r["user_id"] for r in listed["results"]} == {member}
    assert len(sent.reads) == 2 and _strays(url, sent.reads) == []
