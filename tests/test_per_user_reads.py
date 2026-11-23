"""A statement about one user's purchases stays inside that user's rows.

Purchases are kept apart by user, and on a multi-region database by region:
idx_expenses_user_date leads with the user, as the vector index does with
its prefix column. The table used to carry two more indexes that held
every user's purchases, by merchant and by category. Left to pick,
CockroachDB read a category "since ever" or a merchant's visits through
those, for any user it had no statistics on yet: every user's coffee, to
answer for one. And a total about something that is no category read every
label in the table once per process.

Those two indexes are gone (tests/test_schema_registry.py), and every
statement on one user's purchases names the index that leads with the
user, so the next index someone adds cannot do the same. These tests take
the statistics away, which is the state a new user is in, run what the
assistant, the pages and the coach run, and read the plan of every
statement on purchases they sent."""

import os
import re
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


# Statements on the purchases table that name no index, and why they need none.
BY_ITS_ID = re.compile(r"expense_id\s*(?:=|IN\b)")        # one purchase, or a few, by primary key
EVERY_USERS = {
    "banko_ai/services/platform.py": "the Platform page counts every user's purchases, in the background",
    "banko_ai/services/users.py": "deleting users: their purchases are named by a list of users",
}
# The code a signed-in user's pages, answers and nudges run. The 1.x agents, providers and search engine
# under banko_ai/agents, ai_providers and vector_search are not held to this.
HELD = ("banko_ai/services", "banko_ai/assistant", "banko_ai/coach", "banko_ai/utils/aggregations.py")


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
            if BY_ITS_ID.search(statement) or name in EVERY_USERS:
                continue
            unnamed.append(f"{name}:{source.count(chr(10), 0, found.start()) + 1}")
    assert unnamed == []


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


def test_what_the_assistant_reads_for_one_user_stays_in_that_users_rows(nobody_counted):
    from banko_ai.assistant import background
    from banko_ai.assistant import graph as G
    from banko_ai.services import caches, threads
    url, member = nobody_counted
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


def test_what_the_pages_and_the_coach_read_for_one_user_stays_in_that_users_rows(nobody_counted):
    from banko_ai.coach import insights as coach
    from banko_ai.coach import tools as coach_tools
    from banko_ai.services import budgets, caches, recurring, reports
    from banko_ai.services import transactions as txns
    from banko_ai.utils import aggregations
    url, member = nobody_counted
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
