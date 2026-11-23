"""Deterministic SQL aggregation for spending questions.

The LLM narrates these numbers; it never computes them. That is the whole
point: every provider returns the same figure because the figure comes from
one SQL statement against the user's own rows.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date

from sqlalchemy import text

from .intent_classifier import AggregationIntent

# Questions about the largest or smallest purchases order the rows by amount.
SIZE_ORDER = {"largest": "DESC", "smallest": "ASC"}
# The most purchases one size answer lists. More can be asked for: the card
# then says how many it shows.
MOST_LISTED = 10
LATEST_LISTED = 10   # the purchases a total lists under its figure
BROKEN_DOWN = 5      # the merchants or categories a total is broken down into


def _to_pyformat(sql: str) -> str:
    """Convert :name placeholders to %(name)s format for psycopg's client-side binding.

    Word boundaries ensure :start and :start_date convert independently.
    Negative lookbehind (?<!:) ensures :: type casts are left alone.
    """
    return re.sub(r"(?<!:):(\w+)\b", r"%(\1)s", sql)


@dataclass(frozen=True)
class AggregationResult:
    operation: str
    category: str | None
    window_start: date
    window_end: date
    total: float
    count: int
    average: float
    first_date: date | None
    last_date: date | None
    rows: list[dict] = field(default_factory=list)
    merchant: str | None = None
    largest: list[dict] = field(default_factory=list)
    limit: int = 1
    # A total, broken down over every purchase in the window (never over rows):
    # by merchant unless one merchant was asked about, and by category when
    # no category was.
    top_merchants: list[dict] = field(default_factory=list)
    top_categories: list[dict] = field(default_factory=list)


_DETERMINER = re.compile(r"^(?:my|the|our)\s+")


def resolve_category(subject: str | None, database_url: str | None = None) -> str | None:
    """Map a question phrase to a category's label: the catalog first, by
    key, label or alias ("dining" and "dining out" are Restaurant, "gas" is
    Fuel), then a near match against the labels people spend under
    ("electronic", "entertainmant"). None means neither found one.

    No statement is run. The near match used to be against every label in
    the purchases table, read once per process: a scan of every user's
    purchases in every region, for whoever first asked about something that
    is no category ("how much did I spend at starbucks"). Purchases carry
    the catalog's labels, so the catalog has the same list. database_url is
    not used any more."""
    if not subject:
        return None
    from ..config.catalog import get_catalog
    catalog = get_catalog()
    for name in (subject, _DETERMINER.sub("", subject.strip().lower())):
        found = catalog.find(name)
        if found is not None:
            return found.label
    cats = catalog.spending_labels()
    subject_n = subject.lower().rstrip("s")
    for cat in cats:
        if cat.lower().rstrip("s") == subject_n:
            return cat
    close = difflib.get_close_matches(
        subject_n, [c.lower() for c in cats], n=1, cutoff=0.75)
    if close:
        return next(c for c in cats if c.lower() == close[0])
    return None


def _merchants_of(user_id: str, database_url: str) -> list[str]:
    from ..db.engine import get_engine, routed
    with get_engine(routed(database_url, user_id)).connect() as c:
        return [r[0] for r in c.execute(text(
            "SELECT DISTINCT merchant FROM expenses@idx_expenses_user_date "
            "WHERE user_id = CAST(:u AS UUID) AND merchant IS NOT NULL"),
            {"u": user_id}) if r[0]]


def resolve_merchant(subject: str | None, user_id: str, database_url: str) -> str | None:
    """Map a question phrase ("starbucks", "blue bottle coffee") to one of
    this user's merchants, or None. Exact match first, then containment for
    phrases of four letters or more, then a close fuzzy match."""
    if not subject:
        return None
    names = _merchants_of(user_id, database_url)
    want = subject.lower().replace("'", "").strip()
    flat = {n.lower().replace("'", ""): n for n in names}
    if want in flat:
        return flat[want]
    if len(want) >= 4:
        for low, name in flat.items():
            if want in low or low in want:
                return name
    close = difflib.get_close_matches(want, list(flat), n=1, cutoff=0.8)
    return flat[close[0]] if close else None


def spending_category(subject: str | None) -> str | None:
    """The label of the category a word names, when money is spent there:
    not Income, not Transfer, not Other."""
    from ..config.catalog import get_catalog
    found = get_catalog().find(subject) if subject else None
    return found.label if found is not None and found.grp in ("needs", "wants") and found.key != "other" else None


def name_words(name: str | None) -> list[str]:
    """A name's words as a question's are read: lowercase, no apostrophe, letters and digits."""
    return re.findall(r"[^\W_]+", (name or "").lower().replace("'", "").replace("’", ""))


def loose_subject(question: str, subject: str, user_id: str, database_url: str,
                  known: Iterable[str] | None = None) -> tuple[str | None, str | None]:
    """What the words a total could not place name: (category, None), (None,
    merchant), or (None, None), which is the planner's.

    They name something only whole and exactly: a category money is spent
    on, by its key, label or alias (not Income, not Transfer, not Other:
    "the other day" is no total on Other), or one of the user's merchants by
    its name. Never a near miss ("shipping" is not Shopping, "card" is not
    Hallmark Cards), and never a name beside other words ("including
    amazon", "savings target"): those ask for more than its total.

    One case is not the words alone. A merchant's name may hold a word of a
    total's own, which was taken for the total: "best buy total" leaves
    "best". So a merchant of two words or more named whole in the question
    counts when its name has every word left over, and it comes first:
    "total wine total" is the shop, not the wine.

    known: the user's merchants as the caller has them (any spelling of the
    names will do: only their words are compared). With it the user's
    purchases are read only when a merchant is the answer, for its name as
    it is stored."""
    left, asked = name_words(subject), f" {' '.join(name_words(question))} "
    names = None if known is not None else _merchants_of(user_id, database_url)

    def stored(match: list[str]) -> str | None:
        """The merchant with these words, as its purchases name it. A leading the or my may be missing."""
        for name in (names if names is not None else _merchants_of(user_id, database_url)):
            words = name_words(name)
            if words == match or (words[:1] in (["the"], ["my"]) and words[1:] == match):
                return name
        return None

    candidates = [name_words(n) for n in (known if known is not None else names or ())]
    whole = [n for n in candidates if len(n) > 1 and set(left) <= set(n) and f" {' '.join(n)} " in asked]
    if whole:
        return None, stored(max(whole, key=len))
    category = spending_category(subject)
    if category is not None:
        return category, None
    return None, stored(left) if left in candidates else None


def _where_and_params(intent: AggregationIntent, user_id: str,
                      category: str | None,
                      region: str | None,
                      merchant: str | None = None) -> tuple[str, dict]:
    """One WHERE builder for the query and its EXPLAIN so they can never
    drift apart. The region predicate prunes the scan to the user's home
    partition on REGIONAL BY ROW tables; single-region callers pass None
    and the SQL never mentions crdb_region."""
    where = ("WHERE user_id = :u AND expense_date >= :s "
             "AND expense_date < :e")
    params: dict = {"u": user_id, "s": intent.window_start,
                    "e": intent.window_end}
    if merchant:
        where += " AND lower(merchant) = lower(:m)"
        params["m"] = merchant
    elif category:
        where += " AND shopping_type = :c"
        params["c"] = category
    if region:
        where += " AND crdb_region = :r"
        params["r"] = region
    return where, params


def _by_amount_sql(where: str, order: str) -> str:
    """One statement for the rows and for their EXPLAIN, so the plan in the
    trace is the plan of the query that answered. Purchases of the same
    amount come latest first, then by id: the same question, the same card."""
    return f"""
        SELECT expense_date, merchant, expense_amount, shopping_type
        FROM expenses@idx_expenses_user_date {where}
        ORDER BY expense_amount {order}, expense_date DESC, expense_id
        LIMIT :n
    """


def _by_amount(conn, where: str, params: dict, order: str, n: int) -> list[dict]:
    return [{"date": str(r[0]), "merchant": r[1], "amount": float(r[2]), "category": r[3]}
            for r in conn.execute(text(_by_amount_sql(where, order)), {**params, "n": n}).fetchall()]


def _grouped(conn, column: str, where: str, params: dict, name: str | None = None) -> list[dict]:
    """The window's spending by merchant or by category, biggest first.
    column is one of this module's two names, never input. A merchant comes
    with its category when all its purchases share one, so nobody has to
    guess which merchant belongs to which category."""
    found = conn.execute(text(f"""
        SELECT {column}, SUM(expense_amount), COUNT(*),
               CASE WHEN count(DISTINCT shopping_type) = 1 THEN max(shopping_type) END
        FROM expenses@idx_expenses_user_date {where}
        GROUP BY {column}
        ORDER BY 2 DESC, {column}
        LIMIT {BROKEN_DOWN}
    """), params).fetchall()
    return [{name or column: r[0], "total": round(float(r[1]), 2), "count": int(r[2]),
             **({"category": r[3]} if column == "merchant" else {})} for r in found]


def _listed(intent: AggregationIntent) -> int:
    """How many rows a size question reads. One was asked for: the two after
    it come too, so the answer can show what the largest is the largest of."""
    return 3 if intent.limit == 1 else min(intent.limit, MOST_LISTED)


def run_aggregation(intent: AggregationIntent, user_id: str,
                    database_url: str,
                    region: str | None = None,
                    merchant: str | None = None,
                    top: int = 0) -> AggregationResult:
    """rows are the most recent purchases in the window, or, for a question
    about the largest or smallest, those purchases in order of amount. With
    top, largest also carries that many biggest purchases: recent rows say
    nothing about which purchase was the largest. A total also carries its
    breakdown by merchant and by category, summed by SQL over the whole
    window: what the latest rows add up to says nothing about the window."""
    from ..db.engine import get_engine, routed
    category = None if merchant else resolve_category(intent.subject, database_url)
    where, params = _where_and_params(intent, user_id, category, region, merchant)

    eng = get_engine(routed(database_url, user_id))
    with eng.connect() as c:
        agg = c.execute(text(f"""
            SELECT COALESCE(SUM(expense_amount), 0),
                   COUNT(*),
                   MIN(expense_date), MAX(expense_date)
            FROM expenses@idx_expenses_user_date {where}
        """), params).fetchone()
        top_merchants: list[dict] = []
        top_categories: list[dict] = []
        if intent.operation in SIZE_ORDER:
            rows = _by_amount(c, where, params, SIZE_ORDER[intent.operation], _listed(intent))
        else:
            # A category's purchases, listed: whoever asked lists them a page
            # at a time the way Show more goes on, so none are read here.
            rows = [] if intent.operation == "purchases" else [
                {"date": str(r[0]), "merchant": r[1], "amount": float(r[2])} for r in c.execute(text(f"""
                SELECT expense_date, merchant, expense_amount
                FROM expenses@idx_expenses_user_date {where}
                ORDER BY expense_date DESC, expense_id
                LIMIT {LATEST_LISTED}
            """), params).fetchall()]
            if not merchant:
                top_merchants = _grouped(c, "merchant", where, params)
            if not merchant and not category:
                top_categories = _grouped(c, "shopping_type", where, params, name="category")
        largest = _by_amount(c, where, params, "DESC", top) if top else []

    total = float(agg[0] or 0)
    count = int(agg[1] or 0)
    return AggregationResult(
        operation=intent.operation,
        category=category,
        window_start=intent.window_start,
        window_end=intent.window_end,
        total=round(total, 2),
        count=count,
        average=round(total / count, 2) if count else 0.0,
        first_date=agg[2],
        last_date=agg[3],
        rows=rows,
        merchant=merchant,
        largest=largest,
        limit=intent.limit,
        top_merchants=top_merchants,
        top_categories=top_categories,
    )


def explain_aggregation(intent: AggregationIntent, user_id: str,
                        database_url: str,
                        region: str | None = None,
                        merchant: str | None = None) -> str:
    """Run EXPLAIN ANALYZE of the aggregation query and return the plan as text.
    For the largest or smallest purchases that is the statement that ordered
    them, which is where the rows in the answer came from.

    CockroachDB does not accept bound placeholders in EXPLAIN under server-side binding,
    so this statement binds parameters client-side with psycopg.ClientCursor."""
    import psycopg

    category = None if merchant else resolve_category(intent.subject, database_url)
    where, params = _where_and_params(intent, user_id, category, region, merchant)

    # Convert :name placeholders to %(name)s for client-side binding
    if intent.operation in SIZE_ORDER:
        sql = "EXPLAIN ANALYZE " + _to_pyformat(_by_amount_sql(where, SIZE_ORDER[intent.operation]))
        params = {**params, "n": _listed(intent)}
    else:
        sql = f"""
            EXPLAIN ANALYZE
            SELECT COALESCE(SUM(expense_amount), 0),
                   COUNT(*),
                   MIN(expense_date), MAX(expense_date)
            FROM expenses@idx_expenses_user_date {_to_pyformat(where)}
        """

    from ..db.engine import get_engine, routed
    eng = get_engine(routed(database_url, user_id))
    with eng.connect() as c:
        # Get the raw psycopg connection for client-side binding
        raw_conn = c.connection.driver_connection
        with psycopg.ClientCursor(raw_conn) as cur:
            cur.execute(sql, params)
            plan_rows = cur.fetchall()

    return "\n".join(str(row[0]) for row in plan_rows)
