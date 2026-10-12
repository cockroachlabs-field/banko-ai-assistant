"""The transaction list (ledger plus legacy expenses), semantic search over
the per-user vector index, and one transaction's detail."""

from __future__ import annotations

import base64
from datetime import date, datetime, timedelta, timezone
from typing import Any

from sqlalchemy import text

from ..config.catalog import get_catalog
from ..config.settings import get_config
from ..db.regions import regional_tables_ready
from ..db.rls import request_url
from ._sql import jsonable, read_session, valid_uuid

MAX_LIMIT = 200
MAX_K = 500

# Ledger entries, plus expenses that never got a ledger entry (receipts and
# imports from before the ledger existed), as one ordered stream.
_STREAM = """
    SELECT l.entry_id AS id, l.posted_at, l.amount, l.kind, l.description, l.account_id,
           e.merchant, e.shopping_type AS category, l.expense_id
    FROM ledger_entries AS l LEFT JOIN expenses AS e ON e.expense_id = l.expense_id
    WHERE l.user_id = CAST(:u AS UUID)
    UNION ALL
    SELECT e.expense_id, e.expense_date::TIMESTAMPTZ, -e.expense_amount, 'purchase', e.description, NULL,
           e.merchant, e.shopping_type, e.expense_id
    FROM expenses AS e
    WHERE e.user_id = CAST(:u AS UUID)
      AND NOT EXISTS (SELECT 1 FROM ledger_entries AS l2
                      WHERE l2.expense_id = e.expense_id AND l2.user_id = CAST(:u AS UUID))
"""


def _encode_cursor(posted_at: datetime, ident: str) -> str:
    return base64.urlsafe_b64encode(f"{posted_at.isoformat()}|{ident}".encode()).decode()


def _decode_cursor(cursor: str | None) -> tuple[datetime, str] | None:
    if not cursor:
        return None
    try:
        stamp, ident = base64.urlsafe_b64decode(cursor.encode()).decode().split("|")
        if not valid_uuid(ident):
            return None
        return datetime.fromisoformat(stamp), ident
    except (ValueError, UnicodeDecodeError):
        return None


def _category_names(category: str) -> list[str]:
    """Every lowercase spelling a stored shopping_type may use for this category."""
    cat = get_catalog().category_for(category)
    return sorted({cat.label.lower(), cat.key, *(a.lower() for a in cat.aliases)})


def _decorate(item: dict) -> dict:
    catalog = get_catalog()
    if item["kind"] == "income":
        cat = catalog.category_for("income")
    elif item["kind"] in ("transfer_in", "transfer_out", "card_payment"):
        cat = catalog.category_for("transfer")
    else:
        cat = catalog.category_for(item.get("category"))
    item.update(category_label=cat.label, category_icon=cat.icon, category_color=cat.color)
    return item


def list_transactions(user_id: str, *, limit: int = 50, cursor: str | None = None,
                      category: str | None = None, account_id: str | None = None, kind: str | None = None,
                      date_from: date | None = None, date_to: date | None = None,
                      min_amount: Any = None, max_amount: Any = None, url: str | None = None) -> dict:
    url = request_url(url)
    limit = max(1, min(int(limit), MAX_LIMIT))
    where, params = [], {"u": user_id, "limit": limit + 1}
    after = _decode_cursor(cursor)
    if after:
        where.append("(t.posted_at, t.id) < (:cts, CAST(:cid AS UUID))")
        params.update(cts=after[0], cid=after[1])
    if category:
        where.append("lower(t.category) = ANY(CAST(:cats AS STRING[]))")
        params["cats"] = _category_names(category)
    if account_id:
        if not valid_uuid(account_id):
            return {"items": [], "next_cursor": None}
        where.append("t.account_id = CAST(:acct AS UUID)")
        params["acct"] = account_id
    if kind:
        where.append("t.kind = :kind")
        params["kind"] = kind
    if date_from:
        where.append("t.posted_at >= :dfrom")
        params["dfrom"] = datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc)
    if date_to:
        where.append("t.posted_at < :dto")
        params["dto"] = datetime.combine(date_to + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
    if min_amount is not None:
        where.append("abs(t.amount) >= :minamt")
        params["minamt"] = min_amount
    if max_amount is not None:
        where.append("abs(t.amount) <= :maxamt")
        params["maxamt"] = max_amount
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    with read_session(user_id, url=url) as conn:
        rows = conn.execute(text(f"""
            SELECT t.id, t.posted_at, t.amount, t.kind, t.description, t.account_id, t.merchant, t.category, t.expense_id
            FROM ({_STREAM}) AS t {clause}
            ORDER BY t.posted_at DESC, t.id DESC LIMIT :limit
        """), params).fetchall()
    more = len(rows) > limit
    rows = rows[:limit]
    items = [_decorate({"id": str(r[0]), "posted_at": r[1], "amount": r[2], "kind": r[3],
                        "description": r[4], "account_id": str(r[5]) if r[5] else None,
                        "merchant": r[6], "category": r[7], "expense_id": str(r[8]) if r[8] else None})
             for r in rows]
    next_cursor = _encode_cursor(rows[-1][1], str(rows[-1][0])) if more and rows else None
    return jsonable({"items": items, "next_cursor": next_cursor})


def semantic_sql(filters: str = "") -> str:
    """The semantic search statement. The nearest neighbours come from the
    per-user vector index inside a MATERIALIZED CTE, and filters apply
    outside it: any extra predicate inside would make the optimizer skip the
    vector index (cockroachlabs/cockroach#10432)."""
    return f"""
        WITH nn AS MATERIALIZED (
            SELECT expense_id, embedding <=> :qvec AS distance
            -- Fresh users are not in table statistics yet, so without this hint the optimizer picks the date index
            FROM expenses@idx_expenses_user_embedding
            WHERE user_id = CAST(:u AS UUID)
            ORDER BY embedding <=> :qvec LIMIT :k
        )
        SELECT e.expense_id, e.expense_date, e.expense_amount, e.merchant, e.shopping_type, e.description, nn.distance
        FROM nn JOIN expenses AS e ON e.expense_id = nn.expense_id
        WHERE e.user_id = CAST(:u AS UUID) AND nn.distance IS NOT NULL {filters}
        ORDER BY nn.distance LIMIT :limit
    """


def _vector_text(vec) -> str:
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


def _qvec(text_query: str) -> str:
    from ..utils.embeddings import load_embedding_model
    return _vector_text(load_embedding_model().encode([text_query])[0])


def _semantic_rows(conn, user_id: str, qvec: str, *, limit: int, k: int, filters: str = "", extra=None):
    params = {"u": user_id, "qvec": qvec, "k": max(k, limit), "limit": min(limit, MAX_LIMIT), **(extra or {})}
    return conn.execute(text(semantic_sql(filters)), params).fetchall()


def semantic_search(user_id: str, q: str, *, limit: int = 20, k: int = 50, category: str | None = None,
                    date_from: date | None = None, date_to: date | None = None,
                    url: str | None = None) -> dict:
    url = request_url(url)
    q = (q or "").strip()[:500]
    if not q:
        return {"items": []}
    limit = max(1, min(int(limit), MAX_LIMIT))
    k = max(limit, min(int(k), MAX_K))
    filters, extra = [], {}
    if category:
        filters.append("AND lower(e.shopping_type) = ANY(CAST(:cats AS STRING[]))")
        extra["cats"] = _category_names(category)
    if date_from:
        filters.append("AND e.expense_date >= :dfrom")
        extra["dfrom"] = date_from
    if date_to:
        filters.append("AND e.expense_date <= :dto")
        extra["dto"] = date_to
    # Encode before opening the transaction: the model is CPU bound (and
    # loads on first use), and no connection should wait on it.
    qvec = _qvec(q)
    with read_session(user_id, url=url) as conn:
        rows = _semantic_rows(conn, user_id, qvec, limit=limit, k=k, filters=" ".join(filters), extra=extra)
    items = [_decorate({"id": str(r[0]), "expense_id": str(r[0]), "posted_at": r[1], "amount": -r[2],
                        "kind": "purchase", "merchant": r[3], "category": r[4], "description": r[5],
                        "distance": float(r[6]), "account_id": None}) for r in rows]
    return jsonable({"items": items})


def transaction_detail(user_id: str, txn_id: str, *, url: str | None = None) -> dict | None:
    """One transaction by ledger entry id or by expense id, for this user only."""
    # The regional check reads schema metadata, so it keeps the admin URL.
    admin_url = url or get_config().database_url
    url = request_url(url)
    if not valid_uuid(txn_id):
        return None
    regional = regional_tables_ready(admin_url)
    region_col = "crdb_region::STRING" if regional else "NULL::STRING"
    params = {"u": user_id, "t": txn_id}
    with read_session(user_id, url=url) as conn:
        entry = conn.execute(text("""
            SELECT entry_id, posted_at, amount, kind, description, account_id, expense_id FROM ledger_entries
            WHERE entry_id = CAST(:t AS UUID) AND user_id = CAST(:u AS UUID)
        """), params).fetchone()
        expense_id = str(entry[6]) if entry is not None and entry[6] else (None if entry is not None else txn_id)
        row = None
        if expense_id:
            row = conn.execute(text(f"""
                SELECT expense_id, expense_date, expense_amount, merchant, shopping_type, description,
                       payment_method, recurring, {region_col}, embedding::STRING
                FROM expenses WHERE expense_id = CAST(:x AS UUID) AND user_id = CAST(:u AS UUID)
            """), {"u": user_id, "x": expense_id}).fetchone()
        if row is None:
            if entry is None:
                return None
            return jsonable(_decorate({"id": str(entry[0]), "posted_at": entry[1], "amount": entry[2],
                                       "kind": entry[3], "description": entry[4],
                                       "account_id": str(entry[5]), "merchant": None, "category": None,
                                       "similar": [], "residency": None}))
        if entry is None:
            # Looked up by expense id: find the ledger entry that paid for it, if any.
            entry = conn.execute(text("""
                SELECT entry_id, posted_at, amount, kind, description, account_id, expense_id FROM ledger_entries
                WHERE expense_id = CAST(:x AS UUID) AND user_id = CAST(:u AS UUID)
                ORDER BY posted_at, entry_id LIMIT 1
            """), {"u": user_id, "x": expense_id}).fetchone()
        similar = []
        if row[9]:
            neighbours = _semantic_rows(conn, user_id, row[9], limit=6, k=6)
            similar = [{"expense_id": str(s[0]), "date": s[1], "amount": -s[2], "merchant": s[3],
                        "distance": float(s[6])} for s in neighbours if str(s[0]) != str(row[0])][:5]
        home = conn.execute(text("SELECT home_region FROM users WHERE user_id = CAST(:u AS UUID)"),
                            {"u": user_id}).scalar()
    return jsonable(_decorate({
        "id": str(entry[0]) if entry is not None else str(row[0]), "expense_id": str(row[0]),
        "posted_at": entry[1] if entry is not None else row[1],
        "amount": entry[2] if entry is not None else -row[2],
        "kind": entry[3] if entry is not None else "purchase",
        "merchant": row[3], "category": row[4], "description": row[5],
        "payment_method": row[6], "recurring": row[7],
        "account_id": str(entry[5]) if entry is not None else None,
        "similar": similar,
        "residency": {"stored_in": row[8], "home_region": home} if regional else None,
    }))
