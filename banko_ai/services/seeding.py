"""Write personas and their history to CockroachDB, in the background.

Seeding builds the deterministic plan in memory (seed_plan), encodes every
purchase description in batches on the shared CPU model, and writes bounded
batches through ledger.post_history, so a repeated or interrupted run never
duplicates a row. A banko_meta row per user remembers the history window;
the next run writes only the days since."""

from __future__ import annotations

import json
import logging
import secrets
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone

from sqlalchemy import text

from ..config.catalog import get_catalog
from ..config.settings import get_config
from ..db.engine import get_engine, run_transaction
from ..db.regions import detect_database_regions, has_region_column, money_region, region_columns
from .ledger import HistoryRow, bump_data_version, create_goal, open_account, post_history, rebalance
from .seed_plan import SeedPlan, build_plan, history_start

log = logging.getLogger("banko.seeding")
ProgressFn = Callable[[str, int, int, str], None]

_status: dict[str, dict] = {}
_status_lock = threading.Lock()


def seeding_status(user_id: str) -> dict | None:
    with _status_lock:
        found = _status.get(str(user_id))
        return dict(found) if found else None


def _report(user_id: str, done: int, total: int, state: str, on_progress: ProgressFn | None) -> None:
    with _status_lock:
        _status[str(user_id)] = {"done": done, "total": total, "state": state}
    if on_progress:
        try:
            on_progress(str(user_id), done, total, state)
        except Exception:  # a broken progress sink must never stop seeding
            log.debug("seeding progress callback failed", exc_info=True)


def _batch_size() -> int:
    return max(1, get_config().seed_batch_size)


def _vector_text(vec) -> str:
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


def upsert_categories(url: str | None = None) -> int:
    cats = get_catalog().categories
    with get_engine(url).begin() as conn:
        conn.execute(text("""
            UPSERT INTO categories (key, label, icon, color, grp, sort)
            SELECT * FROM unnest(CAST(:keys AS STRING[]), CAST(:labels AS STRING[]), CAST(:icons AS STRING[]),
                                 CAST(:colors AS STRING[]), CAST(:grps AS STRING[]), CAST(:sorts AS INT8[]))
        """), {"keys": [c.key for c in cats], "labels": [c.label for c in cats],
               "icons": [c.icon for c in cats], "colors": [c.color for c in cats],
               "grps": [c.grp for c in cats], "sorts": [c.sort for c in cats]})
    return len(cats)


def ensure_personas(url: str | None = None) -> list[str]:
    """Upsert the catalog's personas: display fields refresh, a home region
    someone already chose is kept, and an ordinary user who took a persona's
    username keeps it (the persona waits)."""
    cfg = get_config()
    personas = get_catalog().personas
    regions = sorted(detect_database_regions(url or cfg.database_url))

    def region_for(i: int, wanted: str) -> str | None:
        if not regions:
            return None
        if wanted != "auto" and wanted in regions:
            return wanted
        return regions[i % len(regions)]

    params = {
        "ids": [p.user_id for p in personas],
        "names": [p.username for p in personas],
        "styles": [p.spending_style for p in personas],
        "regions": [region_for(i, p.home_region) for i, p in enumerate(personas)],
        "displays": [p.display_name for p in personas],
        "colors": [p.avatar_color for p in personas],
    }
    with get_engine(url).begin() as conn:
        conn.execute(text("""
            UPDATE users AS u SET display_name = t.display, avatar_color = t.color,
                                  home_region = COALESCE(u.home_region, t.region)
            FROM unnest(CAST(:ids AS UUID[]), CAST(:displays AS STRING[]), CAST(:colors AS STRING[]),
                        CAST(:regions AS STRING[])) AS t(id, display, color, region)
            WHERE u.user_id = t.id
        """), params)
        conn.execute(text("""
            INSERT INTO users (user_id, username, spending_style, home_region, display_name, avatar_color, demo_user)
            SELECT t.id, t.name, t.style, t.region, t.display, t.color, true
            FROM unnest(CAST(:ids AS UUID[]), CAST(:names AS STRING[]), CAST(:styles AS STRING[]),
                        CAST(:regions AS STRING[]), CAST(:displays AS STRING[]), CAST(:colors AS STRING[]))
                 AS t(id, name, style, region, display, color)
            ON CONFLICT DO NOTHING
        """), params)
        present = conn.execute(text("SELECT username FROM users WHERE user_id = ANY(CAST(:ids AS UUID[])) "
                                    "ORDER BY username"), params).fetchall()
    return [r[0] for r in present]


def _read_meta(user_id: str, url: str | None) -> dict | None:
    with get_engine(url).connect() as conn:
        raw = conn.execute(text("SELECT value FROM banko_meta WHERE key = :k"),
                           {"k": f"seed:{user_id}"}).scalar()
    return json.loads(raw) if raw else None


def _write_meta(user_id: str, value: dict, url: str | None) -> None:
    with get_engine(url).begin() as conn:
        conn.execute(text("UPSERT INTO banko_meta (key, value, updated_at) VALUES (:k, :v, now())"),
                     {"k": f"seed:{user_id}", "v": json.dumps(value)})


def _ensure_accounts(user_id: str, plan: SeedPlan, start: date, url: str | None) -> dict[str, str]:
    with get_engine(url).connect() as conn:
        existing = {r[0]: str(r[1]) for r in conn.execute(text(
            "SELECT name, account_id FROM accounts WHERE user_id = CAST(:u AS UUID)"), {"u": user_id})}
    ids: dict[str, str] = {}
    for acct in plan.accounts:
        ids[acct.key] = existing.get(acct.name) or open_account(
            user_id, kind=acct.kind, name=acct.name, institution=acct.institution, mask=acct.mask,
            opening_balance=acct.opening_balance, credit_limit=acct.credit_limit, apr=acct.apr,
            payment_method=acct.payment_method, opened_at=start, url=url)
    return ids


def _ensure_budgets(user_id: str, plan: SeedPlan, url: str | None) -> None:
    if not plan.budgets:
        return

    def fn(conn):
        region = money_region(conn, user_id, url)
        pinned = region if has_region_column(conn, "user_budgets") else None
        rcol, rval = region_columns(pinned)
        conn.execute(text(f"""
            INSERT INTO user_budgets ({rcol}user_id, category, amount, updated_at)
            SELECT {rval}CAST(:u AS UUID), t.cat, t.amt, now()
            FROM unnest(CAST(:cats AS STRING[]), CAST(:amts AS DECIMAL[])) AS t(cat, amt)
            ON CONFLICT (user_id, category) DO NOTHING
        """), {"u": user_id, "crdb_region": pinned, "cats": list(plan.budgets),
               "amts": list(plan.budgets.values())})
        bump_data_version(conn, user_id, region)

    run_transaction(fn, url=url, user_id=user_id)


def _ensure_goals(user_id: str, plan: SeedPlan, account_ids: dict[str, str], url: str | None) -> dict[str, str]:
    with get_engine(url).connect() as conn:
        existing = {r[0]: str(r[1]) for r in conn.execute(text(
            "SELECT name, goal_id FROM goals WHERE user_id = CAST(:u AS UUID)"), {"u": user_id})}
    ids: dict[str, str] = {}
    for goal in plan.goals:
        ids[goal.key] = existing.get(goal.name) or create_goal(
            user_id, name=goal.name, icon=goal.icon, target_amount=goal.target_amount,
            account_id=account_ids[goal.account_key], target_date=goal.target_date, url=url)
    return ids

LEASE_MINUTES = 15
LEASE_WAIT_SECONDS = 30


def _lease_key(user_id: str) -> str:
    return f"seed_lock:{user_id}"


def _claim_lease(user_id: str, owner: str, url: str | None) -> bool:
    """Take the per-user seeding lease, or take over one that has expired.
    The expiry is computed on the database clock, so app hosts with skewed
    clocks still agree on it."""

    def fn(conn) -> bool:
        conn.execute(text(f"""
            INSERT INTO banko_meta (key, value, updated_at)
            VALUES (:k, json_build_object('owner', CAST(:owner AS STRING),
                                          'expires_at', now() + INTERVAL '{LEASE_MINUTES} minutes')::STRING, now())
            ON CONFLICT (key) DO UPDATE SET value = excluded.value, updated_at = now()
            WHERE CAST(CAST(banko_meta.value AS JSONB)->>'expires_at' AS TIMESTAMPTZ) < now()
        """), {"k": _lease_key(user_id), "owner": owner})
        stored = conn.execute(text("SELECT value FROM banko_meta WHERE key = :k"),
                              {"k": _lease_key(user_id)}).scalar()
        return bool(stored) and json.loads(stored).get("owner") == owner

    return run_transaction(fn, url=url)


def _refresh_lease(user_id: str, owner: str, url: str | None) -> None:
    def fn(conn) -> None:
        conn.execute(text(f"""
            UPDATE banko_meta
            SET value = json_build_object('owner', CAST(:owner AS STRING),
                                          'expires_at', now() + INTERVAL '{LEASE_MINUTES} minutes')::STRING,
                updated_at = now()
            WHERE key = :k AND CAST(value AS JSONB)->>'owner' = :owner
        """), {"k": _lease_key(user_id), "owner": owner})

    run_transaction(fn, url=url)


def _release_lease(user_id: str, owner: str, url: str | None) -> None:
    def fn(conn) -> None:
        conn.execute(text("DELETE FROM banko_meta WHERE key = :k AND CAST(value AS JSONB)->>'owner' = :owner"),
                     {"k": _lease_key(user_id), "owner": owner})

    try:
        run_transaction(fn, url=url)
    except Exception:  # the lease expires on its own; never mask the real outcome
        log.warning("could not release the seeding lease for %s", user_id, exc_info=True)


def _lease_held(user_id: str, url: str | None) -> bool:
    with get_engine(url).connect() as conn:
        return conn.execute(text("SELECT 1 FROM banko_meta WHERE key = :k"),
                            {"k": _lease_key(user_id)}).fetchone() is not None


def _wait_for_lease(user_id: str, url: str | None) -> None:
    deadline = time.monotonic() + LEASE_WAIT_SECONDS
    while time.monotonic() < deadline and _lease_held(user_id, url):
        time.sleep(1.0)


def _wait_for_owner(user_id: str, today: date, on_progress: ProgressFn | None, url: str | None) -> int:
    """Someone else is seeding this user: give them a moment, never write."""
    _wait_for_lease(user_id, url)
    meta = _read_meta(user_id, url)
    current = bool(meta and meta.get("through") and date.fromisoformat(meta["through"]) >= today)
    _report(user_id, 0, 0, "done" if current else "running", on_progress)
    return 0


def _history_start(user_id: str, meta: dict | None, today: date, months: int, url: str | None) -> date:
    """The window start a previous run chose wins, so a resume on a later day
    regenerates the same entry ids instead of shifting the whole history."""
    if meta and meta.get("start"):
        return date.fromisoformat(meta["start"])
    with get_engine(url).connect() as conn:
        opened = conn.execute(text("SELECT min(opened_at) FROM accounts WHERE user_id = CAST(:u AS UUID)"),
                              {"u": user_id}).scalar()
    if opened is not None:
        return opened.date() if isinstance(opened, datetime) else opened
    return history_start(today, months)


def seed_user(user_id: str, style: str | None = None, *, months: int | None = None,
              today: date | None = None, on_progress: ProgressFn | None = None,
              url: str | None = None) -> int:
    """Seed or catch up one user's history through today. Returns the number
    of ledger entries written this run (0 when already current, or when
    another run holds this user's seeding lease)."""
    today = today or datetime.now(timezone.utc).date()
    style, profile = _profile_for(user_id, style, url)
    owner = secrets.token_hex(8)
    if not _claim_lease(user_id, owner, url):
        return _wait_for_owner(user_id, today, on_progress, url)
    try:
        return _seed_owned(user_id, style, profile, months or profile.months or get_config().seed_history_months,
                           today, on_progress, url, owner)
    finally:
        _release_lease(user_id, owner, url)


def _profile_for(user_id: str, style: str | None, url: str | None):
    with get_engine(url).connect() as conn:
        row = conn.execute(text("SELECT spending_style FROM users WHERE user_id = CAST(:u AS UUID)"),
                           {"u": user_id}).fetchone()
    if row is None:
        raise LookupError(f"no user {user_id}")
    style = style or row[0]
    profile = get_catalog().profiles.get(style)
    if profile is None:
        raise ValueError(f"no seed profile for spending style '{style}'")
    return style, profile


def _seed_owned(user_id: str, style: str, profile, months: int, today: date,
                on_progress: ProgressFn | None, url: str | None, owner: str) -> int:
    from ..utils.embeddings import load_embedding_model

    catalog = get_catalog()
    meta = _read_meta(user_id, url)
    start = _history_start(user_id, meta, today, months, url)
    # through None means a run started but never finished: redo the whole
    # window, and ON CONFLICT turns the batches already written into no-ops.
    through = date.fromisoformat(meta["through"]) if meta and meta.get("through") else None
    if through is not None and through >= today:
        _report(user_id, 0, 0, "done", on_progress)
        return 0

    done = total = 0
    try:
        plan = build_plan(user_id, profile, catalog, start=start, end=today)
        payment_by_key = {a.key: a.payment_method for a in plan.accounts}
        account_ids = _ensure_accounts(user_id, plan, start, url)
        if meta is None:
            _write_meta(user_id, {"start": start.isoformat(), "through": None, "style": style}, url)
        _ensure_budgets(user_id, plan, url)
        goal_ids = _ensure_goals(user_id, plan, account_ids, url)
        todo = plan.entries_after(through) if through else plan.entries
        total = len(todo)
        _report(user_id, 0, total, "running", on_progress)
        model = load_embedding_model(wait_seconds=600) if any(e.is_purchase for e in todo) else None
        size = _batch_size()
        for i in range(0, total, size):
            batch = todo[i:i + size]
            purchases = [e for e in batch if e.is_purchase]
            vectors = iter(model.encode([e.description for e in purchases])) if purchases else iter(())
            rows = []
            for e in batch:
                expense = None
                if e.is_purchase:
                    expense = {"expense_id": e.expense_id, "expense_date": e.posted_at.date(),
                               "expense_amount": -e.amount, "shopping_type": e.category,
                               "merchant": e.merchant, "payment_method": payment_by_key[e.account_key],
                               "recurring": e.recurring,
                               "tags": [e.category.lower(), e.merchant.lower().replace(" ", "_")],
                               "embedding": _vector_text(next(vectors))}
                rows.append(HistoryRow(entry_id=e.entry_id, account_id=account_ids[e.account_key],
                                       posted_at=e.posted_at, amount=e.amount, kind=e.kind,
                                       description=e.description, expense=expense,
                                       goal_id=goal_ids.get(e.goal_key) if e.goal_key else None,
                                       transfer_id=e.transfer_ref))
            post_history(user_id, rows, url=url)
            done += len(batch)
            _refresh_lease(user_id, owner, url)
            _report(user_id, done, total, "running", on_progress)
        rebalance(user_id, url=url)
        _write_meta(user_id, {"start": start.isoformat(), "through": today.isoformat(), "style": style}, url)
    except Exception:
        _report(user_id, done, total, "failed", on_progress)
        raise
    _report(user_id, done, total, "done", on_progress)
    return total


def _present_persona_ids(ids: list[str], url: str | None) -> set[str]:
    with get_engine(url).connect() as conn:
        return {str(r[0]) for r in conn.execute(text(
            "SELECT user_id FROM users WHERE user_id = ANY(CAST(:ids AS UUID[]))"), {"ids": ids})}


def seed_personas(url: str | None = None, on_progress: ProgressFn | None = None) -> dict[str, int]:
    ensure_personas(url)
    upsert_categories(url)
    personas = get_catalog().personas
    present = _present_persona_ids([p.user_id for p in personas], url)
    for p in personas:
        if p.user_id not in present:
            # An ordinary user took this persona's username before the
            # persona existed; seeding it would fail on every boot.
            log.warning("persona %s is not present (username taken by another user); skipping its seed",
                        p.username)
    results: dict[str, int] = {}
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, get_config().seed_parallelism),
                            thread_name_prefix="banko-seed") as pool:
        futures = {p.username: pool.submit(seed_user, p.user_id, on_progress=on_progress, url=url)
                   for p in personas if p.user_id in present}
        for name, future in futures.items():
            try:
                results[name] = future.result()
            except Exception as exc:
                log.warning("seeding %s failed: %s", name, exc)
                failures.append(name)
    if failures:
        raise RuntimeError(f"seeding failed for {', '.join(failures)}")
    return results


def seed_generated_users(count: int, *, months: int | None = None, url: str | None = None,
                         on_progress: ProgressFn | None = None) -> list[str]:
    cfg = get_config()
    styles = sorted(get_catalog().profiles)
    regions = sorted(detect_database_regions(url or cfg.database_url))
    if count <= 0:
        return []
    with get_engine(url).begin() as conn:
        rows = conn.execute(text("""
            INSERT INTO users (username, spending_style, home_region, demo_user)
            SELECT t.name, t.style, t.region, true
            FROM unnest(CAST(:names AS STRING[]), CAST(:styles AS STRING[]), CAST(:regions AS STRING[]))
                 AS t(name, style, region)
            RETURNING user_id
        """), {"names": [f"demo-{secrets.token_hex(3)}" for _ in range(count)],
               "styles": [styles[i % len(styles)] for i in range(count)],
               "regions": [regions[i % len(regions)] if regions else None for i in range(count)]}).fetchall()
    ids = [str(r[0]) for r in rows]
    with ThreadPoolExecutor(max_workers=max(1, cfg.seed_parallelism), thread_name_prefix="banko-seed") as pool:
        for future in [pool.submit(seed_user, uid, months=months, on_progress=on_progress, url=url) for uid in ids]:
            future.result()
    return ids


def _delete_money_rows(user_id: str, url: str | None) -> None:
    """The seed meta goes first, so a crash midway leaves a user who reseeds
    from scratch, not one whose meta claims history that is gone."""
    eng = get_engine(url)
    with eng.begin() as conn:
        conn.execute(text("DELETE FROM banko_meta WHERE key = :k"), {"k": f"seed:{user_id}"})
    for table in ("ledger_entries", "transfers", "goals", "accounts", "user_budgets",
                  "user_data_versions", "expenses"):
        while True:
            with eng.begin() as conn:
                deleted = conn.execute(text(f"DELETE FROM {table} WHERE user_id = CAST(:u AS UUID) LIMIT 1000"),
                                       {"u": user_id}).rowcount
            if deleted < 1000:
                break


def reset_user(user_id: str, *, url: str | None = None, on_progress: ProgressFn | None = None,
               today: date | None = None, months: int | None = None) -> int:
    """Delete the user's seed state and money rows, then reseed. Coach
    signals and nudges are left alone. The whole reset holds the user's
    seeding lease, so it never deletes rows out from under a running seed.
    When a seed holds it, the reset waits for that seed and tries once more;
    if the lease is still taken, it gives up without deleting anything
    (returns 0)."""
    today = today or datetime.now(timezone.utc).date()
    style, profile = _profile_for(user_id, None, url)
    owner = secrets.token_hex(8)
    if not _claim_lease(user_id, owner, url):
        _wait_for_lease(user_id, url)
        if not _claim_lease(user_id, owner, url):
            log.warning("reset of %s skipped: another seed or reset of this user still held the lease "
                        "after %ss; nothing was deleted", user_id, LEASE_WAIT_SECONDS)
            return 0
    try:
        _delete_money_rows(user_id, url)
        return _seed_owned(user_id, style, profile, months or profile.months or get_config().seed_history_months,
                           today, on_progress, url, owner)
    finally:
        _release_lease(user_id, owner, url)
