"""Seeding writes deterministic history once, resumes cleanly, and catches up."""

import os
from datetime import date, timedelta

import pytest
from sqlalchemy import text

from banko_ai.config.catalog import get_catalog
from banko_ai.db.engine import get_engine
from banko_ai.services import seeding as S
from banko_ai.services.seed_plan import build_plan, history_start
from tests._money import drop_user, invariant_violations, make_user

pytestmark = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")
TODAY = date(2026, 10, 11)


@pytest.fixture
def saver():
    uid = make_user("saver")
    yield uid
    drop_user(uid)


def _count(table: str, user_id: str) -> int:
    with get_engine().connect() as c:
        return c.execute(text(f"SELECT count(*) FROM {table} WHERE user_id = CAST(:u AS UUID)"),
                         {"u": user_id}).scalar()


def _entry_ids(user_id: str) -> set[str]:
    with get_engine().connect() as c:
        return {str(r[0]) for r in c.execute(text("SELECT entry_id FROM ledger_entries "
                                                  "WHERE user_id = CAST(:u AS UUID)"), {"u": user_id})}


def test_seed_user_writes_a_whole_bank(saver):
    progress = []
    written = S.seed_user(saver, months=1, today=TODAY, on_progress=lambda *a: progress.append(a))
    assert written > 0
    assert _count("accounts", saver) == 3
    assert _count("ledger_entries", saver) == written
    assert _count("expenses", saver) > 0
    assert _count("user_budgets", saver) == 4
    assert _count("goals", saver) == 2
    assert invariant_violations(saver) == []
    with get_engine().connect() as c:
        missing = c.execute(text("SELECT count(*) FROM expenses WHERE user_id = CAST(:u AS UUID) "
                                 "AND embedding IS NULL"), {"u": saver}).scalar()
    assert missing == 0
    assert progress[-1][3] == "done" and progress[-1][1] == progress[-1][2] == written
    assert S.seeding_status(saver)["state"] == "done"


def test_second_run_the_same_day_writes_nothing(saver):
    S.seed_user(saver, months=1, today=TODAY)
    before = _entry_ids(saver)
    assert S.seed_user(saver, months=1, today=TODAY) == 0
    assert _entry_ids(saver) == before


def test_catch_up_matches_a_straight_seed(saver):
    earlier = TODAY - timedelta(days=5)
    S.seed_user(saver, months=1, today=earlier)
    S.seed_user(saver, months=1, today=TODAY)
    cat = get_catalog()
    plan = build_plan(saver, cat.profiles["saver"], cat, start=history_start(earlier, 1), end=TODAY)
    assert _entry_ids(saver) == {e.entry_id for e in plan.entries}
    assert invariant_violations(saver) == []


def test_interrupted_seed_resumes_without_duplicates(saver, monkeypatch):
    real = S.post_history
    calls = {"n": 0}

    def flaky(user_id, rows, *, url=None):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated crash")
        return real(user_id, rows, url=url)

    monkeypatch.setattr(S, "post_history", flaky)
    monkeypatch.setattr(S, "_batch_size", lambda: 20)
    with pytest.raises(RuntimeError):
        S.seed_user(saver, months=1, today=TODAY)
    assert S.seeding_status(saver)["state"] == "failed"
    monkeypatch.setattr(S, "post_history", real)
    S.seed_user(saver, months=1, today=TODAY)
    assert len(_entry_ids(saver)) == _count("ledger_entries", saver)
    assert invariant_violations(saver) == []


def test_reset_reseeds_the_same_history(saver):
    S.seed_user(saver, months=1, today=TODAY)
    first = _entry_ids(saver)
    S.reset_user(saver, today=TODAY, months=1)
    assert _entry_ids(saver) == first


def test_ensure_personas_is_idempotent_and_keeps_a_chosen_region():
    names = S.ensure_personas()
    assert {"maya", "sam", "riley"} <= set(names)
    with get_engine().begin() as c:
        c.execute(text("UPDATE users SET home_region = 'keep-me' "
                       "WHERE user_id = '00000000-0000-0000-0000-0000000000a3'"))
    try:
        S.ensure_personas()
        with get_engine().connect() as c:
            row = c.execute(text("SELECT home_region, display_name FROM users "
                                 "WHERE user_id = '00000000-0000-0000-0000-0000000000a3'")).fetchone()
        assert row == ("keep-me", "Riley Okafor")
    finally:
        with get_engine().begin() as c:
            c.execute(text("UPDATE users SET home_region = NULL "
                           "WHERE user_id = '00000000-0000-0000-0000-0000000000a3'"))


def test_categories_are_upserted_from_the_catalog():
    assert S.upsert_categories() >= 17
    with get_engine().connect() as c:
        assert c.execute(text("SELECT label FROM categories WHERE key = 'groceries'")).scalar() == "Groceries"


def test_generated_users_get_distinct_names_and_history():
    ids = S.seed_generated_users(2, months=1)
    try:
        with get_engine().connect() as c:
            names = {r[0] for r in c.execute(text("SELECT username FROM users "
                                                  "WHERE user_id = ANY(CAST(:ids AS UUID[]))"), {"ids": ids})}
        assert len(names) == 2 and all(n.startswith("demo-") for n in names)
        assert all(_count("ledger_entries", uid) > 0 for uid in ids)
    finally:
        for uid in ids:
            drop_user(uid)
