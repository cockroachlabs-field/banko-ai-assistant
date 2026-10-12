"""Seeding writes deterministic history once, resumes cleanly, and catches up."""

import os
from datetime import date, timedelta

import psycopg
import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

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


def test_seeding_writes_only_through_the_admin_url_even_with_rls_on(saver, monkeypatch):
    from banko_ai.config.settings import get_config
    from banko_ai.db import rls

    # A sentinel app URL: if seeding ever resolved through request_url, the
    # spies below would see it instead of the admin URL.
    monkeypatch.setattr(rls, "_app_url", "cockroachdb+psycopg://banko_app@localhost:1/x?sslmode=disable")
    seen = []
    real_open, real_goal = S.open_account, S.create_goal

    def spy_open(*args, **kwargs):
        seen.append(("open_account", kwargs.get("url")))
        return real_open(*args, **kwargs)

    def spy_goal(*args, **kwargs):
        seen.append(("create_goal", kwargs.get("url")))
        return real_goal(*args, **kwargs)

    monkeypatch.setattr(S, "open_account", spy_open)
    monkeypatch.setattr(S, "create_goal", spy_goal)
    assert S.seed_user(saver, months=1, today=TODAY) > 0
    admin = get_config().database_url
    assert {name for name, _ in seen} == {"open_account", "create_goal"}
    assert all(url == admin for _, url in seen), seen


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


def test_interrupted_seed_resumed_the_next_day_keeps_the_original_window(saver, monkeypatch):
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
    monkeypatch.setattr(S, "post_history", real)
    tomorrow = TODAY + timedelta(days=1)
    S.seed_user(saver, months=1, today=tomorrow)
    cat = get_catalog()
    plan = build_plan(saver, cat.profiles["saver"], cat, start=history_start(TODAY, 1), end=tomorrow)
    assert _entry_ids(saver) == {e.entry_id for e in plan.entries}
    assert invariant_violations(saver) == []


def test_missing_meta_with_existing_accounts_resumes_from_the_accounts_open_date(saver, monkeypatch):
    S.seed_user(saver, months=1, today=TODAY)
    first = _entry_ids(saver)
    with get_engine().begin() as c:
        c.execute(text("DELETE FROM banko_meta WHERE key = :k"), {"k": f"seed:{saver}"})
    S.seed_user(saver, months=1, today=TODAY + timedelta(days=3))
    assert first <= _entry_ids(saver)
    assert invariant_violations(saver) == []


def test_reset_while_a_seed_holds_the_lease_deletes_nothing(saver, monkeypatch):
    S.seed_user(saver, months=1, today=TODAY)
    before = _entry_ids(saver)
    assert S._claim_lease(saver, "running-seed", None)
    monkeypatch.setattr(S, "LEASE_WAIT_SECONDS", 1)
    try:
        assert S.reset_user(saver, today=TODAY, months=1) == 0
        assert _entry_ids(saver) == before
        assert _count("accounts", saver) == 3
        assert S._read_meta(saver, None)["through"] == TODAY.isoformat()
    finally:
        S._release_lease(saver, "running-seed", None)


def test_reset_goes_ahead_when_the_seed_finishes_during_the_wait(saver, monkeypatch):
    import threading

    S.seed_user(saver, months=1, today=TODAY)
    assert S._claim_lease(saver, "running-seed", None)
    monkeypatch.setattr(S, "LEASE_WAIT_SECONDS", 20)
    timer = threading.Timer(1.5, S._release_lease, args=(saver, "running-seed", None))
    timer.start()
    try:
        assert S.reset_user(saver, today=TODAY, months=1) > 0
    finally:
        timer.cancel()
        S._release_lease(saver, "running-seed", None)
    assert _count("accounts", saver) == 3
    assert invariant_violations(saver) == []
    assert not S._lease_held(saver, None)


def test_reset_holds_the_lease_until_the_reseed_is_done(saver, monkeypatch):
    S.seed_user(saver, months=1, today=TODAY)
    seen = []
    real_delete, real_seed = S._delete_money_rows, S._seed_owned

    def spy_delete(*args, **kwargs):
        seen.append(("delete", S._lease_held(saver, None)))
        return real_delete(*args, **kwargs)

    def spy_seed(*args, **kwargs):
        seen.append(("seed", S._lease_held(saver, None)))
        return real_seed(*args, **kwargs)

    monkeypatch.setattr(S, "_delete_money_rows", spy_delete)
    monkeypatch.setattr(S, "_seed_owned", spy_seed)
    assert S.reset_user(saver, today=TODAY, months=1) > 0
    assert seen == [("delete", True), ("seed", True)]
    assert not S._lease_held(saver, None)


def test_concurrent_seeds_of_one_user_never_duplicate(saver):
    import threading

    results, errors = [], []

    def run():
        try:
            results.append(S.seed_user(saver, months=1, today=TODAY))
        except Exception as exc:  # surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(120)
    assert errors == []
    assert _count("accounts", saver) == 3
    assert _count("goals", saver) == 2
    assert len(_entry_ids(saver)) == _count("ledger_entries", saver)
    assert sorted(results)[0] == 0 and sorted(results)[1] > 0
    assert invariant_violations(saver) == []
    with get_engine().connect() as c:
        lock = c.execute(text("SELECT count(*) FROM banko_meta WHERE key = :k"),
                         {"k": f"seed_lock:{saver}"}).scalar()
    assert lock == 0


def test_seed_personas_skips_a_persona_whose_username_is_taken(monkeypatch, caplog):
    import uuid

    from banko_ai.config.catalog import Persona

    squatter = make_user("diner")
    with get_engine().connect() as c:
        squatted_name = c.execute(text("SELECT username FROM users WHERE user_id = CAST(:u AS UUID)"),
                                  {"u": squatter}).scalar()
    ghost = Persona(user_id=str(uuid.uuid4()), username=squatted_name, display_name="Ghost",
                    avatar_color="#000000", spending_style="diner")
    real_catalog = get_catalog()
    catalog = real_catalog.model_copy(update={"personas": (*real_catalog.personas, ghost)})
    monkeypatch.setattr(S, "get_catalog", lambda: catalog)
    seeded = []
    monkeypatch.setattr(S, "seed_user", lambda uid, **kw: seeded.append(uid) or 0)
    try:
        with caplog.at_level("WARNING", logger="banko.seeding"):
            result = S.seed_personas()
        assert ghost.user_id not in seeded
        assert {p.user_id for p in real_catalog.personas} <= set(seeded)
        assert squatted_name not in result
        assert any(squatted_name in r.getMessage() for r in caplog.records)
    finally:
        drop_user(squatter)


def test_the_invariant_holds_after_every_batch(saver, monkeypatch):
    real = S.post_history
    calls = {"n": 0}
    after_batch = []

    def flaky(user_id, rows, *, url=None):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated crash")
        written = real(user_id, rows, url=url)
        after_batch.append(invariant_violations(saver))
        return written

    monkeypatch.setattr(S, "post_history", flaky)
    monkeypatch.setattr(S, "_batch_size", lambda: 20)
    with pytest.raises(RuntimeError):
        S.seed_user(saver, months=1, today=TODAY)
    assert after_batch == [[]]
    assert _count("ledger_entries", saver) == 20
    assert invariant_violations(saver) == []  # a failed seed leaves no drift behind


def test_a_transfer_during_a_seed_of_the_same_user_leaves_no_drift(saver, monkeypatch):
    import threading

    from banko_ai.services import ledger as L

    real = S.post_history
    first_batch, transfer_done = threading.Event(), threading.Event()
    calls = {"n": 0}

    def paced(user_id, rows, *, url=None):
        written = real(user_id, rows, url=url)
        calls["n"] += 1
        if calls["n"] == 1:
            first_batch.set()
            transfer_done.wait(30)
        return written

    monkeypatch.setattr(S, "post_history", paced)
    monkeypatch.setattr(S, "_batch_size", lambda: 20)
    errors = []

    def run():
        try:
            S.seed_user(saver, months=1, today=TODAY)
        except Exception as exc:  # surfaced below
            errors.append(exc)

    seeder = threading.Thread(target=run)
    seeder.start()
    try:
        assert first_batch.wait(60)
        with get_engine().connect() as c:
            kinds = dict(c.execute(text("SELECT kind, account_id::STRING FROM accounts "
                                        "WHERE user_id = CAST(:u AS UUID) AND kind IN ('checking', 'savings')"),
                                   {"u": saver}).fetchall())
        result = L.transfer(saver, kinds["checking"], kinds["savings"], "10.00", idempotency_key="mid-seed")
        assert result.status == "ok"
        assert invariant_violations(saver) == []
    finally:
        transfer_done.set()
        seeder.join(120)
    assert errors == []
    assert invariant_violations(saver) == []


def test_reset_never_moves_the_data_version_backwards(saver):
    from banko_ai.services.ledger import data_version

    S.seed_user(saver, months=1, today=TODAY)
    before = data_version(saver)
    S.reset_user(saver, today=TODAY, months=1)
    assert data_version(saver) > before


def test_ensure_personas_applies_a_changed_spending_style():
    S.ensure_personas()
    riley = "00000000-0000-0000-0000-0000000000a3"
    wanted = next(p.spending_style for p in get_catalog().personas if p.user_id == riley)
    other = next(s for s in sorted(get_catalog().profiles) if s != wanted)
    with get_engine().begin() as c:
        c.execute(text("UPDATE users SET spending_style = :s WHERE user_id = CAST(:u AS UUID)"),
                  {"s": other, "u": riley})
    try:
        S.ensure_personas()
        with get_engine().connect() as c:
            style = c.execute(text("SELECT spending_style FROM users WHERE user_id = CAST(:u AS UUID)"),
                              {"u": riley}).scalar()
        assert style == wanted
    finally:
        with get_engine().begin() as c:
            c.execute(text("UPDATE users SET spending_style = :s WHERE user_id = CAST(:u AS UUID)"),
                      {"s": wanted, "u": riley})


class _FlakyEngine:
    """The real engine, except the first transaction it opens fails with a
    40001, the way a write to a GLOBAL table can lose to a neighbor."""

    def __init__(self, real):
        self.real = real
        self.failed = 0

    def begin(self):
        if not self.failed:
            self.failed += 1
            raise OperationalError("BEGIN", {}, psycopg.errors.SerializationFailure("restart transaction"))
        return self.real.begin()

    def __getattr__(self, name):
        return getattr(self.real, name)


@pytest.fixture
def flaky(monkeypatch):
    from banko_ai.db import engine as E
    real_get = E.get_engine
    engines: dict = {}

    def get(url=None):
        real = real_get(url)
        return engines.setdefault(id(real), _FlakyEngine(real))

    monkeypatch.setattr(E, "get_engine", get)
    monkeypatch.setattr(S, "get_engine", get)
    monkeypatch.setattr(E.time, "sleep", lambda seconds: None)
    return engines


@pytest.mark.parametrize("step", ["ensure_personas", "upsert_categories", "write_meta",
                                  "delete_money_rows", "generated_users"])
def test_seeding_writes_survive_a_serialization_failure(saver, flaky, monkeypatch, step):
    if step == "ensure_personas":
        assert {"maya", "sam", "riley"} <= set(S.ensure_personas())
    elif step == "upsert_categories":
        assert S.upsert_categories() >= 17
    elif step == "write_meta":
        S._write_meta(saver, {"start": "2026-01-01", "through": None}, None)
        assert S._read_meta(saver, None) == {"start": "2026-01-01", "through": None}
    elif step == "delete_money_rows":
        S._delete_money_rows(saver, None)
    else:
        monkeypatch.setattr(S, "seed_user", lambda *a, **k: 0)
        ids = S.seed_generated_users(1)
        assert len(ids) == 1
        for uid in ids:
            drop_user(uid)
    assert sum(e.failed for e in flaky.values()) == 1
