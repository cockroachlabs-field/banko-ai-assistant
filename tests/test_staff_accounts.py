"""The Banko account behind a signed-in member of staff: one per email,
made on the first visit, and let go when nobody has used it for a while."""

import os
import threading
import uuid

import pytest
from sqlalchemy import text

from banko_ai.db.engine import get_engine
from banko_ai.services import staff
from banko_ai.services.users import delete_users
from tests._iap import subject

DB = os.getenv("DATABASE_URL")
needs_db = pytest.mark.skipif(not DB, reason="DATABASE_URL not set")


def _address(name: str = "ana.b") -> str:
    return f"{name}-{uuid.uuid4().hex[:8]}@example.com"


def _id(email: str) -> str:
    return staff.user_id_for(subject(email))


def _ensure(email: str):
    """Sign in the Google account behind this address."""
    return staff.ensure(subject(email), email)


@pytest.fixture
def made():
    """Emails this test signed in with; their accounts are removed afterwards."""
    emails: list[str] = []
    yield emails
    delete_users([_id(e) for e in emails])
    staff.forget()


def test_a_google_account_always_names_the_same_user():
    assert staff.user_id_for("accounts.google.com:1234") == staff.user_id_for(" accounts.google.com:1234 ")
    assert staff.user_id_for("accounts.google.com:1234") != staff.user_id_for("accounts.google.com:1235")
    assert str(uuid.UUID(staff.user_id_for("accounts.google.com:1234"))) == staff.user_id_for("accounts.google.com:1234")


@pytest.mark.parametrize("email, username", [
    ("ana.b@example.com", "ana.b"),
    ("Ana_B-2@example.com", "ana_b-2"),
    ("ana+demo@example.com", "ana-demo"),                     # outside Banko's rule: [A-Za-z0-9._-]{1,32}
    ("renée.o'brien@example.com", "ren-e.o-brien"),
    ("a.very.long.local.part.that.goes.on.and.on@example.com", "a.very.long.local.part.that.goes"),
    ("load-tester@example.com", "staff-load-tester"),         # load- names belong to the load simulator
    ("...@example.com", "staff"),
])
def test_the_username_is_the_local_part_within_bankos_rule(email, username):
    import re
    assert staff.username_for(email) == username
    assert re.fullmatch(r"[A-Za-z0-9._-]{1,32}", staff.username_for(email))


@needs_db
def test_the_first_visit_makes_the_account_and_the_next_finds_it(made):
    email = _address()
    made.append(email)
    user, created = _ensure(email)
    assert created is True and user["user_id"] == _id(email)
    assert user["username"] == staff.username_for(email) and user["spending_style"] == "balanced"
    again, created = staff.ensure(subject(email), email.upper())
    assert created is False and again["user_id"] == user["user_id"]
    with get_engine().connect() as c:
        assert c.execute(text("SELECT demo_user FROM users WHERE user_id = CAST(:u AS UUID)"),
                         {"u": user["user_id"]}).scalar() is True


@needs_db
def test_an_address_handed_to_someone_else_does_not_bring_its_account():
    """Ana leaves and her address goes to a new hire. The account belongs to
    Ana's Google account, which Google never hands on: the new hire starts
    with an account of their own, and Ana's receipts and conversations are
    not in it."""
    email = _address("handed.on")
    ana, new_hire = "accounts.google.com:1001", "accounts.google.com:2002"
    try:
        hers, made_hers = staff.ensure(ana, email)
        theirs, made_theirs = staff.ensure(new_hire, email)
        assert made_hers and made_theirs and hers["user_id"] != theirs["user_id"]
        assert theirs["username"] != hers["username"] and theirs["username"].startswith(hers["username"][:20])
        renamed = staff.ensure(ana, _address("ana.new.name"))[0]       # the same person under a new address
        assert renamed["user_id"] == hers["user_id"] and renamed["username"] == hers["username"]
    finally:
        delete_users([staff.user_id_for(ana), staff.user_id_for(new_hire)])
        staff.forget()


@needs_db
def test_a_username_someone_else_has_gets_a_suffix_and_never_their_account(made):
    """An employee whose address starts with maya must not land on the persona."""
    email = _address("taken")
    made.append(email)
    other = str(uuid.uuid4())
    with get_engine().begin() as c:
        c.execute(text("INSERT INTO users (user_id, username, spending_style, demo_user) "
                       "VALUES (CAST(:u AS UUID), :n, 'saver', true)"), {"u": other, "n": staff.username_for(email)})
    try:
        user, created = _ensure(email)
        assert created is True and user["user_id"] != other
        assert user["username"] == f"{staff.username_for(email)[:27]}-{user['user_id'][:4]}"
    finally:
        delete_users([other])


@needs_db
def test_a_second_taken_username_still_ends_in_an_account(made):
    """One suffix of four characters and then an error meant a person who
    could never sign in."""
    email = _address("twice")
    made.append(email)
    me, name = _id(email), staff.username_for(email)
    others = [str(uuid.uuid4()), str(uuid.uuid4())]
    with get_engine().begin() as c:
        for other, taken in zip(others, (name.upper(), f"{name[:27]}-{me[:4]}")):   # the first in another case
            c.execute(text("INSERT INTO users (user_id, username, spending_style, demo_user) "
                           "VALUES (CAST(:u AS UUID), :n, 'saver', true)"), {"u": other, "n": taken})
    try:
        user, created = _ensure(email)
        assert created is True and user["user_id"] == me
        assert user["username"] not in (name, f"{name[:27]}-{me[:4]}") and user["username"].startswith(name[:20])
        import re
        assert re.fullmatch(r"[A-Za-z0-9._-]{1,32}", user["username"])
    finally:
        delete_users(others)


@needs_db
def test_two_first_requests_at_once_make_one_account(made):
    email = _address()
    made.append(email)
    results, go = [], threading.Barrier(6)

    def visit():
        go.wait()
        results.append(_ensure(email))

    threads = [threading.Thread(target=visit) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert len(results) == 6 and len({user["user_id"] for user, _ in results}) == 1
    assert [created for _, created in results].count(True) == 1     # one of them seeds, the rest find it
    with get_engine().connect() as c:
        assert c.execute(text("SELECT count(*) FROM users WHERE user_id = CAST(:u AS UUID)"),
                         {"u": _id(email)}).scalar() == 1


@needs_db
def test_an_account_that_went_missing_is_made_again(made):
    """Expired, or the database rebuilt: the next visit starts it over."""
    email = _address()
    made.append(email)
    user, _ = _ensure(email)
    delete_users([user["user_id"]])                            # every deletion goes through here
    again, created = _ensure(email)
    assert created is True and again["user_id"] == user["user_id"]


@needs_db
def test_an_account_is_remembered_so_signing_in_needs_no_database(made, monkeypatch):
    """A region outage on the chaos stage is the demo. Whoever is already
    signed in stays signed in through it: once an account has been seen,
    nothing about signing in reads the database again."""
    email = _address()
    made.append(email)
    user, _ = _ensure(email)

    def down(*args, **kwargs):
        raise ConnectionError("connection refused")

    monkeypatch.setattr(staff, "get_engine", down)
    monkeypatch.setattr(staff, "run_transaction", down)
    assert _ensure(email) == (user, False)
    stranger = _address("stranger")
    with pytest.raises(ConnectionError):                       # someone this process has not seen: no guess
        _ensure(stranger)


def _seed_record(user_id: str, through: str | None) -> None:
    with get_engine().begin() as c:
        c.execute(text("UPSERT INTO banko_meta (key, value, updated_at) VALUES (:k, :v, now())"),
                  {"k": f"seed:{user_id}", "v": f'{{"start": "2026-01-01", "through": {through}, "style": "balanced"}}'})


@needs_db
def test_an_account_whose_history_never_finished_is_started_again(made):
    """The process stopped between making the account and seeding it, or in
    the middle of seeding: nobody else would ever finish it, and a member of
    staff cannot start over with a new username."""
    email = _address()
    made.append(email)
    user, start = _ensure(email)
    assert start is True
    staff.forget()                                             # a new process
    assert _ensure(email)[1] is True                      # never seeded
    _seed_record(user["user_id"], "null")
    staff.forget()
    assert _ensure(email)[1] is True                      # started, not finished
    _seed_record(user["user_id"], '"2026-11-20"')
    staff.forget()
    assert _ensure(email)[1] is False                     # finished: nothing to do
    assert _ensure(email)[1] is False


_SPARED = "SELECT count(*) FROM users WHERE user_id = ANY(CAST(:p AS UUID[])) OR username LIKE 'load-%'"


def _personas() -> list[str]:
    from banko_ai.config.catalog import get_catalog
    return [p.user_id for p in get_catalog().personas]


def _last_seen(user_id: str):
    with get_engine().connect() as c:
        return c.execute(text("SELECT updated_at FROM banko_meta WHERE key = :k"), {"k": f"seen:{user_id}"}).scalar()


@needs_db
def test_being_seen_is_written_once_a_day_not_on_every_request(made):
    email = _address()
    made.append(email)
    user, _ = _ensure(email)
    first = _last_seen(user["user_id"])
    assert first is not None                                   # written with the account, in one transaction
    assert staff.due(user["user_id"]) is False                 # and that was today's
    assert staff.seen(user["user_id"]) is False and _last_seen(user["user_id"]) == first
    staff.forget()                                             # a new process, or a new day
    assert staff.due(user["user_id"]) is True
    assert staff.seen(user["user_id"]) is True and _last_seen(user["user_id"]) >= first


def _backdate(user_id: str, *, seen: str | None = None, created: str | None = None) -> None:
    with get_engine().begin() as c:
        if seen:
            c.execute(text(f"UPSERT INTO banko_meta (key, value, updated_at) VALUES (:k, 'seen', now() - INTERVAL '{seen}')"),
                      {"k": f"seen:{user_id}"})
        if created:
            c.execute(text(f"UPDATE users SET created_at = now() - INTERVAL '{created}' WHERE user_id = CAST(:u AS UUID)"),
                      {"u": user_id})


def _exist(ids) -> set[str]:
    with get_engine().connect() as c:
        return {str(r[0]) for r in c.execute(text("SELECT user_id FROM users WHERE user_id = ANY(CAST(:ids AS UUID[]))"),
                                             {"ids": list(ids)})}


@needs_db
def test_an_account_nobody_used_for_a_while_is_removed_with_its_rows(made):
    idle, active, veteran = _address("idle"), _address("active"), _address("veteran")
    made.extend([idle, active, veteran])
    users = {e: _ensure(e)[0]["user_id"] for e in (idle, active, veteran)}
    _backdate(users[idle], seen="32 days", created="60 days")
    _backdate(users[veteran], seen="1 day", created="400 days")         # here from the start, and here yesterday
    assert staff.expire_idle(30) >= 1
    assert _exist(users.values()) == {users[active], users[veteran]}
    assert _last_seen(users[idle]) is None
    assert staff.expire_idle(0) == 0                                      # 0 keeps every account


@needs_db
def test_the_idle_clock_never_removes_an_account_used_within_the_days_it_names(made):
    """A visit is written down once a day, so the note can be up to a day
    older than the last use. The day is given back when counting."""
    email = _address("yesterday")
    made.append(email)
    user_id = _ensure(email)[0]["user_id"]
    _backdate(user_id, seen="47 hours")              # used some time in the day after that note
    assert staff.expire_idle(1) == 0 and _exist([user_id]) == {user_id}
    _backdate(user_id, seen="49 hours")
    assert staff.expire_idle(1) >= 1 and _exist([user_id]) == set()


@needs_db
def test_only_accounts_this_sign_in_made_are_ever_removed():
    """Turned on for a Banko that has been running: the usernames people made
    at the sign-in page, the users banko-ai seed made, the personas and the
    load simulator's customers are not this setting's to remove. Each of them
    here is old, and two even carry a note of a visit long ago."""
    from banko_ai.config.catalog import get_catalog
    persona = get_catalog().personas[0].user_id
    signed_up, seeded, load = str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())
    with get_engine().begin() as c:
        for user, name in ((signed_up, f"t-{signed_up[:8]}"), (seeded, f"demo-{seeded[:6]}"), (load, f"load-00001-{load[:4]}")):
            c.execute(text("INSERT INTO users (user_id, username, spending_style, demo_user, created_at) "
                           "VALUES (CAST(:u AS UUID), :n, 'saver', true, now() - INTERVAL '90 days')"),
                      {"u": user, "n": name})
        had_persona = c.execute(text("SELECT count(*) FROM users WHERE user_id = CAST(:u AS UUID)"), {"u": persona}).scalar()
    _backdate(load, seen="90 days")
    _backdate(persona, seen="90 days")
    try:
        staff.expire_idle(30)
        assert _exist([signed_up, seeded, load]) == {signed_up, seeded, load}
        assert len(_exist([persona])) == had_persona
    finally:
        delete_users([signed_up, seeded, load])
        with get_engine().begin() as c:
            c.execute(text("DELETE FROM banko_meta WHERE key = :k"), {"k": f"seen:{persona}"})


@needs_db
def test_the_age_limit_for_demo_users_leaves_staff_accounts_to_the_idle_clock(made, monkeypatch):
    """DEMO_USER_TTL_DAYS counts from creation. Someone here every day for a
    year would lose their account to it."""
    from banko_ai.config.settings import reset_config
    from banko_ai.utils.migration import DatabaseMigration
    email = _address("regular")
    made.append(email)
    user_id = _ensure(email)[0]["user_id"]
    _backdate(user_id, seen="1 day", created="400 days")
    monkeypatch.setenv("DEMO_USER_TTL_DAYS", "30")
    reset_config()
    DatabaseMigration(DB).expire_demo_users()
    assert _exist([user_id]) == {user_id}


def test_while_the_database_is_away_it_is_asked_about_one_newcomer_not_every_request(monkeypatch):
    """Each read of an account waits for a connection that is not coming. A
    page that polls once a second, from everyone this process has not seen
    yet, would take every worker thread, the health check's included."""
    asked = []

    def down(*args, **kwargs):
        asked.append(1)
        raise ConnectionError("connection refused")

    monkeypatch.setattr(staff, "get_engine", down)
    try:
        for n in range(5):
            with pytest.raises(Exception):
                _ensure(_address(f"newcomer{n}"))
        assert len(asked) == 1
        staff._down_until = 0.0                               # the pause is over
        with pytest.raises(ConnectionError):
            _ensure(_address("later"))
        assert len(asked) == 2
    finally:
        staff.forget()


@needs_db
def test_an_account_another_process_made_a_moment_ago_is_found_not_made_twice(made, monkeypatch):
    """Two Banko processes, one first visit each: the second insert meets the
    first one's row."""
    email = _address()
    made.append(email)
    elsewhere, _ = _ensure(email)                 # the other process made it
    staff.forget()
    real_find, looks = staff._find, []

    def not_yet_there(user_id, url):
        looks.append(1)
        return None if len(looks) == 1 else real_find(user_id, url)

    monkeypatch.setattr(staff, "_find", not_yet_there)
    here, start = _ensure(email)
    assert here["user_id"] == elsewhere["user_id"] and here["username"] == elsewhere["username"]
    with get_engine().connect() as c:
        assert c.execute(text("SELECT count(*) FROM users WHERE user_id = CAST(:u AS UUID)"),
                         {"u": here["user_id"]}).scalar() == 1


@pytest.mark.parametrize("email", ["load@example.com", "load-@example.com", ".load@example.com", "LOAD@example.com",
                                   "load.balancer@example.com", "loader@example.com"])
def test_no_name_tried_for_an_account_is_one_of_the_load_simulators(email):
    """load- names are the simulator's: it moves their money, and Clear load
    customers deletes them. "load" with a piece of the id after it is one."""
    import re
    for name in staff._usernames(email, "4dc2d22b-7599-5c3a-9d1e-0f6b8a7c5e41"):
        assert not name.lower().startswith("load-"), name
        assert re.fullmatch(r"[A-Za-z0-9._-]{1,32}", name), name
