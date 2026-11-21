"""The Banko account behind a signed-in member of staff.

Where Banko sits behind Google's Identity-Aware Proxy (web/iap.py), each
person is signed in as themselves. The account belongs to the Google account
the assertion names ("sub", the identifier Google keeps stable for a user):
its user id is derived from that, the same every time, so an address that
passes to someone else never brings an account with it. The username is the
local part of the address, brought within Banko's rule. Neither the
identifier nor the address is stored. The account is made on the first
visit, the way a sign-up makes one, and removed with its rows when nobody
has used it for a while.

An account this process has seen is remembered for as long as the process
lives, so signing in reads the database once per person. A region outage on
the chaos stage is the demo, and nobody is signed out by it. An account that
is deleted is forgotten here (services/users.delete_users), and one deleted
by another process is found missing by the page that reads its profile.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from datetime import date, datetime, timezone

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from ..config.catalog import get_catalog
from ..config.settings import get_config
from ..db.engine import get_engine, run_transaction
from .load_users import LOAD_PREFIX

log = logging.getLogger("banko.web")

# uuid5(NAMESPACE, subject) is that person's user id. Changing this value
# would hand every member of staff a new, empty account. A user id is a name,
# never a secret: nothing may be served on the strength of knowing one.
NAMESPACE = uuid.UUID("5d0c8a1e-6b1f-4c0e-9a57-3f4b9d2e7c61")
DEFAULT_STYLE = "balanced"
USERNAME_MAX = 32                     # the sign-up rule: [A-Za-z0-9._-]{1,32}
RETRY_SECONDS = 60.0                  # between two tries at a history that failed
PAUSE_SECONDS = 5.0                   # after an account could not be read, before the database is asked again
_NOT_ALLOWED = re.compile(r"[^A-Za-z0-9._-]+")
_COLUMNS = "user_id, username, spending_style, home_region, created_at, demo_user"

_lock = threading.Lock()              # guards the three below; never held across the database
_known: dict[str, dict] = {}          # user id -> account, once this process has seen it
_seen_on: dict[str, date] = {}        # user id -> the day its last visit was written down
_retried_at: dict[str, float] = {}    # user id -> when a failed history was last started again
_down_until = 0.0                     # no account is read before this (time.monotonic)


class OutOfReach(Exception):
    """An account could not be read a moment ago, and the database is not
    asked again yet."""


def _today() -> date:
    """Today in UTC, the day the note of a visit is dated by."""
    return datetime.now(timezone.utc).date()


def user_id_for(subject: str) -> str:
    """The user id of the Google account an assertion names in "sub"."""
    return str(uuid.uuid5(NAMESPACE, subject.strip()))


def username_for(email: str) -> str:
    """The email's local part within the sign-up rule. Names that start with
    load- belong to the load simulator, which deletes them."""
    local = email.strip().lower().split("@", 1)[0]
    name = _NOT_ALLOWED.sub("-", local).strip("-.")[:USERNAME_MAX].rstrip("-.")
    if not name:
        return "staff"
    if name.startswith(LOAD_PREFIX):
        name = f"staff-{name}"[:USERNAME_MAX].rstrip("-.")
    return name


def _usernames(email: str, user_id: str) -> list[str]:
    """The names to try, in order: the local part, then the local part with
    more and more of the user id. A name someone else holds (a persona
    called maya, a sign-up from before) is never theirs to share."""
    name, tail = username_for(email), uuid.UUID(user_id).hex
    names = [name] + [f"{name[:USERNAME_MAX - n - 1].rstrip('-.')}-{tail[:n]}" for n in (4, 8, 12)]
    # "load" with a piece of the id after it is a load- name, and those are
    # the simulator's: it moves their money and deletes them.
    return [f"staff-{n}"[:USERNAME_MAX].rstrip("-.") if n.startswith(LOAD_PREFIX) else n for n in names]


def _style() -> str:
    styles = sorted(get_catalog().profiles)
    return DEFAULT_STYLE if DEFAULT_STYLE in styles else styles[0]


def _row(row) -> dict:
    return {"user_id": str(row[0]), "username": row[1], "spending_style": row[2], "home_region": row[3],
            "created_at": row[4], "demo_user": row[5]}


def _find(user_id: str, url: str | None) -> tuple[dict, bool] | None:
    """The account and whether its history was ever finished, or None."""
    with get_engine(url).connect() as conn:
        row = conn.execute(text(f"""
            SELECT {_COLUMNS}, (SELECT value FROM banko_meta WHERE key = 'seed:' || u.user_id::STRING)
            FROM users AS u WHERE user_id = CAST(:u AS UUID)
        """), {"u": user_id}).fetchone()
    if row is None:
        return None
    try:
        finished = bool(row[6]) and bool(json.loads(row[6]).get("through"))
    except (ValueError, AttributeError):
        finished = False
    return _row(row), finished


def _insert(user_id: str, username: str, url: str | None) -> dict | None:
    """The new account, or None: someone holds that username (in any case:
    signing in by name does not tell Bob from bob), or the account exists
    already. The note of its first visit is written with it, which is what
    marks an account as one this sign-in made."""
    def fn(conn):
        row = conn.execute(text(f"""
            INSERT INTO users (user_id, username, spending_style, home_region, demo_user)
            SELECT CAST(:u AS UUID), :n, :s, NULL, true
            WHERE NOT EXISTS (SELECT 1 FROM users WHERE lower(username) = lower(:n))
            RETURNING {_COLUMNS}
        """), {"u": user_id, "n": username, "s": _style()}).fetchone()
        if row is not None:
            conn.execute(text("UPSERT INTO banko_meta (key, value, updated_at) VALUES (:k, 'seen', now())"),
                         {"k": f"seen:{user_id}"})
        return row
    try:
        row = run_transaction(fn, url=url)
    except IntegrityError:
        return None
    return _row(row) if row is not None else None


def _find_or_make(user_id: str, email: str, url: str | None) -> tuple[dict, bool, bool]:
    """(account, its history has to be started, this call made it)."""
    found = _find(user_id, url)
    if found is not None:
        return found[0], not found[1], False
    for username in _usernames(email, user_id):
        user = _insert(user_id, username, url)
        if user is not None:
            return user, True, True
        found = _find(user_id, url)        # made a moment ago, by another request or another process
        if found is not None:
            return found[0], not found[1], False
    raise RuntimeError("no username is free for this account")


def ensure(subject: str, email: str, *, url: str | None = None) -> tuple[dict, bool]:
    """This person's account, and whether its history has to be started:
    this call made the account, or found one whose history never finished
    (the process stopped before or during seeding, and a member of staff
    cannot start over under a new name). Said once, also when a page's
    requests arrive together. Reads the database only for someone this
    process has not seen, and raises when it cannot."""
    url = url or get_config().database_url
    user_id = user_id_for(subject)
    known = _known.get(user_id)
    if known is not None:
        return known, False
    # While the database is away each read waits for a connection that is
    # not coming. One request finds that out; the ones behind it, from
    # everyone this process has not seen, are answered at once for a moment.
    global _down_until
    if time.monotonic() < _down_until:
        raise OutOfReach()
    try:
        user, start, made = _find_or_make(user_id, email, url)
    except Exception:
        _down_until = time.monotonic() + PAUSE_SECONDS
        raise
    with _lock:
        first = user_id not in _known
        if first:
            _known[user_id] = user
            if made:
                _seen_on[user_id] = _today()    # written with the account
        else:
            user = _known[user_id]
    return user, start and first


def made_by_sign_in(user_id: str, *, url: str | None = None) -> bool:
    """Whether this account is one the Google sign-in made: those carry the
    note of a visit from the moment they are made."""
    with get_engine(url or get_config().database_url).connect() as conn:
        return conn.execute(text("SELECT 1 FROM banko_meta WHERE key = :k"),
                            {"k": f"seen:{user_id}"}).fetchone() is not None


def retry_due(user_id: str, *, now: float) -> bool:
    """Whether a history that failed may be started again now: once every
    RETRY_SECONDS, however many pages ask."""
    with _lock:
        if now - _retried_at.get(user_id, float("-inf")) < RETRY_SECONDS:
            return False
        _retried_at[user_id] = now
        return True


def forget(user_id: str | None = None) -> None:
    """Stop taking an account (or every account) as still there: it expired,
    or the database was rebuilt."""
    global _down_until
    with _lock:
        for remembered in (_known, _seen_on, _retried_at):
            if user_id is None:
                remembered.clear()
            else:
                remembered.pop(user_id, None)
        if user_id is None:
            _down_until = 0.0


def due(user_id: str) -> bool:
    """Whether today's visit by this account has yet to be written down."""
    return _seen_on.get(user_id) != _today()


def seen(user_id: str, *, url: str | None = None) -> bool:
    """Write down that this account was used today. Once a day per process:
    banko_meta is a GLOBAL table, and a write to one is slow by design.
    Returns whether this call wrote."""
    today = _today()
    with _lock:
        if _seen_on.get(user_id) == today:
            return False
        _seen_on[user_id] = today
    try:
        run_transaction(lambda conn: conn.execute(text(
            "UPSERT INTO banko_meta (key, value, updated_at) VALUES (:k, 'seen', now())"),
            {"k": f"seen:{user_id}"}), url=url or get_config().database_url)
        return True
    except Exception as exc:
        with _lock:
            _seen_on.pop(user_id, None)
        # The class only: a database error's text carries the statement and its parameters.
        log.warning("a visit could not be written down (%s); the account's idle clock was not reset",
                    exc.__class__.__name__)
        return False


def expire_idle(days: int, *, url: str | None = None) -> int:
    """Remove the accounts this sign-in made that nobody has used for this
    many days, with every row they own. An account of this sign-in's carries
    the note of a visit from the moment it is made; usernames made at the
    sign-in page, seeded demo users, personas and the load simulator's
    customers carry none and are never touched. A visit is written down once
    a day, so the note can be a day older than the last use: that day is
    given back, and no account used within the days named is removed.
    0 keeps them all."""
    if days <= 0:
        return 0
    from .users import delete_users
    url = url or get_config().database_url
    personas = [p.user_id for p in get_catalog().personas]
    with get_engine(url).connect() as conn:
        idle = [str(r[0]) for r in conn.execute(text("""
            SELECT u.user_id FROM users AS u
            JOIN banko_meta AS m ON m.key = 'seen:' || u.user_id::STRING
            WHERE u.demo_user AND u.user_id <> ALL(CAST(:personas AS UUID[])) AND u.username NOT LIKE :load
              AND m.updated_at < now() - ((:days + 1) * INTERVAL '1 day')
        """), {"personas": personas, "load": LOAD_PREFIX + "%", "days": int(days)})]
    if not idle:
        return 0
    removed = delete_users(idle, url=url)
    log.info("removed %d accounts nobody had used for %d days", removed, days)
    return removed
