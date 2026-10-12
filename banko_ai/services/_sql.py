"""Shared plumbing for read services: a user-scoped read transaction and a
converter that makes service results safe to hand to jsonify."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection
from sqlalchemy.exc import DBAPIError

from ..config.settings import get_config
from ..db.engine import _sqlstate, get_engine

log = logging.getLogger("banko.services")
_FOLLOWER = "SET TRANSACTION AS OF SYSTEM TIME follower_read_timestamp()"
_follower_ok: dict[str, bool] = {}  # per database URL, definitive answers only
# SQLSTATE classes that say nothing about follower read support: connection
# trouble, transaction rollbacks, operator intervention.
_TRANSIENT_CLASSES = ("08", "40", "57")


def _follower_available(url: str | None) -> bool:
    key = url or get_config().database_url
    known = _follower_ok.get(key)
    if known is not None:
        return known
    try:
        with get_engine(url).begin() as conn:
            conn.execute(text(_FOLLOWER))
            conn.execute(text("SELECT set_config('banko.user_id', '', true)"))
        _follower_ok[key] = True
        return True
    except DBAPIError as exc:
        state = _sqlstate(exc)
        definitive = bool(state) and not state.startswith(_TRANSIENT_CLASSES) and not exc.connection_invalidated
        if definitive:
            _follower_ok[key] = False
            log.info("follower reads unavailable, reports read current data: %s", state)
        else:
            log.info("follower read probe failed, reading current data for now: %s",
                     state or exc.__class__.__name__)
        return False


@contextmanager
def read_session(user_id: str, *, url: str | None = None, follower: bool = False) -> Iterator[Connection]:
    """A read transaction with banko.user_id set (row-level security reads
    it). follower=True reads a few seconds in the past from the nearest
    replica, for reports only."""
    use_follower = follower and get_config().reporting_follower_reads and _follower_available(url)
    with get_engine(url).begin() as conn:
        if use_follower:
            conn.execute(text(_FOLLOWER))
        conn.execute(text("SELECT set_config('banko.user_id', :uid, true)"), {"uid": str(user_id)})
        yield conn


def jsonable(value: Any) -> Any:
    """Decimals become numbers rounded to cents, dates ISO strings, UUIDs strings."""
    if isinstance(value, Decimal):
        return float(round(value, 2))
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def valid_uuid(value: Any) -> bool:
    try:
        uuid.UUID(str(value))
        return True
    except ValueError:
        return False
