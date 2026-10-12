"""Checks that every account's balance equals its opening balance plus its
ledger, using follower reads so the check never competes with live traffic.
The Platform view shows the latest report; drift is the one number a bank
demo must keep at zero while a region is down."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from ..config.settings import get_config
from ..db.engine import get_engine

log = logging.getLogger("banko.integrity")
_FOLLOWER = "AS OF SYSTEM TIME follower_read_timestamp()"


@dataclass(frozen=True)
class IntegrityReport:
    checked_at: datetime
    accounts: int
    drifted: tuple[dict, ...]
    skipped: str | None = None
    follower_reads: bool = True

    @property
    def ok(self) -> bool:
        return self.skipped is None and not self.drifted


_latest: IntegrityReport | None = None
_latest_lock = threading.Lock()


def _run(url: str | None, follower: bool, user_id: str | None) -> IntegrityReport:
    cfg = get_config()
    aost = _FOLLOWER if follower else ""
    where = "WHERE a.user_id = CAST(:u AS UUID)" if user_id else ""
    with get_engine(url).connect() as conn:
        count = int(conn.execute(text(f"SELECT count(*) FROM accounts AS a {aost} {where}"),
                                 {"u": user_id}).scalar() or 0)
        if count > cfg.integrity_max_accounts:
            return IntegrityReport(datetime.now(timezone.utc), count, (),
                                   skipped=f"more than {cfg.integrity_max_accounts} accounts",
                                   follower_reads=follower)
        rows = conn.execute(text(f"""
            SELECT a.account_id, a.user_id, a.balance, a.opening_balance + COALESCE(l.total, 0) AS expected
            FROM accounts AS a
            LEFT JOIN (SELECT account_id, sum(amount) AS total FROM ledger_entries GROUP BY account_id) AS l
              ON l.account_id = a.account_id
            {aost}
            {where}
        """), {"u": user_id}).fetchall()
    drifted = tuple({"account_id": str(r[0]), "user_id": str(r[1]),
                     "balance": float(r[2]), "expected": float(r[3])}
                    for r in rows if r[2] != r[3])
    return IntegrityReport(datetime.now(timezone.utc), count, drifted, follower_reads=follower)


def check_integrity(*, url: str | None = None, follower_reads: bool = True,
                    user_id: str | None = None) -> IntegrityReport:
    if follower_reads:
        try:
            return _run(url, True, user_id)
        except DBAPIError as exc:
            log.info("follower reads unavailable, checking current data instead: %s", exc.__class__.__name__)
    return _run(url, False, user_id)


def latest_report() -> IntegrityReport | None:
    with _latest_lock:
        return _latest


class IntegrityWorker:
    def __init__(self, url: str | None = None, interval: int | None = None,
                 stop: threading.Event | None = None,
                 on_report: Callable[[IntegrityReport], None] | None = None):
        self.url = url
        self.interval = get_config().integrity_interval_seconds if interval is None else interval
        self.stop = stop or threading.Event()
        self.on_report = on_report
        self._latest: IntegrityReport | None = None

    def latest(self) -> IntegrityReport | None:
        return self._latest

    def _loop(self) -> None:
        global _latest
        while not self.stop.is_set():
            try:
                report = check_integrity(url=self.url)
                self._latest = report
                with _latest_lock:
                    _latest = report
                if self.on_report:
                    self.on_report(report)
            except Exception:
                log.warning("integrity check failed", exc_info=True)
            self.stop.wait(self.interval)

    def start(self) -> threading.Thread | None:
        if self.interval <= 0:
            return None
        thread = threading.Thread(target=self._loop, name="banko-integrity", daemon=True)
        thread.start()
        return thread
