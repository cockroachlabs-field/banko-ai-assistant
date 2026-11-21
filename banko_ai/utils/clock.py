"""What day it is, on the one clock Banko keeps: UTC.

Every date Banko stores is a UTC date. The ledger dates a purchase by it,
seeding writes history up to it, the reports count the month by it. A
question about "today", "this week" or "the last 30 days" has to mean the
same day, or a purchase made a minute ago can fall outside it: west of UTC
the machine's own calendar is a day behind every evening.

Nothing in banko_ai reads the machine's own date or time of day;
tests/test_one_clock.py lists whoever does.
"""

from __future__ import annotations

from datetime import date, datetime, timezone


def now() -> datetime:
    """The time, in UTC, with its zone."""
    return datetime.now(timezone.utc)


def today() -> date:
    """Today's date in UTC: the day a purchase made now is dated."""
    return now().date()
