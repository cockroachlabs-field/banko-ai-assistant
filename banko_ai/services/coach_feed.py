"""The coach inbox: the user's nudges with the signal behind each one. The
provider shown is the display label, mapped at read time."""

from __future__ import annotations

import json

from sqlalchemy import text

from ..ai_providers.display import display_label
from ..config.settings import get_config
from ..db.rls import request_url
from ._sql import jsonable, read_session, valid_uuid

_COLS = """n.nudge_id, n.message, n.created_at, n.provider_used, s.signal_type, s.severity, s.signal_id"""


def _obj(value):
    return value if isinstance(value, (dict, list)) or value is None else json.loads(value)


def _row(r) -> dict:
    cfg = get_config()
    provider = display_label(cfg, r[3]) or r[3]
    return {"nudge_id": str(r[0]), "message": r[1], "created_at": r[2], "provider": provider,
            "signal_type": r[4], "severity": r[5], "signal_id": str(r[6]) if r[6] else None}


def list_nudges(user_id: str, *, limit: int = 50, url: str | None = None) -> list[dict]:
    url = request_url(url)
    with read_session(user_id, url=url) as conn:
        rows = conn.execute(text(f"""
            SELECT {_COLS} FROM coach_nudges AS n
            LEFT JOIN spending_signals AS s ON s.signal_id = n.signal_id
            WHERE n.user_id = CAST(:u AS UUID) ORDER BY n.created_at DESC LIMIT :n
        """), {"u": user_id, "n": max(1, min(int(limit), 200))}).fetchall()
    return jsonable([_row(r) for r in rows])


def nudge_detail(user_id: str, nudge_id: str, *, url: str | None = None) -> dict | None:
    if not valid_uuid(nudge_id):
        return None
    url = request_url(url)
    with read_session(user_id, url=url) as conn:
        r = conn.execute(text(f"""
            SELECT {_COLS}, s.payload, n.tool_trace FROM coach_nudges AS n
            LEFT JOIN spending_signals AS s ON s.signal_id = n.signal_id
            WHERE n.nudge_id = CAST(:n AS UUID) AND n.user_id = CAST(:u AS UUID)
        """), {"n": nudge_id, "u": user_id}).fetchone()
    if r is None:
        return None
    return jsonable({**_row(r), "payload": _obj(r[7]) or {}, "tool_trace": _obj(r[8]) or []})
