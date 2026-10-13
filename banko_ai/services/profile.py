"""Who the signed-in user is, for the shell and /api/v1/me."""

from __future__ import annotations

from sqlalchemy import text

from ..db.engine import get_engine
from ._sql import jsonable, valid_uuid


def get_profile(user_id: str, *, url: str | None = None) -> dict | None:
    if not valid_uuid(user_id):
        return None
    with get_engine(url).connect() as conn:
        row = conn.execute(text("""
            SELECT user_id, username, display_name, avatar_color, spending_style, home_region
            FROM users WHERE user_id = CAST(:u AS UUID)
        """), {"u": user_id}).fetchone()
    if row is None:
        return None
    return jsonable({"user_id": str(row[0]), "username": row[1], "display_name": row[2] or row[1],
                     "avatar_color": row[3] or "#6933FF", "spending_style": row[4], "home_region": row[5]})


def get_profiles(user_ids: list[str], *, url: str | None = None) -> list[dict]:
    ids = [u for u in user_ids if valid_uuid(u)]
    if not ids:
        return []
    with get_engine(url).connect() as conn:
        rows = conn.execute(text("""
            SELECT user_id, username, display_name, avatar_color, spending_style, home_region
            FROM users WHERE user_id = ANY(CAST(:ids AS UUID[]))
        """), {"ids": ids}).fetchall()
    found = {str(r[0]): r for r in rows}
    return jsonable([{"user_id": i, "username": found[i][1], "display_name": found[i][2] or found[i][1],
                      "avatar_color": found[i][3] or "#6933FF", "spending_style": found[i][4],
                      "home_region": found[i][5]} for i in ids if i in found])
