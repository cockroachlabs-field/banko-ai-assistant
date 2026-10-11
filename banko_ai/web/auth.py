"""
User authentication and management.

This module provides user authentication backed by CockroachDB for the
Banko AI Assistant.
"""

from typing import Any

from flask import session
from sqlalchemy import text

from ..utils.db_retry import get_database_url

# Success-only cache: lru_cache used to pin None after one connection
# blip, unpinning the user's rows for the rest of the process. Only a
# lookup that actually reached the users table is remembered.
_user_region_cache: dict[tuple[str, str], str | None] = {}


def resolve_user_region(user_id: str, database_url: str | None = None) -> str | None:
    """Resolve user's home region from the users table.

    Returns None if the user has no home_region set or on lookup errors.
    Successful lookups are cached per user; errors and not-yet-visible
    rows are retried on the next call.

    Args:
        user_id: User UUID
        database_url: Database connection string (defaults to env DATABASE_URL)
    """
    from ..db.engine import get_engine

    url = get_database_url(database_url)
    key = (user_id, url)
    if key in _user_region_cache:
        return _user_region_cache[key]

    engine = get_engine(url)
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT home_region FROM users WHERE user_id = :u"),
                {"u": user_id}
            )
            row = result.fetchone()
            if row is None:
                # Unknown user (or a signup race): answer None but let the
                # next call look again.
                return None
            if len(_user_region_cache) > 4096:
                _user_region_cache.clear()
            region = str(row[0]) if row[0] else None
            _user_region_cache[key] = region
            return region
    except Exception:
        return None


resolve_user_region.cache_clear = _user_region_cache.clear  # type: ignore[attr-defined]


class UserManager:
    """Database-backed user management."""

    def __init__(self, database_url: str | None = None):
        """Initialize user manager with database connection."""
        self.database_url = get_database_url(database_url)

    @property
    def engine(self):
        """Lazy engine initialization."""
        from ..db.engine import get_engine
        return get_engine(self.database_url)

    def ensure_schema(self) -> None:
        """Ensure users table exists (delegates to migration)."""
        from ..utils.migration import DatabaseMigration
        DatabaseMigration(self.database_url).migrate_users_table()

    def create(self, username: str, spending_style: str,
               home_region: str | None = None, demo_user: bool = True) -> dict[str, Any]:
        """Create a new user and return their record."""
        with self.engine.connect() as conn:
            result = conn.execute(
                text("""
                    INSERT INTO users (username, spending_style, home_region, demo_user)
                    VALUES (:username, :spending_style, :home_region, :demo_user)
                    RETURNING user_id, username, spending_style, home_region, created_at, demo_user
                """),
                {
                    "username": username,
                    "spending_style": spending_style,
                    "home_region": home_region,
                    "demo_user": demo_user,
                }
            )
            conn.commit()
            row = result.fetchone()
            return {
                "user_id": str(row[0]),
                "username": row[1],
                "spending_style": row[2],
                "home_region": row[3],
                "created_at": row[4],
                "demo_user": row[5],
            }

    def get(self, user_id: str) -> dict[str, Any] | None:
        """Get user by ID."""
        with self.engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT user_id, username, spending_style, home_region, created_at, demo_user
                    FROM users
                    WHERE user_id = :user_id
                """),
                {"user_id": user_id}
            )
            row = result.fetchone()
            if row:
                return {
                    "user_id": str(row[0]),
                    "username": row[1],
                    "spending_style": row[2],
                    "home_region": row[3],
                    "created_at": row[4],
                    "demo_user": row[5],
                }
            return None

    def get_by_username(self, username: str) -> dict[str, Any] | None:
        """Get user by username, case-insensitively: 'Maya' at the login
        prompt finds 'maya', and signup cannot mint a case-variant twin.
        The users table is tiny, so the lower() scan costs nothing."""
        with self.engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT user_id, username, spending_style, home_region, created_at, demo_user
                    FROM users
                    WHERE lower(username) = lower(:username)
                    LIMIT 1
                """),
                {"username": username}
            )
            row = result.fetchone()
            if row:
                return {
                    "user_id": str(row[0]),
                    "username": row[1],
                    "spending_style": row[2],
                    "home_region": row[3],
                    "created_at": row[4],
                    "demo_user": row[5],
                }
            return None

    def delete_by_username(self, username: str) -> None:
        """Delete user by username (test cleanup)."""
        with self.engine.connect() as conn:
            conn.execute(
                text("DELETE FROM users WHERE username = :username"),
                {"username": username}
            )
            conn.commit()

    def backfill_personas(self) -> None:
        """UPSERT the three legacy demo personas (idempotent)."""
        personas = [
            ("00000000-0000-0000-0000-0000000000a1", "maya", "diner"),
            ("00000000-0000-0000-0000-0000000000a2", "sam", "subscriber"),
            ("00000000-0000-0000-0000-0000000000a3", "riley", "saver"),
        ]
        with self.engine.connect() as conn:
            for user_id, username, spending_style in personas:
                conn.execute(
                    text("""
                        INSERT INTO users (user_id, username, spending_style, demo_user)
                        VALUES (:user_id, :username, :spending_style, true)
                        ON CONFLICT (username) DO NOTHING
                    """),
                    {
                        "user_id": user_id,
                        "username": username,
                        "spending_style": spending_style,
                    }
                )
            conn.commit()

    def get_current_user(self) -> dict[str, Any] | None:
        """Get current user from Flask session."""
        user_id = session.get('user_id')
        if user_id:
            return self.get(user_id)
        return None

    def logout_user(self) -> None:
        """Logout current user (clear session)."""
        session.pop('user_id', None)
        session.pop('username', None)

    def create_user(self, username: str, email: str = None) -> str:
        """Legacy interface for existing app.py call sites (maps to create
        with default style)."""
        user = self.create(username, spending_style="diner", demo_user=True)
        return user["user_id"]

    def login_user(self, user_id: str) -> bool:
        """Legacy interface for existing app.py login route."""
        user = self.get(user_id)
        if user:
            session['user_id'] = user_id
            session['username'] = user['username']
            return True
        return False


def clear_demo_users(database_url: str) -> dict[str, int]:
    """Delete all demo users and their dependent data, preserving the three
    legacy personas. Returns per-table deletion counts."""
    from ..db.engine import get_engine

    engine = get_engine(get_database_url(database_url))
    counts = {}

    legacy_persona_ids = [
        "00000000-0000-0000-0000-0000000000a1",
        "00000000-0000-0000-0000-0000000000a2",
        "00000000-0000-0000-0000-0000000000a3",
    ]

    with engine.connect() as conn:
            demo_user_ids_result = conn.execute(
                text("""
                    SELECT user_id FROM users
                    WHERE demo_user = true
                    AND user_id NOT IN (:p1, :p2, :p3)
                """),
                {"p1": legacy_persona_ids[0], "p2": legacy_persona_ids[1], "p3": legacy_persona_ids[2]}
            )
            demo_user_ids = [str(row[0]) for row in demo_user_ids_result]

            if not demo_user_ids:
                return {"users": 0, "expenses": 0, "user_budgets": 0, "spending_signals": 0, "coach_nudges": 0}

            placeholders = ", ".join([f":uid{i}" for i in range(len(demo_user_ids))])
            uid_params = {f"uid{i}": uid for i, uid in enumerate(demo_user_ids)}

            result = conn.execute(
                text(f"""
                    DELETE FROM coach_nudges
                    WHERE signal_id IN (
                        SELECT signal_id FROM spending_signals
                        WHERE user_id IN ({placeholders})
                    )
                """),
                uid_params
            )
            counts["coach_nudges"] = result.rowcount

            result = conn.execute(
                text(f"DELETE FROM spending_signals WHERE user_id IN ({placeholders})"),
                uid_params
            )
            counts["spending_signals"] = result.rowcount

            result = conn.execute(
                text(f"DELETE FROM user_budgets WHERE user_id IN ({placeholders})"),
                uid_params
            )
            counts["user_budgets"] = result.rowcount

            result = conn.execute(
                text(f"DELETE FROM expenses WHERE user_id IN ({placeholders})"),
                uid_params
            )
            counts["expenses"] = result.rowcount

            result = conn.execute(
                text(f"DELETE FROM users WHERE user_id IN ({placeholders})"),
                uid_params
            )
            counts["users"] = result.rowcount

            conn.commit()

    return counts
