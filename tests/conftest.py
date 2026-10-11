import os

import pytest

# Normalize the scheme once for every test module: postgresql:// works for
# the app (it normalizes internally), so it should work for the suite too.
from banko_ai.utils.db_retry import normalize_db_url

if os.getenv("DATABASE_URL"):
    os.environ["DATABASE_URL"] = normalize_db_url(os.environ["DATABASE_URL"])


@pytest.fixture(autouse=True)
def _fresh_settings():
    """Settings are cached per process. Tests change the environment, so
    each one starts from a clean read."""
    from banko_ai.config.settings import reset_config

    reset_config()
    yield
    reset_config()


@pytest.fixture(scope="session", autouse=True)
def _ensure_app_schema():
    """Create the real application schema before any test touches the
    database. CI boots a bare CockroachDB, and without this the first
    test that creates a minimal expenses table poisons every later test
    that expects the full schema (embedding column and all)."""
    url = os.getenv("DATABASE_URL")
    if not url:
        yield
        return
    try:
        from banko_ai.utils.database import DatabaseManager
        from banko_ai.utils.migration import DatabaseMigration

        DatabaseManager(url).create_tables()
        migrator = DatabaseMigration(url)
        migrator.migrate_users_table()
        migrator.migrate_to_coach_v1()
    except Exception as e:
        # No database or no permissions: individual tests skip themselves.
        print(f"schema bootstrap skipped: {e}")
    yield
