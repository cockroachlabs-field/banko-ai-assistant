import os
from urllib.parse import urlsplit

import pytest


def _guard_database_url() -> None:
    """The suite writes and drops tables. A shell profile that exports the
    live demo cluster (port 26257) must never be picked up by accident, so
    the target has to be named explicitly, and 26257 needs an opt-in."""
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        pytest.exit(
            "DATABASE_URL is not set. Point the suite at a throwaway CockroachDB, for example\n"
            '  DATABASE_URL="postgresql://root@localhost:26258/defaultdb?sslmode=disable" uv run pytest tests/',
            returncode=4)
    port = urlsplit(url).port or 26257
    if port == 26257 and os.environ.get("BANKO_ALLOW_CHAOS_TESTS") != "1":
        pytest.exit(
            "DATABASE_URL points at port 26257, which is usually the live demo cluster.\n"
            "Use a throwaway node (for example port 26258), or set BANKO_ALLOW_CHAOS_TESTS=1 "
            "if that database really is disposable (CI does).",
            returncode=4)


_guard_database_url()

# The app no longer sniffs PYTEST_CURRENT_TEST; tests opt out of sample data explicitly.
os.environ.setdefault("BANKO_SKIP_AUTOSETUP", "1")
# No background integrity sweeps; tests call check_integrity directly.
os.environ.setdefault("INTEGRITY_INTERVAL_SECONDS", "0")
# The shell profile turns Kafka on for the live demo, and any app a test
# builds would join that consumer group. A test that wants it sets it itself.
os.environ["COACH_KAFKA_ENABLED"] = "false"

# Normalize the scheme once for every test module: postgresql:// works for
# the app (it normalizes internally), so it should work for the suite too.
from banko_ai.utils.db_retry import normalize_db_url

if os.getenv("DATABASE_URL"):
    os.environ["DATABASE_URL"] = normalize_db_url(os.environ["DATABASE_URL"])


@pytest.fixture(autouse=True)
def _fresh_settings():
    """Settings are cached per process. Tests change the environment, so
    each one starts from a clean read."""
    from banko_ai.config.catalog import reset_catalog
    from banko_ai.config.settings import reset_config

    reset_config()
    reset_catalog()
    yield
    reset_config()
    reset_catalog()


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
