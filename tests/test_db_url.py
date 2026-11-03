"""Every database URL spelling people already have in env files lands on
the CockroachDB dialect with the psycopg 3 driver."""

import os

import pytest

from banko_ai.utils.db_retry import normalize_db_url, to_libpq_url

BASE = "root@localhost:26257/defaultdb?sslmode=disable"


@pytest.mark.parametrize("given", [
    f"postgresql://{BASE}",
    f"postgres://{BASE}",
    f"cockroachdb://{BASE}",
    f"postgresql+psycopg2://{BASE}",
    f"cockroachdb+psycopg2://{BASE}",
    f"postgresql+psycopg://{BASE}",
    f"cockroachdb+psycopg://{BASE}",
])
def test_every_spelling_lands_on_psycopg3(given):
    assert normalize_db_url(given) == f"cockroachdb+psycopg://{BASE}"


def test_asyncpg_urls_pass_through():
    url = f"cockroachdb+asyncpg://{BASE}"
    assert normalize_db_url(url) == url


def test_libpq_url_for_raw_psycopg_connections():
    assert to_libpq_url(f"cockroachdb://{BASE}") == f"postgresql://{BASE}"
    assert to_libpq_url(f"postgresql+psycopg2://{BASE}") == f"postgresql://{BASE}"


@pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")
def test_engine_uses_psycopg3_driver():
    from sqlalchemy import create_engine, text

    engine = create_engine(normalize_db_url(os.environ["DATABASE_URL"]))
    try:
        with engine.connect() as conn:
            assert conn.execute(text("SELECT 1")).scalar() == 1
        assert engine.dialect.driver == "psycopg"
    finally:
        engine.dispose()
