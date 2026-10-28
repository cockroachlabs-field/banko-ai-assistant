"""A named pool: its own engine and connections, labeled in SHOW SESSIONS."""

import os

import pytest
from sqlalchemy import text

from banko_ai.db.engine import get_engine, with_application_name

pytestmark = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")


def test_a_named_pool_shows_its_application_name():
    url = with_application_name(os.environ["DATABASE_URL"], "banko-load-test")
    engine = get_engine(url, pool_size=3)
    with engine.connect() as c:
        assert c.execute(text("SHOW application_name")).scalar() == "banko-load-test"
    assert engine.pool.size() == 3 and get_engine(url) is engine
    with get_engine().connect() as c:
        assert c.execute(text("SHOW application_name")).scalar() == "banko"


def test_with_application_name_keeps_the_rest_of_the_url():
    url = with_application_name("postgresql://root@localhost:26258/defaultdb?sslmode=disable", "banko-load")
    assert "sslmode=disable" in url and "application_name=banko-load" in url
