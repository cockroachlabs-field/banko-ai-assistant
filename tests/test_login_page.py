"""The login page: persona cards from the catalog, create your own with the
catalog's styles, a neutral error when signup fails, and no Tailwind."""

import os

import pytest

from banko_ai.config.catalog import get_catalog
from banko_ai.web import auth
from tests._app import app_client, stop_apps

pytestmark = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    stop_apps()


def test_persona_cards(monkeypatch):
    html = app_client(monkeypatch).get("/login").get_data(as_text=True)
    for p in get_catalog().personas:
        assert p.display_name in html
        assert f'name="username" value="{p.username}"' in html
    assert "css/app.css" in html and "tailwind" not in html.lower()


def test_signup_offers_the_catalog_styles(monkeypatch):
    html = app_client(monkeypatch).post("/login", data={"username": "nobody-here-123"}).get_data(as_text=True)
    rest = html.split('id="signup-section"', 1)[1]
    assert "hidden" not in rest.split(">", 1)[0]
    section = rest.split("</form>", 1)[0]
    for style in get_catalog().profiles:
        assert f'value="{style}"' in section


def test_one_click_persona_sign_in(monkeypatch):
    client = app_client(monkeypatch)
    persona = get_catalog().personas[1]
    resp = client.post("/login", data={"username": persona.username})
    assert resp.status_code == 302
    with client.session_transaction() as s:
        assert str(s["user_id"]) == persona.user_id


def test_a_failed_signup_names_nothing_internal(monkeypatch):
    client = app_client(monkeypatch)

    def boom(self, *args, **kwargs):
        raise RuntimeError("postgresql://root@10.0.0.7:26257 refused")

    monkeypatch.setattr(auth.UserManager, "create", boom)
    style = sorted(get_catalog().profiles)[0]
    html = client.post("/signup", data={"username": "signup-fail-check", "spending_style": style}).get_data(as_text=True)
    assert "Signup failed" in html
    assert "postgresql" not in html and "10.0.0.7" not in html
