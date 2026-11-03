"""The login page: persona cards from the catalog, create your own with the
catalog's styles, a neutral error when signup fails, and no Tailwind."""

import os
import time
import uuid

import pytest

from banko_ai.config.catalog import get_catalog
from banko_ai.web import auth
from tests._app import app_client, stop_apps
from tests._money import drop_user

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


def test_a_signup_history_reports_progress_in_small_steps(monkeypatch):
    """A new user's history is written in batches of 50, so the setup bar
    moves on a 3-region cluster instead of sitting at 0 for the first 200."""
    client = app_client(monkeypatch)
    socketio = client.application.extensions["socketio"]
    real_emit, sent = socketio.emit, []

    def spy(event, data=None, **kwargs):
        if event == "seeding":
            sent.append(dict(data))
        return real_emit(event, data, **kwargs)

    monkeypatch.setattr(socketio, "emit", spy)
    name = f"steps-{uuid.uuid4().hex[:10]}"
    style = sorted(get_catalog().profiles)[0]
    assert client.post("/signup", data={"username": name, "spending_style": style}).status_code == 302
    with client.session_transaction() as s:
        uid = str(s["user_id"])
    try:
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline and not any(e["state"] in ("done", "failed") for e in sent):
            time.sleep(0.5)
        assert [e["state"] for e in sent][-1] == "done"
        running = [e["done"] for e in sent if e["user_id"] == uid and e["state"] == "running"]
        total = sent[-1]["total"]
        assert total > 50
        assert len([d for d in running if 0 < d < total]) > 1
        assert max(b - a for a, b in zip(running, running[1:])) <= 50
    finally:
        drop_user(uid)


def test_a_failed_signup_names_nothing_internal(monkeypatch):
    client = app_client(monkeypatch)

    def boom(self, *args, **kwargs):
        raise RuntimeError("postgresql://root@10.0.0.7:26257 refused")

    monkeypatch.setattr(auth.UserManager, "create", boom)
    style = sorted(get_catalog().profiles)[0]
    html = client.post("/signup", data={"username": "signup-fail-check", "spending_style": style}).get_data(as_text=True)
    assert "Signup failed" in html
    assert "postgresql" not in html and "10.0.0.7" not in html
