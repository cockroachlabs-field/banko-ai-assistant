"""The old UI is gone: no static tabs, no Tailwind or Font Awesome, and old
links land on the new pages."""

import os
from pathlib import Path

import pytest

from tests._app import app_client, stop_apps
from tests._money import drop_user, make_user

ROOT = Path(__file__).resolve().parents[1] / "banko_ai"
GONE = ["templates/index.html", "templates/dashboard.html", "templates/data_generator.html",
        "templates/agent_dashboard.html", "templates/coach.html", "templates/base.html", "templates/_identity.html",
        "templates/login.html", "static/vendor/tailwind.js", "static/vendor/fontawesome", "static/Graph.png",
        "static/Graph2.png", "static/css/style.css"]


def test_old_files_are_gone():
    for rel in GONE:
        assert not (ROOT / rel).exists(), rel


def test_no_template_loads_tailwind_or_font_awesome():
    for tpl in (ROOT / "templates").rglob("*.html"):
        text = tpl.read_text().lower()
        assert "tailwind" not in text and "fontawesome" not in text and "fa-solid" not in text, tpl.name


@pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")
def test_old_links_redirect(monkeypatch):
    uid = make_user()
    try:
        client = app_client(monkeypatch)
        with client.session_transaction() as s:
            s["user_id"] = uid
        for old, new in [("/banko", "/assistant"), ("/home", "/"), ("/benefits", "/"), ("/savings", "/accounts"),
                         ("/wallet", "/accounts"), ("/credit-card", "/accounts"), ("/statements", "/accounts"),
                         ("/data-generator", "/settings#data")]:
            resp = client.get(old)
            assert resp.status_code in (302, 303), old
            assert resp.headers["Location"].replace("http://localhost", "") == new, old
        assert client.post("/banko", data={"user_input": "hi"}).status_code == 303
    finally:
        drop_user(uid)
        stop_apps()


@pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")
def test_the_unscoped_legacy_apis_are_gone(monkeypatch):
    """No sign-in and no user scope: these read and cleared any session's
    chat history and searched every user's expense vectors."""
    try:
        client = app_client(monkeypatch)
        assert client.get("/api/chat-history/default").status_code == 404
        assert client.delete("/api/chat-history/default").status_code == 404
        assert client.post("/api/vectorstore-search", json={"query": "coffee"}).status_code == 404
    finally:
        stop_apps()
