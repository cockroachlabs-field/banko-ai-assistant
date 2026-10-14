"""The app shell: identity, persona, nav from registered pages, the user
guard header, and HTML errors for pages."""

import os
import re

import pytest
from flask import render_template_string

from tests._app import app_client, stop_apps
from tests._money import drop_user, make_user

pytestmark = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")
PAGE = '{% extends "layout.html" %}{% block content %}<p id="probe">hello</p>{% endblock %}'


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    stop_apps()


@pytest.fixture
def member():
    uid = make_user("balanced")
    yield uid
    drop_user(uid)


def _render(client, user_id):
    app = client.application
    with app.test_request_context("/"):
        from flask import session
        session["user_id"] = user_id
        session["username"] = "tester"
        return render_template_string(PAGE)


def test_shell_renders_identity_brand_and_persona(monkeypatch, member):
    client = app_client(monkeypatch)
    html = _render(client, member)
    assert 'id="probe"' in html
    assert 'href="/static/css/app.css"' in html or "css/app.css" in html
    assert "--ac: #6933FF" in html
    assert "Switch user" in html
    assert 'id="banko-shell"' in html
    shown = client.application.extensions["banko_ai"]["shown"]()
    assert shown.name in html


def test_no_external_scripts_or_styles(monkeypatch, member):
    html = _render(app_client(monkeypatch), member)
    assert not re.search(r'(src|href)="https?://', html)


def test_persona_hides_the_actual_provider(monkeypatch, member):
    client = app_client(monkeypatch, AI_DISPLAY_PROVIDER="watsonx", AI_DISPLAY_MODEL="ibm/granite-3-3-8b-instruct")
    html = _render(client, member).lower()
    assert "ollama" not in html
    assert "watsonx" in html


def test_nav_only_lists_registered_pages(monkeypatch):
    from banko_ai.web.shell import nav_groups
    app = app_client(monkeypatch).application
    with app.test_request_context("/"):
        hrefs = [i["href"] for g in nav_groups(app) for i in g["items"]]
        for group in nav_groups(app):
            for item in group["items"]:
                assert item["key"] and item["href"].startswith("/")
        endpoints = set(app.view_functions)
        from banko_ai.web.shell import NAV
        expected = [n for n in NAV if n["endpoint"] in endpoints]
        assert len(hrefs) == len(expected)
        assert "/platform" not in hrefs


def test_api_responses_name_the_user(monkeypatch, member):
    client = app_client(monkeypatch)
    with client.session_transaction() as s:
        s["user_id"] = member
    assert client.get("/api/v1/me").headers["X-Banko-User"] == member
    anon = app_client(monkeypatch)
    from banko_ai.config.settings import get_config
    assert anon.get("/api/v1/me").headers["X-Banko-User"] == get_config().coach_default_user_id


def test_page_404_is_html_and_api_404_is_json(monkeypatch, member):
    client = app_client(monkeypatch)
    with client.session_transaction() as s:
        s["user_id"] = member
    page = client.get("/no-such-page")
    assert page.status_code == 404 and page.mimetype == "text/html" and b"app.css" in page.data
    api = client.get("/api/v1/no-such-thing")
    assert api.status_code == 404 and api.is_json


def test_a_session_for_a_user_who_is_gone_starts_over_at_login(monkeypatch):
    # An expired demo user, or a rebuilt cluster, while the browser keeps its cookie.
    import uuid
    client = app_client(monkeypatch)
    for path in ("/", "/settings", "/assistant"):
        with client.session_transaction() as s:
            s["user_id"] = str(uuid.uuid4())
            s["username"] = "gone"
        resp = client.get(path)
        assert resp.status_code == 302 and resp.headers["Location"].endswith("/login"), path
        with client.session_transaction() as s:
            assert "user_id" not in s and "username" not in s
    assert client.get("/login").status_code == 200


def test_error_pages_render_when_the_profile_lookup_fails(monkeypatch, member):
    import banko_ai.web.shell as shell
    client = app_client(monkeypatch)
    with client.session_transaction() as s:
        s["user_id"] = member

    def down(*args, **kwargs):
        raise RuntimeError("connection to server at 10.0.0.7 port 26257 failed")

    real = shell.get_profile
    monkeypatch.setattr(shell, "get_profile", down)
    missing = client.get("/no-such-page")
    assert missing.status_code == 404 and missing.mimetype == "text/html" and b"app.css" in missing.data
    broken = client.get("/")
    assert broken.status_code == 500 and broken.mimetype == "text/html" and b"app.css" in broken.data
    for resp in (missing, broken):
        assert b"10.0.0.7" not in resp.data and b"RuntimeError" not in resp.data
    with client.session_transaction() as s:
        assert s["user_id"] == member  # a blip signs nobody out
    monkeypatch.setattr(shell, "get_profile", real)
    assert client.get("/").status_code == 200


def test_profile_service(member):
    from banko_ai.services.profile import get_profile
    p = get_profile(member)
    assert p["user_id"] == member and p["spending_style"] == "balanced"
    assert get_profile("not-a-uuid") is None
