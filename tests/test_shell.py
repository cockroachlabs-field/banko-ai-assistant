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


def test_profile_service(member):
    from banko_ai.services.profile import get_profile
    p = get_profile(member)
    assert p["user_id"] == member and p["spending_style"] == "balanced"
    assert get_profile("not-a-uuid") is None
