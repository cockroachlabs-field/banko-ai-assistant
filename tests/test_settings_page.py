"""Settings: every section renders, and the data generator reports real
progress over Socket.IO."""

import os
import time

import pytest

from banko_ai.services import seeding
from tests._app import app_client, stop_apps
from tests._money import drop_user, make_user

pytestmark = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    stop_apps()


def test_admin_seed_emits_progress(monkeypatch):
    client = app_client(monkeypatch)
    emitted = []
    socketio = client.application.extensions["socketio"]
    monkeypatch.setattr(socketio, "emit", lambda event, payload=None, **kw: emitted.append((event, payload)))

    def fake_seed(count, *, months=None, url=None, on_progress=None):
        on_progress("00000000-0000-0000-0000-00000000beef", 3, 10, "running")
        on_progress("00000000-0000-0000-0000-00000000beef", 10, 10, "done")
        return ["00000000-0000-0000-0000-00000000beef"]

    monkeypatch.setattr(seeding, "seed_generated_users", fake_seed)
    resp = client.post("/api/v1/admin/seed", json={"users": 1, "months": 1})
    assert resp.status_code == 202
    deadline = time.time() + 5
    while len([e for e in emitted if e[0] == "seeding"]) < 2 and time.time() < deadline:
        time.sleep(0.05)
    assert [p["state"] for e, p in emitted if e == "seeding"] == ["running", "done"]


def test_settings_page_sections(monkeypatch):
    uid = make_user()
    try:
        client = app_client(monkeypatch)
        with client.session_transaction() as s:
            s["user_id"] = uid
        html = client.get("/settings").get_data(as_text=True)
        for section in ("Profile", "Language", "AI model", "Theme", "Demo data", "Memories", "System"):
            assert section in html, section
        assert 'id="settings-personas"' in html
    finally:
        drop_user(uid)
