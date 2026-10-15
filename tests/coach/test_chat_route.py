"""/api/coach/chat answers for the signed-in user, or the default persona
without a session, never for a user id named in the request body: the
route sits under /api/, where no sign-in is enforced."""

import os

import pytest

from banko_ai.coach.agent import CoachAgent
from banko_ai.config.settings import get_config
from tests._app import app_client, stop_apps
from tests._money import drop_user, make_user

pytestmark = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")


def test_the_coach_answers_for_the_session_never_the_body(monkeypatch):
    asked_for: list[str] = []

    def converse(self, user_id, message, history=None, context=None):
        asked_for.append(user_id)
        return {"message": "ok", "tool_trace": [], "provider_used": "stub"}

    monkeypatch.setattr(CoachAgent, "converse", converse)
    me, someone_else = make_user(), make_user()
    try:
        client = app_client(monkeypatch)
        assert client.post("/api/coach/chat", json={"message": "how am I doing?",
                                                    "user_id": someone_else}).status_code == 200
        assert asked_for == [get_config().coach_default_user_id]
        with client.session_transaction() as s:
            s["user_id"] = me
        client.post("/api/coach/chat", json={"message": "and now?", "user_id": someone_else})
        assert asked_for[-1] == me
    finally:
        drop_user(me)
        drop_user(someone_else)
        stop_apps()
