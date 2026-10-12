"""
Test the signup flow: login, signup, welcome signal.
"""

import os

import pytest
from sqlalchemy import create_engine, text

from banko_ai.web.app import create_app
from banko_ai.web.auth import UserManager
from tests._money import drop_user

DB = os.getenv("DATABASE_URL")
pytestmark = pytest.mark.skipif(not DB, reason="DATABASE_URL not set")


@pytest.fixture(scope="module")
def client():
    os.environ["BANKO_SKIP_AUTOSETUP"] = "1"
    app = create_app()
    app.config["TESTING"] = True
    yield app.test_client()
    app.extensions["banko_warmup_stop"].set()
    # Cleanup: delete the test user and their rows. Retries because on a
    # shared cluster this commit can lose a RETRY_SERIALIZABLE race with
    # a running app and fail the whole gate from teardown.
    if DB:
        import time
        engine = create_engine(DB)
        for attempt in range(4):
            try:
                with engine.begin() as conn:
                    # Welcome-signal rows too, or every run leaks a
                    # spending_signals/coach_nudges pair onto a shared DB.
                    conn.execute(text(
                        "DELETE FROM coach_nudges WHERE user_id IN "
                        "(SELECT user_id FROM users WHERE username = 'flow-test-user')"))
                    conn.execute(text(
                        "DELETE FROM spending_signals WHERE user_id IN "
                        "(SELECT user_id FROM users WHERE username = 'flow-test-user')"))
                    row = conn.execute(text(
                        "SELECT user_id FROM users WHERE username = 'flow-test-user'")).fetchone()
                if row is not None:
                    drop_user(str(row[0]))
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(0.5 * (attempt + 1))
        engine.dispose()


def test_fresh_visit_redirects_to_login(client):
    r = client.get("/")
    assert r.status_code == 302 and "/login" in r.headers["Location"]


def test_signup_creates_seeds_and_logs_in(client):
    r = client.post("/signup", data={
        "username": "flow-test-user", "spending_style": "saver"},
        follow_redirects=False)
    assert r.status_code == 302
    r2 = client.get("/")
    assert r2.status_code == 200
    # History seeds in the background; wait for it so later tests (and
    # teardown) see a finished user, not a half-written one.
    user = UserManager(DB).get_by_username("flow-test-user")
    assert _wait_seeded(user["user_id"]) == "done"


def _wait_seeded(user_id: str, seconds: float = 300) -> str | None:
    import time

    from banko_ai.services.seeding import seeding_status
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        status = seeding_status(user_id)
        if status and status["state"] in ("done", "failed"):
            return status["state"]
        time.sleep(0.5)
    return None


def test_returning_username_signs_in(client):
    r = client.post("/login", data={"username": "flow-test-user"})
    assert r.status_code == 302 and r.headers["Location"].endswith("/")


def test_unknown_login_offers_signup(client):
    r = client.post("/login", data={"username": "never-seen"})
    assert r.status_code == 200  # re-renders with the style picker open


def test_duplicate_signup_never_deletes_existing_user(client):
    # A second signup with a taken username must leave the original
    # account and its seeded history untouched (it once deleted both).
    r = client.post("/signup", data={
        "username": "flow-test-user", "spending_style": "diner"},
        follow_redirects=False)
    assert r.status_code == 200  # back to login, not an account wipe
    um = UserManager(DB)
    user = um.get_by_username("flow-test-user")
    assert user is not None
    engine = create_engine(DB)
    with engine.connect() as conn:
        count = conn.execute(
            text("SELECT count(*) FROM expenses WHERE user_id = :uid"),
            {"uid": user["user_id"]}).scalar()
    assert count > 0


def test_identity_pill_shows_signed_in_user(client):
    # Signed-in pages carry the header pill with the username and a
    # switch-user link; the login page (no session) must not.
    client.post("/login", data={"username": "flow-test-user"})
    page = client.get("/").get_data(as_text=True)
    assert "flow-test-user" in page
    assert "Switch user" in page
    client.get("/logout")
    login_page = client.get("/login").get_data(as_text=True)
    assert "Switch user" not in login_page


def test_login_page_radios_not_required_when_hidden(client):
    # The first login screen hides the style picker; required radios
    # inside a hidden section block browser form submission entirely,
    # which made the Continue button dead in real browsers.
    r = client.get("/login")
    html = r.get_data(as_text=True)
    assert 'name="spending_style"' in html
    assert "required" not in html.split('id="signup-section"')[1].split("</form>")[0]
