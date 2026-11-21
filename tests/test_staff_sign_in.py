"""Behind Google's Identity-Aware Proxy everyone is signed in as themselves.

The hosted Banko asked each person for a username with no password, so in
practice everyone shared maya, sam and riley, and one person's threads,
memories and receipts showed up for the next. With IAP_AUDIENCE set, the
identity comes from the assertion IAP signs, and from nothing else."""

import os
import re
import time
import uuid

import pytest
from sqlalchemy import text

from banko_ai.db.engine import get_engine
from banko_ai.services import staff
from banko_ai.services.users import delete_users
from tests._app import app_client, stop_apps
from tests._iap import AUDIENCE, Key, claims, subject

pytestmark = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")

KEY = Key("iap-key")
MAYA = "00000000-0000-0000-0000-0000000000a1"


def _address(name: str) -> str:
    return f"{name}-{uuid.uuid4().hex[:8]}@example.com"


def _as(email: str) -> dict:
    return {"x-goog-iap-jwt-assertion": KEY.sign(claims(email))}


def _id(email: str) -> str:
    """The Banko user id of the Google account behind this address."""
    return staff.user_id_for(subject(email))


@pytest.fixture
def hosted(monkeypatch):
    """An app behind IAP: Google's keys are the test's, and starting an
    account (seeding it, then the welcome nudge) is recorded, not run."""
    from banko_ai.web import iap
    monkeypatch.setattr(iap, "fetch_keys", lambda: {KEY.kid: KEY.public})
    client = app_client(monkeypatch, IAP_AUDIENCE=AUDIENCE)
    started: list[dict] = []
    client.application.extensions["banko_start_account"] = started.append
    emails: list[str] = []

    def visitor(name: str) -> str:
        emails.append(_address(name))
        return emails[-1]

    yield client, visitor, started
    stop_apps()
    delete_users([_id(e) for e in emails])
    staff.forget()


def _who(client) -> str | None:
    with client.session_transaction() as session:
        return session.get("user_id")


def test_the_first_request_signs_the_person_in_as_themselves(hosted):
    client, visitor, started = hosted
    ana = visitor("ana.b")
    page = client.get("/", headers=_as(ana))
    assert page.status_code == 200
    assert _who(client) == _id(ana)
    assert [u["user_id"] for u in started] == [_id(ana)]      # made once, and its history started
    with get_engine().connect() as c:
        name, style = c.execute(text("SELECT username, spending_style FROM users WHERE user_id = CAST(:u AS UUID)"),
                                {"u": _id(ana)}).one()
    assert name == staff.username_for(ana) and style == "balanced"
    assert staff.username_for(ana).encode() in page.data                    # the page says who is signed in


def test_the_next_request_finds_the_same_account(hosted):
    client, visitor, started = hosted
    ana = visitor("ana.b")
    client.get("/", headers=_as(ana))
    again = client.get("/accounts", headers=_as(ana))
    assert again.status_code == 200 and _who(client) == _id(ana) and len(started) == 1
    api = client.get("/api/v1/me", headers=_as(ana))
    assert api.headers["X-Banko-User"] == _id(ana)


def test_a_posted_username_is_never_an_identity(hosted):
    """Nobody becomes somebody else by typing a name."""
    client, visitor, _ = hosted
    ana = visitor("ana.b")
    client.get("/", headers=_as(ana))
    typed = client.post("/login", data={"username": "maya"}, headers=_as(ana))
    assert typed.status_code in (302, 303) and _who(client) == _id(ana)
    made_up = f"typed-{uuid.uuid4().hex[:8]}"
    signed_up = client.post("/signup", data={"username": made_up, "spending_style": "saver"}, headers=_as(ana))
    assert signed_up.status_code in (302, 303) and _who(client) == _id(ana)
    with get_engine().connect() as c:
        assert c.execute(text("SELECT count(*) FROM users WHERE username = :n"), {"n": made_up}).scalar() == 0


def test_the_header_anyone_can_forge_is_never_read(hosted):
    client, visitor, started = hosted
    boss = visitor("the.boss")
    page = client.get("/", headers={"x-goog-authenticated-user-email": f"accounts.google.com:{boss}",
                                    "x-goog-authenticated-user-id": "accounts.google.com:1"})
    assert page.status_code == 302 and "/login" in page.headers["Location"]
    assert _who(client) is None and started == []
    with get_engine().connect() as c:
        assert c.execute(text("SELECT count(*) FROM users WHERE user_id = CAST(:u AS UUID)"),
                         {"u": _id(boss)}).scalar() == 0


def test_an_assertion_that_does_not_check_signs_nobody_in(hosted):
    client, visitor, started = hosted
    ana = visitor("ana.b")
    stranger = Key(KEY.kid)                                                  # same key id, someone else's key
    for bad in ({"x-goog-iap-jwt-assertion": stranger.sign(claims(ana))},
                {"x-goog-iap-jwt-assertion": KEY.sign(claims(ana, aud="/projects/1/locations/x/services/other"))},
                {"x-goog-iap-jwt-assertion": "not.a.token"}, {}):
        page = client.get("/", headers=bad)
        assert page.status_code == 302 and "/login" in page.headers["Location"], bad
    assert started == []
    notice = client.get("/login")                                            # never an error page, and no way in by name
    assert notice.status_code == 200 and b'name="username"' not in notice.data
    assert b"Google" in notice.data


def test_a_session_alone_is_not_enough_without_the_assertion(hosted):
    """The cookie of an earlier visit, replayed past the proxy: not signed in."""
    client, visitor, _ = hosted
    ana = visitor("ana.b")
    assert client.get("/", headers=_as(ana)).status_code == 200
    replayed = client.get("/")
    assert replayed.status_code == 302 and _who(client) is None


def _events(socket) -> list[str]:
    return [e["name"] for e in socket.get_received()]


def test_a_socket_reaches_a_users_room_only_with_the_assertion(hosted):
    """Nudges, agent activity and seeding progress go to the signed-in user's
    room. Socket.IO requests pass none of the page's checks, so the room asks
    for the assertion itself: the cookie of an earlier visit does not open it."""
    from banko_ai.web.assistant_api import user_room
    client, visitor, _ = hosted
    ana, ben = visitor("ana.b"), visitor("ben.c")
    client.get("/", headers=_as(ben))
    client.get("/", headers=_as(ana))                         # the cookie now says ana
    app = client.application
    socketio = app.extensions["socketio"]
    room = user_room(_id(ana))

    replayed = socketio.test_client(app, flask_test_client=client)
    someone_else = socketio.test_client(app, flask_test_client=client, headers=_as(ben))
    signed_in = socketio.test_client(app, flask_test_client=client, headers=_as(ana))
    for socket in (replayed, someone_else, signed_in):
        socket.emit("coach.join", {})
    socketio.emit("coach.nudge", {"nudge_id": "n-1"}, to=room)
    assert _events(replayed) == [] and _events(someone_else) == []
    assert _events(signed_in) == ["coach.nudge"]


def test_a_socket_follows_the_persona_being_viewed(hosted):
    from banko_ai.web.assistant_api import user_room
    client, visitor, _ = hosted
    ana = visitor("ana.b")
    client.get("/", headers=_as(ana))
    client.post("/view-as", data={"who": "maya"}, headers=_as(ana))
    app = client.application
    socketio = app.extensions["socketio"]
    viewing = socketio.test_client(app, flask_test_client=client, headers=_as(ana))
    viewing.emit("coach.join", {})
    socketio.emit("coach.nudge", {"nudge_id": "n-2"}, to=user_room(MAYA))
    socketio.emit("coach.nudge", {"nudge_id": "n-3"}, to=user_room(_id(ana)))
    assert [e["args"][0]["nudge_id"] for e in viewing.get_received() if e["name"] == "coach.nudge"] == ["n-2"]


def test_personas_are_a_view_away_and_so_is_the_way_back(hosted):
    """The scripted demo numbers live on maya, sam and riley."""
    client, visitor, _ = hosted
    ana = visitor("ana.b")
    client.get("/", headers=_as(ana))
    picker = client.get("/login", headers=_as(ana))
    assert picker.status_code == 200 and b"View as" in picker.data and b'name="username"' not in picker.data
    switched = client.post("/view-as", data={"who": "maya"}, headers=_as(ana))
    assert switched.status_code in (302, 303) and _who(client) == MAYA
    as_maya = client.get("/", headers=_as(ana))
    assert as_maya.status_code == 200 and b"Viewing as" in as_maya.data
    assert client.get("/api/v1/me", headers=_as(ana)).headers["X-Banko-User"] == MAYA
    back = client.post("/view-as", data={"who": "me"}, headers=_as(ana))
    assert back.status_code in (302, 303) and _who(client) == _id(ana)


def test_view_as_reaches_personas_only(hosted):
    client, visitor, _ = hosted
    ana, ben = visitor("ana.b"), visitor("ben.c")
    client.get("/", headers=_as(ben))                       # ben has an account
    client.get("/", headers=_as(ana))
    for who in (staff.username_for(ben), _id(ben), "", "load-00001-ab12"):
        refused = client.post("/view-as", data={"who": who}, headers=_as(ana))
        assert refused.status_code in (302, 303, 400) and _who(client) == _id(ana), who


def test_another_person_on_the_same_browser_gets_their_own_account(hosted):
    client, visitor, _ = hosted
    ana, ben = visitor("ana.b"), visitor("ben.c")
    client.get("/", headers=_as(ana))
    client.post("/view-as", data={"who": "maya"}, headers=_as(ana))
    assert _who(client) == MAYA
    client.get("/", headers=_as(ben))                       # same cookie jar, a different Google account
    assert _who(client) == _id(ben)


def test_an_account_that_was_removed_is_made_again_on_the_next_visit(hosted):
    """It expired, or the database was rebuilt: no loop back to the sign-in page."""
    client, visitor, started = hosted
    ana = visitor("ana.b")
    client.get("/", headers=_as(ana))
    delete_users([_id(ana)])
    page = client.get("/", headers=_as(ana), follow_redirects=True)
    assert page.status_code == 200 and _who(client) == _id(ana) and len(started) == 2


def test_a_visit_is_written_down_for_the_idle_clock(hosted):
    """The note is written with the account. A visit on a later day has to
    write it again, or the idle clock would count from the first visit."""
    client, visitor, _ = hosted
    ana = visitor("ana.b")
    client.get("/", headers=_as(ana))
    key = f"seen:{_id(ana)}"
    with get_engine().begin() as c:
        c.execute(text("UPDATE banko_meta SET updated_at = now() - INTERVAL '3 days' WHERE key = :k"), {"k": key})
    staff._seen_on.pop(_id(ana), None)                       # a new day, as far as this process knows
    client.get("/accounts", headers=_as(ana))
    deadline, age = time.monotonic() + 10, None
    while time.monotonic() < deadline:
        with get_engine().connect() as c:
            age = c.execute(text("SELECT now() - updated_at < INTERVAL '1 hour' FROM banko_meta WHERE key = :k"),
                            {"k": key}).scalar()
        if age:
            break
        time.sleep(0.1)
    assert age is True


def test_without_the_setting_banko_asks_for_a_username_as_before(monkeypatch):
    client = app_client(monkeypatch)                         # no IAP_AUDIENCE: a laptop, Docker Compose
    try:
        ana = _address("ana.b")
        page = client.get("/", headers=_as(ana))             # the header means nothing here
        assert page.status_code == 302 and "/login" in page.headers["Location"]
        assert b'name="username"' in client.get("/login").data
        assert client.post("/view-as", data={"who": "maya"}).status_code == 404
        signed_in = client.post("/login", data={"username": "maya"})
        assert signed_in.status_code == 302 and _who(client) == MAYA
        with get_engine().connect() as c:
            assert c.execute(text("SELECT count(*) FROM users WHERE user_id = CAST(:u AS UUID)"),
                             {"u": _id(ana)}).scalar() == 0
    finally:
        stop_apps()


def test_a_vanished_account_is_made_again_on_the_same_request_with_no_redirect(hosted):
    """Nothing from the request's address goes into a Location header: there
    is none. The page asked for is the page answered."""
    client, visitor, started = hosted
    ana = visitor("ana.b")
    client.get("/", headers=_as(ana))
    delete_users([_id(ana)])
    with client.session_transaction() as session:
        session["user_id"] = _id(ana)                         # the cookie still names the account that is gone
    page = client.get("/accounts", headers=_as(ana))
    assert page.status_code == 200 and "Location" not in page.headers
    assert len(started) == 2


def test_a_request_without_the_header_says_what_it_did_carry_once(hosted, caplog):
    """Whoever sets this up needs to know whether the proxy passes the
    assertion on. The names of the headers that came, never their values."""
    client, _, _ = hosted
    with caplog.at_level("INFO", logger="banko.web"):
        client.get("/", headers={"X-Forwarded-Host": "banko.example.com", "X-Some-Token": "value-not-to-log"})
        client.get("/accounts", headers={"X-Forwarded-Host": "banko.example.com"})
    said = [r.getMessage() for r in caplog.records if "x-goog-iap-jwt-assertion" in r.getMessage()]
    assert len(said) == 1
    assert "x-forwarded-host" in said[0] and "x-some-token" in said[0]
    assert "value-not-to-log" not in said[0] and "banko.example.com" not in said[0]


def test_an_api_request_without_the_assertion_is_refused_not_served_as_the_default_persona(hosted):
    """/api/ asks nobody to sign in and answers for the default persona when
    there is no session. Behind IAP that would put one person's receipt on
    the persona everyone can view, the moment their sign-in failed once."""
    client, visitor, _ = hosted
    ana = visitor("ana.b")
    assert client.get("/", headers=_as(ana)).status_code == 200
    for method, path in (("get", "/api/v1/me"), ("get", "/api/v1/accounts"), ("get", "/api/coach/nudges"),
                         ("post", "/api/v1/receipts"), ("post", "/api/coach/chat"), ("get", "/cache-stats")):
        refused = getattr(client, method)(path)
        assert refused.status_code == 401, path
        assert refused.headers.get("X-Banko-User") is None, path
        assert MAYA not in refused.get_data(as_text=True)
    assert client.get("/api/health").status_code in (200, 503)       # health checks carry no person
    assert client.get("/api/v1/me", headers=_as(ana)).headers["X-Banko-User"] == _id(ana)


def test_a_third_account_planted_in_the_session_is_not_viewed(hosted):
    client, visitor, _ = hosted
    ana, ben = visitor("ana.b"), visitor("ben.c")
    client.get("/", headers=_as(ben))
    client.get("/", headers=_as(ana))
    with client.session_transaction() as session:           # ana's own cookie, with someone else's id in it
        session["user_id"] = _id(ben)
    assert client.get("/api/v1/me", headers=_as(ana)).headers["X-Banko-User"] == _id(ana)


def test_view_as_needs_the_assertion_too(hosted):
    client, visitor, _ = hosted
    ana = visitor("ana.b")
    client.get("/", headers=_as(ana))
    switched = client.post("/view-as", data={"who": "maya"})          # the cookie alone
    assert switched.status_code in (302, 303) and _who(client) is None


@pytest.mark.parametrize("headers, allowed", [
    ({"Sec-Fetch-Site": "same-origin"}, True),
    ({"Sec-Fetch-Site": "none"}, True),                          # typed or bookmarked
    ({}, True),                                                    # curl, the smoke: not a browser, nothing to forge
    ({"Sec-Fetch-Site": "cross-site"}, False),
    ({"Sec-Fetch-Site": "same-site"}, False),                     # another app on a sibling host
    ({"Origin": "https://elsewhere.example.org"}, False),         # a browser that sends no Sec-Fetch-Site
    ({"Origin": "null"}, False),
    ({"Origin": "https://banko.example.com", "X-Forwarded-Host": "banko.example.com"}, True),
])
def test_another_site_cannot_post_as_the_person_looking_at_it(hosted, headers, allowed):
    """The session cookie is not what signs anyone in here: the proxy's
    assertion is, and the browser sends the proxy's cookie with a form
    another site submits. So a request that changes something has to come
    from Banko's own pages."""
    client, visitor, _ = hosted
    ana = visitor("ana.b")
    client.get("/", headers=_as(ana))
    posted = client.post("/view-as", data={"who": "maya"}, headers={**_as(ana), **headers})
    if allowed:
        assert posted.status_code in (302, 303) and _who(client) == MAYA
    else:
        assert posted.status_code == 403 and _who(client) == _id(ana)
    reading = client.get("/api/v1/me", headers={**_as(ana), **headers})     # reading is never refused for this
    assert reading.status_code == 200


def test_someone_already_signed_in_stays_signed_in_while_the_database_is_away(hosted, monkeypatch):
    """A region outage on the chaos stage is the demo. The Platform page's
    live numbers never touch the database, and neither may signing in."""
    client, visitor, _ = hosted
    ana = visitor("ana.b")
    client.get("/", headers=_as(ana))
    client.post("/view-as", data={"who": "maya"}, headers=_as(ana))

    def down(*args, **kwargs):
        raise ConnectionError("connection refused")

    monkeypatch.setattr(staff, "get_engine", down)
    monkeypatch.setattr(staff, "run_transaction", down)
    live = client.get("/api/v1/platform/live", headers=_as(ana))
    assert live.status_code == 200
    assert _who(client) == MAYA                                    # still viewing who she was viewing


def test_a_first_visit_while_the_database_is_away_waits_instead_of_guessing(hosted, monkeypatch):
    """Someone this process has not seen, and no database to look them up
    in: not signed in as anyone, not signed out, and told to try again."""
    client, visitor, started = hosted
    ana, ben = visitor("ana.b"), visitor("ben.c")
    client.get("/", headers=_as(ana))

    def down(*args, **kwargs):
        raise ConnectionError("connection refused for ben.c at 10.0.0.7")

    monkeypatch.setattr(staff, "get_engine", down)
    page = client.get("/", headers=_as(ben))
    assert page.status_code == 503 and b"getting ready" in page.data and b"10.0.0.7" not in page.data
    api = client.get("/api/v1/me", headers=_as(ben))
    assert api.status_code == 503 and api.get_json()["error"] == "unavailable"
    assert api.headers.get("X-Banko-User") is None
    assert len(started) == 1                                       # only ana's


def test_the_old_agents_feed_is_not_served_behind_iap(hosted):
    """It lists every user's agent decisions, with the merchant and the
    amount in the reasoning."""
    client, visitor, _ = hosted
    ana = visitor("ana.b")
    for path in ("/api/agents/activity", "/api/agents/status", "/api/agents/stats"):
        assert client.get(path, headers=_as(ana)).status_code == 404, path


def test_a_history_that_failed_is_started_again_by_the_next_page_not_by_a_new_username(hosted):
    """Someone who signed up by name can start over under another name. A
    member of staff has one account: the page says to try again, and trying
    again is what a page load does, at most once a minute."""
    from banko_ai.services import seeding
    client, visitor, started = hosted
    ana = visitor("ana.b")
    client.get("/", headers=_as(ana))
    assert len(started) == 1
    seeding._report(_id(ana), 120, 600, "failed", None)
    page = client.get("/", headers=_as(ana))
    assert page.status_code == 200 and len(started) == 2
    html = page.get_data(as_text=True)
    assert "create a new one to start over" not in html and "Try again" in html
    client.get("/accounts", headers=_as(ana))
    client.get("/api/v1/me", headers=_as(ana))
    assert len(started) == 2                                       # not on every request


def test_an_idle_account_is_gone_after_the_boot_looked(monkeypatch):
    """At boot, and once a day while Banko stays up."""
    from banko_ai.web import iap
    monkeypatch.setattr(iap, "fetch_keys", lambda: {KEY.kid: KEY.public})
    idle = _address("long.gone")
    user_id = staff.ensure(subject(idle), idle)[0]["user_id"]
    with get_engine().begin() as c:
        c.execute(text("UPSERT INTO banko_meta (key, value, updated_at) VALUES (:k, 'seen', now() - INTERVAL '90 days')"),
                  {"k": f"seen:{user_id}"})
    try:
        client = app_client(monkeypatch, IAP_AUDIENCE=AUDIENCE)
        readiness = client.application.extensions["banko_readiness"]
        assert readiness.wait_for("demo_users", 60)
        with get_engine().connect() as c:
            assert c.execute(text("SELECT count(*) FROM users WHERE user_id = CAST(:u AS UUID)"),
                             {"u": user_id}).scalar() == 0
    finally:
        stop_apps()
        delete_users([user_id])
        staff.forget()


def test_no_address_reaches_the_log_when_an_account_cannot_be_read(hosted, monkeypatch, caplog):
    client, visitor, _ = hosted
    ben = visitor("ben.private")

    def down(*args, **kwargs):
        raise ConnectionError(f"could not look up {staff.username_for(ben)} ({_id(ben)})")

    monkeypatch.setattr(staff, "get_engine", down)
    with caplog.at_level("INFO"):
        client.get("/", headers=_as(ben))
    logged = " ".join(r.getMessage() + str(r.exc_text or "") for r in caplog.records)
    assert "ConnectionError" in logged
    assert "ben.private" not in logged and _id(ben) not in logged


def test_another_site_cannot_upload_a_receipt_into_someones_account(hosted):
    """A form upload needs no preflight, so the browser would send it."""
    import io
    client, visitor, _ = hosted
    ana = visitor("ana.b")
    client.get("/", headers=_as(ana))
    for path in ("/api/v1/receipts", "/api/upload-receipt"):
        sent = client.post(path, data={"receipt": (io.BytesIO(b"%PDF-1.4 not a receipt"), "r.pdf")},
                           content_type="multipart/form-data", headers={**_as(ana), "Sec-Fetch-Site": "cross-site"})
        assert sent.status_code == 403 and sent.get_json()["error"] == "cross_site", path


def test_banko_cannot_be_put_in_a_frame_on_another_site(hosted):
    """A page on another site can frame Banko, and the browser sends the
    proxy's cookie for the frame, so the person's own page is what loads in
    it. A click they are tricked into is then sent by Banko's own script, and
    counts as coming from Banko. So no site may frame it at all."""
    client, visitor, _ = hosted
    ana = visitor("ana.b")
    for response in (client.get("/", headers=_as(ana)), client.get("/login"), client.get("/api/v1/me", headers=_as(ana))):
        assert response.headers["X-Frame-Options"] == "DENY"
        assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]


def test_without_the_setting_no_header_about_frames_is_added(monkeypatch):
    client = app_client(monkeypatch)
    try:
        response = client.get("/login")
        assert "X-Frame-Options" not in response.headers and "Content-Security-Policy" not in response.headers
    finally:
        stop_apps()


_OPEN = ("/static/", "/favicon", "/api/health", "/health", "/api/cdc/")          # carry no person
_OWN_PAGES = ("/login", "/logout", "/signup", "/view-as")                         # say so themselves


def test_no_route_answers_for_anyone_without_the_assertion(hosted):
    """Every route the app has, asked with the cookie of someone who was
    signed in a moment ago and no assertion. A route added later is walked
    too: nothing here names the routes."""
    client, visitor, _ = hosted
    ana = visitor("ana.b")
    client.get("/", headers=_as(ana))
    asked = 0
    for rule in client.application.url_map.iter_rules():
        path = re.sub(r"<[^>]+>", "00000000-0000-0000-0000-0000000000a1", rule.rule)
        if path.startswith(_OPEN):
            continue
        for method in sorted(rule.methods - {"HEAD", "OPTIONS"}):
            with client.session_transaction() as session:           # the cookie, put back before each ask
                session["me"], session["user_id"], session["username"] = _id(ana), _id(ana), "ana.b"
            answer = client.open(path, method=method)
            asked += 1
            where = f"{method} {rule.rule} -> {answer.status_code}"
            assert answer.headers.get("X-Banko-User") is None, where
            assert _id(ana) not in answer.get_data(as_text=True) and MAYA not in answer.get_data(as_text=True), where
            if path.startswith(_OWN_PAGES):
                assert answer.status_code in (200, 302, 303, 405), where
                assert b'name="username"' not in answer.data, where
            elif method == "GET" and not path.startswith(("/api/", "/cache-")):
                assert answer.status_code in (302, 404) and (answer.status_code == 404 or "/login" in answer.headers["Location"]), where
            else:
                assert answer.status_code in (401, 404, 405), where
            assert _who(client) is None, where
    assert asked > 60                                                # the walk really walked


def test_the_forgeable_header_changes_nothing_beside_a_valid_assertion(hosted):
    client, visitor, _ = hosted
    ana, boss = visitor("ana.b"), visitor("the.boss")
    api = client.get("/api/v1/me", headers={**_as(ana), "x-goog-authenticated-user-email": f"accounts.google.com:{boss}",
                                            "x-goog-authenticated-user-id": f"accounts.google.com:{boss}"})
    assert api.headers["X-Banko-User"] == _id(ana)


@pytest.mark.parametrize("method", ["PUT", "PATCH", "DELETE", "POST"])
def test_every_method_that_changes_something_is_checked_for_where_it_came_from(hosted, method):
    client, visitor, _ = hosted
    ana = visitor("ana.b")
    client.get("/", headers=_as(ana))
    refused = client.open("/api/v1/budgets/Groceries", method=method, json={"amount": 1},
                          headers={**_as(ana), "Sec-Fetch-Site": "cross-site"})
    assert refused.status_code == 403 and refused.get_json()["error"] == "cross_site"


@pytest.mark.parametrize("origin, allowed", [
    ("http://localhost", True),                 # the app's own address, as the test client asks for it
    ("https://[::1", False),                    # not an address at all
    ("", False),
    ("banko.example.com", False),               # no scheme: not an origin
])
def test_an_origin_is_compared_with_the_apps_own_address(hosted, origin, allowed):
    client, visitor, _ = hosted
    ana = visitor("ana.b")
    client.get("/", headers=_as(ana))
    posted = client.post("/view-as", data={"who": "maya"}, headers={**_as(ana), "Origin": origin})
    assert posted.status_code == (302 if allowed else 403)


def test_the_boot_log_says_which_sign_in_is_on(monkeypatch, caplog):
    """A misspelt variable leaves the username page up on a hosted Banko, and
    nothing else would say so."""
    from banko_ai.web import iap
    monkeypatch.setattr(iap, "fetch_keys", lambda: {KEY.kid: KEY.public})
    try:
        with caplog.at_level("INFO"):
            app_client(monkeypatch, IAP_AUDIENCE=AUDIENCE)
        assert any("sign-in: from the Google identity" in r.getMessage() for r in caplog.records)
        caplog.clear()
        with caplog.at_level("INFO"):
            app_client(monkeypatch, IAP_AUDIENCE="   ")
        assert any("sign-in: a username, no password" in r.getMessage() for r in caplog.records)
    finally:
        stop_apps()


def test_an_account_the_google_sign_in_made_is_not_opened_by_typing_its_name(monkeypatch):
    """The setting off again, by choice or by a typo: the username page is
    back, and every account made for a person would be one typed name away."""
    email = _address("ana.b")
    user = staff.ensure(subject(email), email)[0]
    try:
        client = app_client(monkeypatch)                       # no IAP_AUDIENCE
        typed = client.post("/login", data={"username": user["username"]})
        assert _who(client) is None
        assert typed.status_code == 200 and b"Google" in typed.data
        assert client.post("/login", data={"username": "maya"}).status_code == 302 and _who(client) == MAYA
    finally:
        stop_apps()
        delete_users([user["user_id"]])
        staff.forget()


def test_the_old_nudge_list_survives_a_limit_that_is_no_number(monkeypatch):
    client = app_client(monkeypatch)
    try:
        assert client.get("/api/coach/nudges?limit=abc").status_code == 200
    finally:
        stop_apps()
