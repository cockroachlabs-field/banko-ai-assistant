"""Identity comes only from the assertion Google's Identity-Aware Proxy
signs. Anything wrong with it means nobody is signed in: never an error,
and never a guess."""

import base64
import json
import time

import pytest

from banko_ai.web import iap
from tests._iap import AUDIENCE, Key, claims, subject

KEY, OTHER, ROTATED = Key("key-1"), Key("key-1"), Key("key-2")


def _verifier(keys=None, **kw):
    fetched = []

    def fetch():
        fetched.append(time.time())
        found = keys() if callable(keys) else (keys if keys is not None else {KEY.kid: KEY.public})
        return dict(found)

    return iap.Verifier(AUDIENCE, fetch_keys=fetch, **kw), fetched


def _email(verifier, assertion):
    who = verifier.identity(assertion)
    return who.email if who else None


def _without_kid(token: str) -> str:
    """The same token under a header that names no key."""
    def part(value: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(value).encode()).rstrip(b"=").decode()
    return f"{part({'alg': 'ES256', 'typ': 'JWT'})}.{token.split('.', 1)[1]}"


def _unsigned(payload: dict) -> str:
    def part(value: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(value).encode()).rstrip(b"=").decode()
    return f"{part({'alg': 'none', 'typ': 'JWT', 'kid': KEY.kid})}.{part(payload)}."


@pytest.mark.parametrize("why, assertion, email", [
    ("a valid assertion", lambda: KEY.sign(claims()), "ana.b@example.com"),
    ("the email is read as Google sent it, lowered", lambda: KEY.sign(claims("Ana.B@Example.com")), "ana.b@example.com"),
    ("made for another service", lambda: KEY.sign(claims(aud=AUDIENCE.replace("banko-demo", "dojo"))), None),
    ("no audience at all", lambda: KEY.sign(claims(aud=None)), None),
    ("not issued by IAP", lambda: KEY.sign(claims(iss="https://accounts.google.com")), None),
    ("expired", lambda: KEY.sign(claims(iat=int(time.time()) - 900, exp=int(time.time()) - 300)), None),
    ("signed by someone else's key under the same id", lambda: OTHER.sign(claims()), None),
    ("no email in it", lambda: KEY.sign(claims(email="x@example.com") | {"email": None}), None),
    ("no account identifier in it", lambda: KEY.sign(claims(sub=None)), None),
    ("no issuer", lambda: KEY.sign(claims(iss=None)), None),
    ("no expiry", lambda: KEY.sign(claims(exp=None)), None),
    ("no issue time", lambda: KEY.sign(claims(iat=None)), None),
    ("no key id", lambda: _without_kid(KEY.sign(claims())), None),
    ("an expiry that is not a time", lambda: KEY.sign(claims(exp=float("inf"))), None),
    ("valid for a year, which IAP never signs", lambda: KEY.sign(claims(exp=int(time.time()) + 365 * 86400)), None),
    ("an email that is not one", lambda: KEY.sign(claims() | {"email": "not-an-address"}), None),
    ("unsigned", lambda: _unsigned(claims()), None),
    ("the header missing", lambda: None, None),
    ("the header empty", lambda: "", None),
    ("not a token", lambda: "x-goog-authenticated-user-email: ana.b@example.com", None),
    ("three parts of nothing", lambda: "a.b.c", None),
])
def test_only_a_valid_assertion_names_anyone(why, assertion, email):
    verifier, _ = _verifier()
    assert _email(verifier, assertion()) == email, why


def test_a_slow_clock_does_not_sign_anyone_out():
    verifier, _ = _verifier()
    now = int(time.time())
    assert _email(verifier, KEY.sign(claims(iat=now + 20, exp=now + 620))) == "ana.b@example.com"   # issued "in the future"
    assert _email(verifier, KEY.sign(claims(iat=now - 620, exp=now - 20))) == "ana.b@example.com"   # expired seconds ago
    assert _email(verifier, KEY.sign(claims(iat=now - 900, exp=now - 120))) is None                 # expired minutes ago


def test_a_key_google_rotated_in_is_fetched_once():
    published = {KEY.kid: KEY.public}
    verifier, fetched = _verifier(lambda: published)
    assert _email(verifier, KEY.sign(claims())) == "ana.b@example.com" and len(fetched) == 1
    published[ROTATED.kid] = ROTATED.public                      # Google publishes a new key
    assert _email(verifier, ROTATED.sign(claims("ben@example.com"))) == "ben@example.com"
    assert len(fetched) == 2                                     # the unknown key id asked for the keys again
    assert _email(verifier, ROTATED.sign(claims("cy@example.com"))) == "cy@example.com" and len(fetched) == 2


def test_made_up_key_ids_do_not_send_banko_to_google_on_every_request():
    """An unknown key id fetches the keys again, at most once a minute:
    anyone can put any id in a header."""
    verifier, fetched = _verifier()
    stranger = Key("never-published")
    for _ in range(25):
        assert _email(verifier, stranger.sign(claims())) is None
    assert len(fetched) == 2          # the first load, and one more look for the unknown id


def test_the_keys_are_kept_for_hours_and_then_read_again():
    clock = [time.time()]
    verifier, fetched = _verifier(now=lambda: clock[0], elapsed=lambda: clock[0])
    token = KEY.sign(claims(iat=int(clock[0]) - 5, exp=int(clock[0]) + 600))
    assert _email(verifier, token) == "ana.b@example.com" and len(fetched) == 1
    clock[0] += iap.KEYS_TTL_SECONDS - 60
    later = KEY.sign(claims(iat=int(clock[0]) - 5, exp=int(clock[0]) + 600))
    assert _email(verifier, later) == "ana.b@example.com" and len(fetched) == 1
    clock[0] += 120
    last = KEY.sign(claims(iat=int(clock[0]) - 5, exp=int(clock[0]) + 600))
    assert _email(verifier, last) == "ana.b@example.com" and len(fetched) == 2


def test_google_out_of_reach_means_not_signed_in_not_an_error():
    def down():
        raise ConnectionError("no route to gstatic.com")

    verifier = iap.Verifier(AUDIENCE, fetch_keys=down)
    assert _email(verifier, KEY.sign(claims())) is None


def test_keys_already_held_keep_working_when_google_cannot_be_reached():
    answers = [{KEY.kid: KEY.public}]
    clock = [time.time()]

    def fetch():
        if not answers:
            raise ConnectionError("no route to gstatic.com")
        return answers.pop()

    verifier = iap.Verifier(AUDIENCE, fetch_keys=fetch, now=lambda: clock[0], elapsed=lambda: clock[0])
    assert _email(verifier, KEY.sign(claims(iat=int(clock[0]) - 5, exp=int(clock[0]) + 600))) == "ana.b@example.com"
    clock[0] += iap.KEYS_TTL_SECONDS + 60                         # due for a refresh, and Google is down
    assert _email(verifier, KEY.sign(claims(iat=int(clock[0]) - 5, exp=int(clock[0]) + 600))) == "ana.b@example.com"


def test_an_assertion_seen_before_is_not_verified_again_until_it_expires(monkeypatch):
    verifier, _ = _verifier()
    checked = []
    real = iap.jwt.decode
    monkeypatch.setattr(iap.jwt, "decode", lambda *a, **k: (checked.append(1), real(*a, **k))[1])
    token = KEY.sign(claims())
    assert [_email(verifier, token) for _ in range(5)] == ["ana.b@example.com"] * 5
    assert len(checked) == 1
    clock = [time.time()]
    verifier, _ = _verifier(now=lambda: clock[0], elapsed=lambda: clock[0])
    short = KEY.sign(claims(iat=int(clock[0]) - 5, exp=int(clock[0]) + 60))
    assert _email(verifier, short) == "ana.b@example.com"
    clock[0] += 600
    assert _email(verifier, short) is None                           # remembered, but past its expiry


def test_the_mode_is_off_without_an_audience():
    assert iap.Verifier("", fetch_keys=dict).identity(KEY.sign(claims())) is None


def test_an_assertion_issued_in_the_future_is_refused():
    verifier, _ = _verifier()
    now = int(time.time())
    assert _email(verifier, KEY.sign(claims(iat=now + 3600, exp=now + 4200))) is None


def test_an_assertion_under_another_algorithm_is_refused_for_that(caplog):
    def part(value: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(value).encode()).rstrip(b"=").decode()

    signed = KEY.sign(claims())
    other = f"{part({'alg': 'RS256', 'typ': 'JWT', 'kid': KEY.kid})}.{signed.split('.', 2)[1]}.{signed.split('.', 2)[2]}"
    verifier, fetched = _verifier()
    with caplog.at_level("INFO", logger="banko.web"):
        assert _email(verifier, other) is None
    assert "ES256" in " ".join(r.getMessage() for r in caplog.records)
    assert fetched == []                               # turned down before Google is asked for anything


def test_with_no_keys_and_google_out_of_reach_one_request_asks_and_the_rest_are_told_at_once():
    """A cold start with no way out to Google. Every request used to ask for
    itself, one after the other, each for the length of a timeout."""
    import threading
    asked, clock = [], [time.time()]

    def down():
        asked.append(1)
        time.sleep(0.3)
        raise ConnectionError("no route to gstatic.com")

    verifier = iap.Verifier(AUDIENCE, fetch_keys=down, now=lambda: clock[0], elapsed=lambda: clock[0])
    token = KEY.sign(claims())
    answers: list = []
    threads = [threading.Thread(target=lambda: answers.append(_email(verifier, token))) for _ in range(8)]
    started = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert answers == [None] * 8 and len(asked) == 1
    assert time.monotonic() - started < 2
    assert _email(verifier, token) is None and len(asked) == 1          # still within the pause
    clock[0] += iap.RETRY_SECONDS + 1
    assert _email(verifier, token) is None and len(asked) == 2          # and then one more look


def test_an_empty_answer_from_google_does_not_replace_the_keys_in_hand():
    answers, clock = [{KEY.kid: KEY.public}, {}, {}], [time.time()]
    asked = []

    def fetch():
        asked.append(1)
        return answers.pop(0)

    def token():
        return KEY.sign(claims(iat=int(clock[0]) - 5, exp=int(clock[0]) + 600))

    verifier = iap.Verifier(AUDIENCE, fetch_keys=fetch, now=lambda: clock[0], elapsed=lambda: clock[0])
    assert _email(verifier, token()) == "ana.b@example.com"
    clock[0] += iap.KEYS_TTL_SECONDS + 60
    assert _email(verifier, token()) == "ana.b@example.com" and len(asked) == 2     # asked, and got nothing usable
    clock[0] += 1
    assert _email(verifier, token()) == "ana.b@example.com" and len(asked) == 2     # not asked again a second later


@pytest.mark.parametrize("why", ["the keys are hours old", "someone sent a made-up key id"])
def test_a_look_at_googles_keys_holds_up_nobody_the_keys_in_hand_can_answer(why):
    """One request reads the keys again. Until it is back, every other
    request with an assertion the held keys can check goes on with those."""
    import threading
    clock, slow, entered = [time.time()], threading.Event(), threading.Event()
    first = [True]

    def fetch():
        if first[0]:
            first[0] = False
            return {KEY.kid: KEY.public}
        entered.set()
        slow.wait(10)
        return {KEY.kid: KEY.public}

    def token(email: str) -> str:
        return KEY.sign(claims(email, iat=int(clock[0]) - 5, exp=int(clock[0]) + 600))

    verifier = iap.Verifier(AUDIENCE, fetch_keys=fetch, now=lambda: clock[0], elapsed=lambda: clock[0])
    assert _email(verifier, token("ana.b@example.com")) == "ana.b@example.com"
    if why == "the keys are hours old":
        clock[0] += iap.KEYS_TTL_SECONDS + 60
        trigger = token("cy@example.com")
    else:
        trigger = Key("never-published").sign(claims())
    looking = threading.Thread(target=verifier.identity, args=(trigger,))
    looking.start()
    try:
        assert entered.wait(5)
        started = time.monotonic()
        assert _email(verifier, token("ben@example.com")) == "ben@example.com"
        assert time.monotonic() - started < 1
    finally:
        slow.set()
        looking.join(timeout=10)


def test_what_came_in_the_header_never_reaches_the_log(caplog):
    """google-auth quotes the token back in some of its errors, and the
    token's payload is the person's email in base64."""
    def part(value: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(value).encode()).rstrip(b"=").decode()

    private = "someone.private@example.com"
    payload = part({"email": private, "aud": AUDIENCE})
    sent = [f"{part({'alg': 'ES256', 'kid': KEY.kid})}.{payload}",          # two parts where three belong
            f"accounts.google.com:{private}",                               # the other header's value, pasted here
            f"{part({'alg': 'made up ' + private, 'kid': KEY.kid})}.{payload}.c2ln"]
    with caplog.at_level("INFO", logger="banko.web"):
        for value in sent:
            verifier, _ = _verifier()                      # each says its one line
            assert _email(verifier, value) is None
    logged = " ".join(r.getMessage() for r in caplog.records)
    assert logged.count("not accepted") == len(sent)
    assert private not in logged and payload not in logged and "someone" not in logged


def test_an_assertion_names_the_google_account_and_the_address():
    """The account is keyed on what Google calls the unique, stable
    identifier for the user. The address is what the username is made of."""
    verifier, _ = _verifier()
    who = verifier.identity(KEY.sign(claims("Ana.B@Example.com")))
    assert who == iap.Identity(subject("ana.b@example.com"), "ana.b@example.com")
    moved = verifier.identity(KEY.sign(claims("ana.b@example.com", sub="accounts.google.com:77")))
    assert moved.email == who.email and moved.subject != who.subject       # the address, with someone else behind it


def test_the_clock_being_set_back_does_not_stop_the_keys_being_read():
    """How long ago the keys were read, or a read failed, is counted on a
    clock that only runs forward. Counted on the wall clock, a step back of
    six hours after one failed read left nobody able to sign in for six."""
    wall, ticks, asked = [time.time()], [1000.0], []

    def fetch():
        asked.append(1)
        if len(asked) == 1:
            raise ConnectionError("no route to gstatic.com")
        return {KEY.kid: KEY.public}

    def token():
        return KEY.sign(claims(iat=int(wall[0]) - 5, exp=int(wall[0]) + 600))

    verifier = iap.Verifier(AUDIENCE, fetch_keys=fetch, now=lambda: wall[0], elapsed=lambda: ticks[0])
    assert _email(verifier, token()) is None and len(asked) == 1
    wall[0] -= 6 * 3600                               # the machine's clock is corrected backwards
    ticks[0] += iap.RETRY_SECONDS + 1                 # and a little time passes
    assert _email(verifier, token()) == "ana.b@example.com" and len(asked) == 2


def test_each_reason_for_turning_an_assertion_down_is_said_and_none_crowds_out_another(caplog):
    """One line a minute for all reasons let junk traffic take the line that
    would have named the real problem."""
    verifier, _ = _verifier()
    with caplog.at_level("INFO", logger="banko.web"):
        for _ in range(3):
            assert _email(verifier, "not.a.token") is None
            assert _email(verifier, KEY.sign(claims(aud="/projects/1/locations/x/services/other"))) is None
    said = [r.getMessage() for r in caplog.records if "not accepted" in r.getMessage()]
    assert len(said) == 2 and any("IAP_AUDIENCE" in line for line in said)


def test_with_no_keys_read_yet_the_reason_given_is_the_keys(caplog):
    def down():
        raise ConnectionError("no route to gstatic.com")

    verifier = iap.Verifier(AUDIENCE, fetch_keys=down)
    with caplog.at_level("INFO", logger="banko.web"):
        assert _email(verifier, KEY.sign(claims())) is None           # this one asks Google and fails
        assert _email(verifier, KEY.sign(claims("ben@example.com"))) is None   # this one is told at once
    said = " ".join(r.getMessage() for r in caplog.records)
    assert "keys could not be read" in said and "does not publish" not in said
