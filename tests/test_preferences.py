"""A person's own settings for the assistant.

The deployment's settings say how Banko behaves for everyone, and changing
one means a new environment and a restart. A few of them shape the answers
one person gets and are read when a question is asked: how the answer cache
matches questions, how many purchases a search page holds, what the
assistant remembers. Those a person sets for themselves on the Settings
page. Their own account keeps them, so they
follow the person to whichever persona they view, and nobody else's
questions are answered differently for it."""

import os
import uuid

import pytest
from sqlalchemy import text

from banko_ai.config.settings import Settings, get_config, reset_config
from banko_ai.db.engine import get_engine
from banko_ai.services import preferences as P
from tests._app import app_client, stop_apps
from tests._money import drop_user, make_user

needs_db = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")
KEYS = [s.key for s in P.SETTINGS]


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    stop_apps()


@pytest.fixture
def user():
    uid = make_user()
    yield uid
    drop_user(uid)


def _stored(uid: str):
    with get_engine().connect() as c:
        return c.execute(text("SELECT preferences FROM users WHERE user_id = CAST(:u AS UUID)"), {"u": uid}).scalar()


# What can be set.

def test_every_personal_setting_stands_in_for_one_of_the_deployments():
    """A person overrides a setting the deployment has; there is no second
    list of defaults to drift from the first."""
    assert KEYS and len(set(KEYS)) == len(KEYS)
    assert set(KEYS) <= set(Settings.model_fields)


def test_nothing_secret_or_about_the_deployment_itself_is_offered():
    """Only what shapes one person's answers. Keys, connections, pools, the
    model and whatever is read once at boot stay with whoever runs Banko.
    So does how long a cached answer or search is kept: a persona's cache
    is shared, and one person's lifetime would decide how old an answer the
    next person gets."""
    offered = {Settings.model_fields[k].json_schema_extra["group"] for k in KEYS}
    assert offered <= {"cache", "assistant"}
    assert not any(Settings.model_fields[k].json_schema_extra["secret"] for k in KEYS)
    assert not {"cache_ttl_hours", "search_cache_ttl_minutes"} & set(KEYS)


def test_without_a_choice_of_ones_own_the_deployments_settings_apply():
    mine, cfg = P.defaults(), get_config()
    for key in KEYS:
        assert getattr(mine, key) == getattr(cfg, key), key
    assert mine.own == {}
    with pytest.raises(AttributeError):
        _ = mine.database_url                       # not a personal setting: nothing else is reachable through it


def test_every_setting_says_what_it_is_in_words_the_page_can_show():
    listed = P.describe(P.defaults())
    assert [s["key"] for s in listed["settings"]] == KEYS
    assert listed["groups"] == list(dict.fromkeys(s["group"] for s in listed["settings"]))
    for s in listed["settings"]:
        assert s["label"] and s["help"] and s["kind"] in ("choice", "number", "whole", "flag"), s
        assert s["value"] == s["default"] and s["changed"] is False
        if s["kind"] == "choice":
            assert s["default"] in [c["value"] for c in s["choices"]]
            assert all(c["label"] and c["help"] for c in s["choices"])        # the page explains the one chosen
        if s["kind"] in ("number", "whole"):
            assert s["min"] <= s["default"] <= s["max"], s
    similarity = next(s for s in listed["settings"] if s["key"] == "cache_similarity_threshold")
    assert similarity["only_with"] == {"key": "cache_match", "value": "similar"}    # the page greys it out otherwise


def test_the_settings_that_shape_an_answer_have_one_fingerprint():
    """A page of three purchases, or one memory instead of five, makes
    another answer. Cached answers are kept and looked up by this, so
    nobody is served one that other settings shaped. How questions are
    matched shapes no answer."""
    usual = P.defaults().shape
    assert P.Preferences({"cache_match": "similar", "cache_similarity_threshold": 0.6,
                          "assistant_implicit_memory": False}).shape == usual
    apart = {usual, P.Preferences({"search_page_size": 3}).shape,
             P.Preferences({"assistant_memory_recall_k": 1}).shape,
             P.Preferences({"assistant_memory_max_distance": 0.3}).shape}
    assert len(apart) == 4
    assert P.Preferences({"search_page_size": 3}).shape == P.Preferences({"search_page_size": 3}).shape


# Saving and reading.

@needs_db
def test_a_person_sets_their_own_and_keeps_the_rest(user):
    saved = P.save(user, {"cache_match": "similar", "cache_similarity_threshold": 0.6, "search_page_size": 5,
                          "assistant_implicit_memory": False})
    again = P.of(user)
    for mine in (saved, again):
        assert (mine.cache_match, mine.cache_similarity_threshold, mine.search_page_size,
                mine.assistant_implicit_memory) == ("similar", 0.6, 5, False)
        assert mine.assistant_memory_recall_k == get_config().assistant_memory_recall_k   # never touched
    changed = {s["key"] for s in P.describe(again)["settings"] if s["changed"]}
    assert changed == {"cache_match", "cache_similarity_threshold", "search_page_size", "assistant_implicit_memory"}


@needs_db
def test_only_what_differs_from_the_deployment_is_stored(user):
    cfg = get_config()
    P.save(user, {key: getattr(cfg, key) for key in KEYS})
    assert _stored(user) is None
    P.save(user, {**{key: getattr(cfg, key) for key in KEYS}, "search_page_size": 7})
    assert _stored(user) == {"search_page_size": 7}


@needs_db
def test_saving_replaces_what_was_set_and_saving_nothing_goes_back_to_the_deployments(user):
    P.save(user, {"search_page_size": 7, "cache_match": "exact"})
    P.save(user, {"cache_match": "similar"})
    assert P.of(user).own == {"cache_match": "similar"}
    P.save(user, {})
    assert P.of(user).own == {} and _stored(user) is None


@needs_db
def test_a_deployment_setting_that_changes_reaches_everyone_who_never_chose(user, monkeypatch):
    other = make_user()
    try:
        P.save(other, {"search_page_size": 3})
        monkeypatch.setenv("SEARCH_PAGE_SIZE", "7")
        reset_config()
        assert P.of(user).search_page_size == 7
        assert P.of(other).search_page_size == 3
    finally:
        monkeypatch.undo()
        reset_config()
        drop_user(other)


@needs_db
def test_one_persons_settings_are_nobody_elses(user):
    other = make_user()
    try:
        P.save(user, {"cache_match": "exact"})
        assert P.of(other).own == {}
    finally:
        drop_user(other)


@needs_db
@pytest.mark.parametrize("values, key, said", [
    ({"no_such_setting": 1}, "no_such_setting", "no setting"),
    ({"database_url": "postgresql://elsewhere"}, "database_url", "no setting"),
    ({"cache_match": "fuzzy"}, "cache_match", "one of"),
    ({"cache_similarity_threshold": 0.2}, "cache_similarity_threshold", "between 0.5 and 0.99"),
    ({"cache_similarity_threshold": "0.6"}, "cache_similarity_threshold", "number"),
    ({"cache_similarity_threshold": True}, "cache_similarity_threshold", "number"),
    ({"search_page_size": 0}, "search_page_size", "between 1 and 25"),
    ({"search_page_size": 26}, "search_page_size", "between 1 and 25"),
    ({"search_page_size": 5.5}, "search_page_size", "whole number"),
    ({"assistant_memory_recall_k": 0}, "assistant_memory_recall_k", "between 1 and 20"),
    ({"search_page_size": float("inf")}, "search_page_size", "whole number"),
    ({"search_page_size": 10 ** 400}, "search_page_size", "between 1 and 25"),
    ({"cache_similarity_threshold": 10 ** 400}, "cache_similarity_threshold", "number"),
    ({"cache_similarity_threshold": float("nan")}, "cache_similarity_threshold", "number"),
    ({"cache_similarity_threshold": float("-inf")}, "cache_similarity_threshold", "number"),
    ({"assistant_implicit_memory": "yes"}, "assistant_implicit_memory", "on or off"),
    ({"assistant_implicit_memory": 1}, "assistant_implicit_memory", "on or off"),
])
def test_a_value_that_does_not_fit_is_refused_in_a_sentence_and_nothing_is_saved(user, values, key, said):
    P.save(user, {"search_page_size": 7})
    with pytest.raises(P.Invalid) as refused:
        P.save(user, {"cache_match": "exact", **values})          # one good value beside the bad one
    assert refused.value.key == key and said in str(refused.value), str(refused.value)
    assert P.of(user).own == {"search_page_size": 7}


@needs_db
def test_whole_numbers_may_arrive_as_floats(user):
    """A browser's number box sends 5 as 5, a script may send 5.0."""
    assert P.save(user, {"search_page_size": 5.0}).search_page_size == 5
    assert _stored(user) == {"search_page_size": 5}


@needs_db
def test_a_stored_value_that_no_longer_fits_is_passed_over(user):
    """A build that narrows a range, or drops a setting, must not break the
    questions of whoever had set it."""
    with get_engine().begin() as c:
        c.execute(text("UPDATE users SET preferences = CAST(:p AS JSONB) WHERE user_id = CAST(:u AS UUID)"),
                  {"u": user, "p": '{"search_page_size": 500, "gone_setting": 1, "cache_match": "similar", '
                                   '"assistant_implicit_memory": "maybe"}'})
    mine = P.of(user)
    assert mine.own == {"cache_match": "similar"}
    assert mine.search_page_size == get_config().search_page_size


@needs_db
def test_an_account_that_is_not_there_has_the_deployments_settings_and_cannot_save():
    nobody = str(uuid.uuid4())
    assert P.of(nobody).own == {} and P.of("not-a-uuid").own == {}
    with pytest.raises(LookupError):
        P.save(nobody, {"cache_match": "exact"})


# A persona is everybody's.

@needs_db
def test_a_personas_settings_go_back_to_the_deployments_when_banko_boots(user):
    """With no sign-in in front, whoever picks Maya is Maya. What one
    visitor set must not shape the next demo: boot makes a persona what its
    file says, settings included. An account of one's own keeps its
    settings."""
    from banko_ai.config.catalog import get_catalog
    from banko_ai.services import seeding
    persona = get_catalog().personas[0].user_id
    seeding.ensure_personas()                        # a database nobody booted on yet has no personas
    try:
        P.save(persona, {"cache_match": "similar", "cache_similarity_threshold": 0.5})
        P.save(user, {"search_page_size": 7})
        seeding.ensure_personas()
        assert P.of(persona).own == {}
        assert P.of(user).own == {"search_page_size": 7}
    finally:
        P.save(persona, {})


@needs_db
def test_resetting_an_account_resets_its_settings_too():
    """Reset gives an account back as new: its history, and how its assistant is set."""
    from banko_ai.services import seeding
    uid = make_user("balanced")
    try:
        P.save(uid, {"search_page_size": 7})
        assert seeding.reset_user(uid, months=1) > 0
        assert P.of(uid).own == {}
    finally:
        drop_user(uid)


# The API the Settings page reads.

@needs_db
def test_the_api_lists_saves_and_resets(user, monkeypatch):
    client = app_client(monkeypatch)
    with client.session_transaction() as s:
        s["user_id"] = user
    listed = client.get("/api/v1/me/settings").get_json()
    assert [s["key"] for s in listed["settings"]] == KEYS and not any(s["changed"] for s in listed["settings"])
    saved = client.post("/api/v1/me/settings", json={"values": {"cache_match": "similar",
                                                                "cache_similarity_threshold": 0.6}})
    assert saved.status_code == 200, saved.get_data(as_text=True)
    by_key = {s["key"]: s for s in saved.get_json()["settings"]}
    assert by_key["cache_match"]["value"] == "similar" and by_key["cache_match"]["changed"]
    assert by_key["cache_similarity_threshold"]["value"] == 0.6
    assert {s["key"]: s["value"] for s in client.get("/api/v1/me/settings").get_json()["settings"]}["cache_match"] \
        == "similar"
    assert P.of(user).cache_match == "similar"
    reset = client.post("/api/v1/me/settings", json={"values": {}})
    assert reset.status_code == 200 and not any(s["changed"] for s in reset.get_json()["settings"])
    assert P.of(user).own == {}
    # With no sign-in in front, the settings are the account's: the page does not promise more.
    page = client.get("/settings").get_data(as_text=True)
    assert "Kept with this account" in page and "Yours alone" not in page
    # A persona is everybody's, and starts every boot as its file describes it.
    from banko_ai.config.catalog import get_catalog
    with client.session_transaction() as s:
        s["user_id"] = get_catalog().personas[0].user_id
    assert "Kept with this persona until Banko restarts" in client.get("/settings").get_data(as_text=True)


@needs_db
def test_the_api_refuses_a_bad_value_in_words_the_page_can_show(user, monkeypatch):
    client = app_client(monkeypatch)
    with client.session_transaction() as s:
        s["user_id"] = user
    refused = client.post("/api/v1/me/settings", json={"values": {"search_page_size": 99}})
    assert refused.status_code == 400
    assert refused.get_json() == {"error": "Purchases per page takes a whole number between 1 and 25.",
                                  "key": "search_page_size"}
    assert client.post("/api/v1/me/settings", json={"values": [1, 2]}).status_code == 400
    assert client.post("/api/v1/me/settings", json={}).status_code == 400            # say which values
    assert P.of(user).own == {}


@needs_db
def test_settings_belong_to_the_person_not_to_the_account_on_screen(monkeypatch):
    """Signed in through Google and viewing Maya: the settings are one's own.
    Maya's row is everybody's, and one person's experiment must not change
    the answers a colleague gets while showing her."""
    from banko_ai.services import staff
    from banko_ai.services.users import delete_users
    from banko_ai.web import iap
    from tests._iap import AUDIENCE, Key, claims, subject
    key, maya = Key("iap-key"), "00000000-0000-0000-0000-0000000000a1"
    monkeypatch.setattr(iap, "fetch_keys", lambda: {key.kid: key.public})
    client = app_client(monkeypatch, IAP_AUDIENCE=AUDIENCE)
    client.application.extensions["banko_start_account"] = lambda user: None      # no history needed here
    email = f"pat-{uuid.uuid4().hex[:8]}@example.com"
    me, signed = staff.user_id_for(subject(email)), {"x-goog-iap-jwt-assertion": key.sign(claims(email))}
    try:
        assert client.get("/", headers=signed).status_code == 200                  # the first visit makes the account
        client.post("/view-as", data={"who": "maya"}, headers=signed)
        with client.session_transaction() as session:
            assert session["user_id"] == maya and session["me"] == me
        saved = client.post("/api/v1/me/settings", json={"values": {"cache_match": "exact"}}, headers=signed)
        assert saved.status_code == 200, saved.get_data(as_text=True)
        assert P.of(me).own == {"cache_match": "exact"} and P.of(maya).own == {}
        listed = client.get("/api/v1/me/settings", headers=signed).get_json()
        assert {s["key"]: s["value"] for s in listed["settings"]}["cache_match"] == "exact"
        assert "Yours alone" in client.get("/settings", headers=signed).get_data(as_text=True)
    finally:
        stop_apps()
        delete_users([me])
        staff.forget()
        P.save(maya, {})        # had it gone to her after all, the next run starts clean
