"""The Platform API: signed in only, /live from memory, cluster and data
cards from snapshots that never make a request wait, load controls with
plain errors, chaos proxied with the plane's warnings, a read-only switch,
and node changes that become events."""

import os
import time

import pytest

from banko_ai.clients.chaos import ChaosConfirmRequired, ChaosUnavailable
from banko_ai.observability.events import events
from banko_ai.web import platform_api as PA
from banko_ai.workers.traffic import SimulatorBusy
from tests._app import app_client

pytestmark = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")
MAYA = "00000000-0000-0000-0000-0000000000a1"


def _signed_in(monkeypatch, **env):
    client = app_client(monkeypatch, **env)
    with client.session_transaction() as s:
        s["user_id"], s["username"] = MAYA, "maya"
    return client


class FakeSim:
    def __init__(self):
        self.state, self.calls, self.config = "idle", [], None

    def status(self):
        return {"state": self.state, "config": self.config.as_dict() if self.config else None, "error": None,
                "started_at": None, "elapsed_s": 0.0, "users_ready": 0, "serving": {}}

    def start(self, cfg):
        if self.state != "idle":
            raise SimulatorBusy("A load run is already going; stop it first")
        self.state, self.config = "running", cfg
        self.calls.append(("start", cfg))
        return self.status()

    def apply(self, cfg):
        self.config = cfg
        self.calls.append(("apply", cfg))
        return self.status()

    def stop(self, timeout=5.0):
        self.state = "idle"
        self.calls.append(("stop",))
        return self.status()


class FakePlane:
    def __init__(self, state=None, act=None, slow=0.0):
        self._state, self._act, self.slow, self.acts = state, act, slow, []

    def state(self):
        if self.slow:
            time.sleep(self.slow)
            raise ChaosUnavailable("The chaos control plane did not answer.")
        return self._state

    def act(self, action, kind, value="", *, confirm=False):
        self.acts.append((action, kind, value, confirm))
        return self._act(confirm)

    def latency(self, setting):
        return {"latency": setting}


@pytest.fixture
def fake_sim(monkeypatch):
    sim = FakeSim()
    monkeypatch.setattr(PA, "get_simulator", lambda: sim)
    return sim


def test_platform_needs_a_signed_in_session(monkeypatch):
    assert app_client(monkeypatch).get("/api/v1/platform/live").status_code == 401


def test_live_never_touches_the_database(monkeypatch, fake_sim):
    client = _signed_in(monkeypatch)

    def down(*args, **kwargs):
        raise RuntimeError("the database is down")

    import banko_ai.db.engine as engine_mod
    monkeypatch.setattr(engine_mod, "get_engine", down)
    monkeypatch.setattr(PA.platform, "get_engine", down)
    r = client.get("/api/v1/platform/live")
    assert r.status_code == 200
    body = r.get_json()
    assert {"load", "summary", "ops", "regions", "series", "http", "db", "events", "controls", "as_of"} <= set(body)
    assert body["controls"] == {"load": True, "chaos": False, "defaults": {"users": 100, "rate": 50, "workers": 16},
                                "limits": {"users": 2000, "rate": 2000, "workers": 64}}


def test_load_controls_validate_and_start_once(monkeypatch, fake_sim):
    client = _signed_in(monkeypatch)
    bad = client.post("/api/v1/platform/load", json={"action": "start", "rate": "abc"})
    assert bad.status_code == 400 and bad.get_json() == {"error": "The rate must be a number", "code": "bad_request"}
    assert client.post("/api/v1/platform/load", json={"action": "dance"}).status_code == 400
    ok = client.post("/api/v1/platform/load", json={"action": "start", "users": 50, "rate": 20, "workers": 4})
    assert ok.status_code == 202 and ok.get_json()["state"] == "running"
    assert client.post("/api/v1/platform/load", json={"action": "start"}).status_code == 409
    applied = client.post("/api/v1/platform/load", json={"action": "apply", "rate": 80})
    assert applied.status_code == 200 and fake_sim.calls[-1][1].users == 50 and fake_sim.calls[-1][1].rate == 80
    assert client.post("/api/v1/platform/load", json={"action": "stop"}).get_json()["state"] == "idle"


def test_controls_off_is_read_only(monkeypatch, fake_sim):
    client = _signed_in(monkeypatch, PLATFORM_CONTROLS="false")
    assert client.post("/api/v1/platform/load", json={"action": "start"}).status_code == 403
    live = client.get("/api/v1/platform/live")
    assert live.status_code == 200 and live.get_json()["controls"]["load"] is False


def test_writes_need_json(monkeypatch, fake_sim):
    client = _signed_in(monkeypatch)
    assert client.post("/api/v1/platform/load", data="action=start").status_code == 400


def test_chaos_without_a_plane_is_404_and_the_cluster_card_has_no_chaos(monkeypatch):
    client = _signed_in(monkeypatch, CHAOS_CONTROL_URL="")
    assert client.post("/api/v1/platform/chaos", json={"action": "outage", "kind": "region", "value": "x"}).status_code == 404
    body = client.get("/api/v1/platform/cluster").get_json()
    assert body["chaos"] is None and body["topology"]["source"] in ("sql", "status_api")
    # The test database's derived console address is another cluster's: never trusted.
    assert body["topology"]["source"] == "sql"


def test_a_slow_plane_does_not_hold_the_cluster_card(monkeypatch):
    client = _signed_in(monkeypatch)
    monkeypatch.setattr(PA, "chaos_client", lambda: FakePlane(slow=3.0))
    started = time.monotonic()
    body = client.get("/api/v1/platform/cluster").get_json()
    assert time.monotonic() - started < 4 and body["chaos"] is None


def test_a_disruptive_action_needing_confirm_is_409_with_the_warning(monkeypatch):
    client = _signed_in(monkeypatch)

    def act(confirm):
        if not confirm:
            raise ChaosConfirmRequired("3 ranges lose quorum")
        return {"ok": True, "targets": ["n1", "n2"]}

    plane = FakePlane(state={"cluster": {}, "regions": [], "nodes": [], "scenario": None, "events": []}, act=act)
    monkeypatch.setattr(PA, "chaos_client", lambda: plane)
    first = client.post("/api/v1/platform/chaos", json={"action": "outage", "kind": "region", "value": "us-east-1"})
    assert first.status_code == 409 and first.get_json() == {"error": "3 ranges lose quorum", "code": "needs_confirm",
                                                            "needs_confirm": True, "warning": "3 ranges lose quorum"}
    before = events.since(None)[-1]["id"] if events.since(None) else 0
    second = client.post("/api/v1/platform/chaos", json={"action": "outage", "kind": "region", "value": "us-east-1",
                                                        "confirm": True})
    assert second.status_code == 200 and plane.acts[-1] == ("outage", "region", "us-east-1", True)
    assert any(e["source"] == "chaos" and "us-east-1" in e["text"] for e in events.since(before))
    bad = client.post("/api/v1/platform/chaos", json={"action": "explode", "kind": "region", "value": "x"})
    assert bad.status_code == 400


def test_topology_changes_become_events(monkeypatch):
    client = _signed_in(monkeypatch)
    nodes = {"live": True}

    class FakeApi:
        base_url = "http://console:8080"

        def nodes(self):
            return [{"node_id": 3, "address": None, "region": "us-west-2", "zone": "us-west-2a",
                     "live": nodes["live"], "status": "live" if nodes["live"] else "dead", "replicas": 9, "leases": 4}]

    monkeypatch.setattr(PA, "chaos_client", lambda: None)
    monkeypatch.setattr(PA, "status_client", lambda: FakeApi())
    monkeypatch.setenv("CRDB_ADMIN_URL", "http://console:8080")  # configured, so trusted without the region check
    from banko_ai.config.settings import reset_config
    reset_config()
    before = events.since(None)[-1]["id"] if events.since(None) else 0
    assert client.get("/api/v1/platform/cluster").get_json()["topology"]["source"] == "status_api"
    nodes["live"] = False
    client.application.extensions["banko_platform"].cluster.refresh_soon()
    client.get("/api/v1/platform/cluster")
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not any("Node 3" in e["text"] for e in events.since(before)):
        time.sleep(0.05)
    assert any(e["level"] == "bad" and e["text"] == "Node 3 in us-west-2 is down" for e in events.since(before))
    body = client.get("/api/v1/platform/cluster").get_json()
    # A node that is down holds no leases, whatever it last reported; its
    # region stays on the card at zero while the survivors take them.
    assert body["leases"] == {"source": "nodes", "by_region": {"us-west-2": 0}}


def test_the_data_card_reports_users_residency_vectors_cdc_and_integrity(monkeypatch):
    client = _signed_in(monkeypatch)
    body = client.get("/api/v1/platform/data").get_json()
    assert {"users", "residency", "vectors", "cdc", "integrity", "as_of", "stale"} <= set(body)
    assert sum(r["personas"] for r in body["users"]) == 3
    checked = client.post("/api/v1/platform/integrity/check", json={}).get_json()
    assert checked["ok"] is True and checked["drift"] == 0


def test_load_users_prepare_and_clear(monkeypatch, fake_sim):
    client = _signed_in(monkeypatch)
    assert client.post("/api/v1/platform/load/users", json={"action": "prepare", "count": 2}).get_json()["users"] >= 2
    fake_sim.state = "running"
    assert client.post("/api/v1/platform/load/users", json={"action": "clear"}).status_code == 409
    fake_sim.state = "idle"
    assert client.post("/api/v1/platform/load/users", json={"action": "clear"}).get_json()["removed"] >= 2
    assert client.post("/api/v1/platform/load/users", json={"action": "prepare", "count": 0}).status_code == 400


def test_live_events_start_over_for_a_cursor_from_before_a_restart(monkeypatch):
    client = _signed_in(monkeypatch)
    first = client.get("/api/v1/platform/live").get_json()
    epoch = first["events_epoch"]
    events.publish("after the cursor", source="test")
    newest = events.since(None)[-1]["id"]
    assert client.get(f"/api/v1/platform/live?since={newest}&epoch={epoch}").get_json()["events"] == []
    stale = client.get(f"/api/v1/platform/live?since={newest + 1000}&epoch=before-restart").get_json()
    assert any(e["text"] == "after the cursor" for e in stale["events"])
