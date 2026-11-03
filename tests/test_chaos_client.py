"""The chaos control plane client: a summarized state, actions that carry
the plane's warnings, a token that is sent and never logged, and a dead
plane that fails fast."""

import logging
from urllib.parse import urlparse

import pytest
import requests

from banko_ai.clients.chaos import (
    ChaosClient,
    ChaosConfirmRequired,
    ChaosError,
    ChaosUnavailable,
    chaos_client,
)

STATE = {
    "cluster": {"display_name": "Global Payments", "cluster_id": "gp", "crdb_version": "v26.3.2",
                "survival_goal": "region", "status": "degraded", "latency": "realistic", "live_nodes": 3,
                "total_nodes": 4, "model": "multi-region", "warnings": []},
    "regions": [{"name": "us-east-1", "label": "N. Virginia", "primary": True, "state": "healthy", "nodes": ["n1", "n2"]},
                {"name": "us-west-2", "label": "Oregon", "primary": False, "state": "degraded", "nodes": ["n3", "n4"]}],
    "nodes": [{"name": "n1", "region": "us-east-1", "zone": "us-east-1a", "runtime": "running", "liveness": "live",
               "node_id": 1, "leases": 10, "replicas": 30, "partitioned": False, "isolated": False, "brownout": None},
              {"name": "n2", "region": "us-east-1", "zone": "us-east-1b", "runtime": "running", "liveness": "live",
               "node_id": 2, "leases": 11, "replicas": 30, "partitioned": False, "isolated": False, "brownout": None},
              {"name": "n3", "region": "us-west-2", "zone": "us-west-2a", "runtime": "stopped", "liveness": "dead",
               "node_id": 3, "leases": 0, "replicas": 30, "partitioned": False, "isolated": False, "brownout": None},
              {"name": "n4", "region": "us-west-2", "zone": "us-west-2b", "runtime": "running", "liveness": "live",
               "node_id": 4, "leases": 9, "replicas": 30, "partitioned": False, "isolated": False, "brownout": None}],
    "scenario": None,
    "events": [{"id": i, "ts": 100.0 + i, "text": f"e{i}", "level": "info"} for i in range(30)],
}


class _Resp:
    def __init__(self, status: int, body):
        self.status_code, self._body = status, body

    def json(self):
        return self._body


class _Session:
    def __init__(self, responses=None, raise_exc=None):
        self.responses, self.raise_exc, self.calls = responses or {}, raise_exc, []

    def request(self, method, url, **kw):
        self.calls.append((method, urlparse(url).path, kw))
        if self.raise_exc is not None:
            raise self.raise_exc
        status, body = self.responses[(method, urlparse(url).path)]
        return _Resp(status, body)


def _client(token: str = "", **kw) -> tuple[ChaosClient, _Session]:
    session = _Session(**kw)
    return ChaosClient("http://plane:8088/", token, session=session), session


def test_state_is_summarized():
    client, _ = _client(responses={("GET", "/api/state"): (200, STATE)})
    s = client.state()
    assert s["cluster"]["status"] == "degraded" and s["cluster"]["live_nodes"] == 3
    west = next(r for r in s["regions"] if r["name"] == "us-west-2")
    assert (west["nodes_up"], west["nodes_total"], west["state"]) == (1, 2, "degraded")
    assert [n["name"] for n in s["nodes"]] == ["n1", "n2", "n3", "n4"] and s["nodes"][2]["liveness"] == "dead"
    assert len(s["events"]) == 20 and s["events"][-1]["text"] == "e29"


def test_a_disruptive_action_that_needs_confirming_carries_the_warning():
    client, session = _client(responses={("POST", "/api/chaos"): (409, {"detail": {
        "needs_confirm": True, "warning": "3 ranges lose quorum"}})})
    with pytest.raises(ChaosConfirmRequired) as caught:
        client.act("outage", "region", "us-east-1")
    assert "quorum" in caught.value.warning
    assert session.calls[0][2]["json"] == {"action": "outage", "kind": "region", "value": "us-east-1", "confirm": False}


def test_a_confirmed_action_says_so():
    client, session = _client(responses={("POST", "/api/chaos"): (200, {"ok": True, "targets": ["n1", "n2"]})})
    assert client.act("outage", "region", "us-east-1", confirm=True)["targets"] == ["n1", "n2"]
    assert session.calls[0][2]["json"]["confirm"] is True


def test_the_token_is_sent_and_never_logged(caplog):
    client, session = _client("s3cret", responses={("POST", "/api/chaos"): (401, {"detail": "control token required"})})
    with caplog.at_level(logging.DEBUG), pytest.raises(ChaosError) as caught:
        client.act("recover", "all")
    assert session.calls[0][2]["headers"]["x-control-token"] == "s3cret"
    assert "refused" in str(caught.value) and "s3cret" not in str(caught.value) and "s3cret" not in caplog.text


def test_unknown_actions_and_settings_never_reach_the_plane():
    client, session = _client()
    for call in (lambda: client.act("explode", "region", "x"), lambda: client.act("crash", "planet", "x"),
                 lambda: client.latency("ludicrous")):
        with pytest.raises(ValueError):
            call()
    assert session.calls == []


def test_latency_posts_the_setting():
    client, session = _client(responses={("POST", "/api/cluster/latency"): (200, {"latency": "off"})})
    client.latency("off")
    assert session.calls[0][2]["json"] == {"setting": "off"}


def test_a_dead_plane_is_unavailable_fast():
    client, session = _client(raise_exc=requests.ConnectTimeout("no route"))
    with pytest.raises(ChaosUnavailable):
        client.state()
    assert session.calls[0][2]["timeout"][0] <= 2
    client, _ = _client(responses={("GET", "/api/state"): (502, {"detail": "bad gateway"})})
    with pytest.raises(ChaosUnavailable):
        client.state()


def test_a_rejected_action_says_why():
    client, _ = _client(responses={("POST", "/api/chaos"): (400, {"detail": "unknown target region 'mars'"})})
    with pytest.raises(ChaosError, match="mars"):
        client.act("outage", "region", "mars")


def test_chaos_client_follows_the_settings(monkeypatch):
    from banko_ai.config.settings import reset_config

    monkeypatch.setenv("CHAOS_CONTROL_URL", "")
    reset_config()
    assert chaos_client() is None
    monkeypatch.setenv("CHAOS_CONTROL_URL", "http://localhost:8088")
    reset_config()
    try:
        assert chaos_client().base_url == "http://localhost:8088"
    finally:
        reset_config()


def test_actions_wait_for_the_plane_longer_than_state_reads():
    """Recovering a region or switching latency runs Docker work on the plane
    before it answers; only reading its state is held to the short timeout."""
    client, session = _client(responses={("GET", "/api/state"): (200, STATE), ("POST", "/api/chaos"): (200, {}),
                                         ("POST", "/api/cluster/latency"): (200, {})})
    client.state()
    client.act("recover", "region", "us-west-2")
    client.latency("realistic")
    timeouts = [kw["timeout"][1] for _, _, kw in session.calls]
    assert timeouts[0] == client.timeout and timeouts[1] == timeouts[2] == client.action_timeout > client.timeout


def test_one_client_per_plane_so_connections_are_reused(monkeypatch):
    monkeypatch.setenv("CHAOS_CONTROL_URL", "http://plane:8088")
    from banko_ai.config.settings import reset_config
    reset_config()
    try:
        assert chaos_client() is chaos_client()
    finally:
        monkeypatch.delenv("CHAOS_CONTROL_URL")
        reset_config()
