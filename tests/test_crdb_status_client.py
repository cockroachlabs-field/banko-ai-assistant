"""The CockroachDB HTTP API client behind the topology card: nodes with
their region, liveness, replicas and leases; a login on secure clusters;
a derived address on insecure ones; a dead API that fails fast."""

import logging
from urllib.parse import urlparse

import pytest
import requests

from banko_ai.clients.crdb_status import StatusClient, StatusUnavailable, status_client

NODES = {"nodes": [
    {"node_id": 1, "address": {"address_field": "n1:26257"}, "liveness_status": 3,
     "locality": {"tiers": [{"key": "cloud", "value": "aws"}, {"key": "region", "value": "us-east-1"},
                            {"key": "zone", "value": "us-east-1a"}]},
     "store_metrics": {"1": {"replicas": 301, "replicas.leaseholders": 78}}},
    {"node_id": 2, "address": {"address_field": "n2:26257"}, "liveness_status": 1,
     "locality": {"tiers": [{"key": "region", "value": "us-west-2"}, {"key": "zone", "value": "us-west-2a"}]},
     "store_metrics": {"2": {"replicas": 290, "replicas.leaseholders": 0}}},
]}


class _Resp:
    def __init__(self, status, body):
        self.status_code, self._body = status, body

    def json(self):
        return self._body


class _Session:
    def __init__(self, script=None, raise_exc=None):
        self.script, self.raise_exc, self.calls = list(script or []), raise_exc, []

    def request(self, method, url, **kw):
        self.calls.append((method, urlparse(url).path, kw))
        if self.raise_exc is not None:
            raise self.raise_exc
        status, body = self.script.pop(0)
        return _Resp(status, body)


def test_nodes_are_summarized():
    client = StatusClient("http://db:8080/", session=_Session([(200, NODES)]))
    assert client.nodes() == [
        {"node_id": 1, "address": "n1:26257", "region": "us-east-1", "zone": "us-east-1a", "live": True,
         "status": "live", "replicas": 301, "leases": 78},
        {"node_id": 2, "address": "n2:26257", "region": "us-west-2", "zone": "us-west-2a", "live": False,
         "status": "dead", "replicas": 290, "leases": 0},
    ]


def test_a_secure_cluster_logs_in_once_and_never_logs_the_password(caplog):
    session = _Session([(401, {}), (200, {"session": "tok"}), (200, NODES), (200, NODES)])
    client = StatusClient("https://db:8080", "banko", "pa55word", session=session)
    with caplog.at_level(logging.DEBUG):
        assert len(client.nodes()) == 2 and len(client.nodes()) == 2
    assert [(m, p) for m, p, _ in session.calls] == [("GET", "/api/v2/nodes/"), ("POST", "/api/v2/login/"),
                                                     ("GET", "/api/v2/nodes/"), ("GET", "/api/v2/nodes/")]
    assert session.calls[1][2]["data"] == {"username": "banko", "password": "pa55word"}
    assert session.calls[3][2]["headers"]["X-Cockroach-API-Session"] == "tok"
    assert "pa55word" not in caplog.text


def test_a_dead_api_is_unavailable_fast():
    session = _Session(raise_exc=requests.ConnectTimeout("no route"))
    with pytest.raises(StatusUnavailable):
        StatusClient("http://db:8080", session=session).nodes()
    assert session.calls[0][2]["timeout"][0] <= 2


def test_status_client_derives_an_insecure_address(monkeypatch):
    from banko_ai.config.settings import reset_config

    monkeypatch.setenv("CRDB_ADMIN_URL", "")
    monkeypatch.setenv("DATABASE_URL", "postgresql://root@db.example:26257/defaultdb?sslmode=disable")
    reset_config()
    try:
        assert status_client().base_url == "http://db.example:8080"
        monkeypatch.setenv("DATABASE_URL", "postgresql://banko@db.example:26257/defaultdb?sslmode=verify-full")
        reset_config()
        assert status_client() is None
        monkeypatch.setenv("CRDB_ADMIN_URL", "https://console.example:8080")
        reset_config()
        assert status_client().base_url == "https://console.example:8080"
    finally:
        reset_config()
