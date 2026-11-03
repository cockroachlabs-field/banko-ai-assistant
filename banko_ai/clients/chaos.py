"""The cockroach-chaos-demo control plane, seen from Banko.

The Platform page breaks the cluster through Banko, never straight from
the browser: Banko holds the plane's control token and the browser needs
no cross-origin access. Calls time out fast (a down plane hides the chaos
card, it never holds up the page), and only the plane's known actions,
target kinds and latency settings are forwarded.
"""

from __future__ import annotations

import logging
import threading

import requests

from ..config.settings import get_config

log = logging.getLogger("banko.chaos")

ACTIONS = ("crash", "outage", "stop", "start", "partition", "heal", "brownout", "isolate", "recover")
KINDS = ("node", "zone", "region", "all")
LATENCY = ("off", "realistic")
_CLUSTER_KEYS = ("display_name", "cluster_id", "crdb_version", "survival_goal", "status", "latency",
                 "live_nodes", "total_nodes", "model")
_NODE_KEYS = ("name", "node_id", "region", "zone", "runtime", "liveness", "partitioned", "isolated", "brownout",
              "leases", "replicas")


class ChaosError(Exception):
    """The plane said no; the message is safe to show."""


class ChaosConfirmRequired(ChaosError):
    """The action would make data unavailable; the plane's warning says what."""

    def __init__(self, warning: str):
        super().__init__(warning)
        self.warning = warning


class ChaosUnavailable(ChaosError):
    """The plane did not answer, or failed."""


def summarize(state: dict) -> dict:
    """The parts of the plane's state the Platform page draws."""
    nodes = [{k: n.get(k) for k in _NODE_KEYS} for n in state.get("nodes") or []]
    up = {n["name"] for n in nodes if n["liveness"] == "live" and n["runtime"] == "running"}
    regions = [{"name": r.get("name"), "label": r.get("label"), "primary": bool(r.get("primary")),
                "state": r.get("state"), "nodes_total": len(r.get("nodes") or []),
                "nodes_up": sum(1 for name in r.get("nodes") or [] if name in up)}
               for r in state.get("regions") or []]
    return {"cluster": {k: (state.get("cluster") or {}).get(k) for k in _CLUSTER_KEYS}, "regions": regions,
            "nodes": nodes, "scenario": state.get("scenario"), "events": list(state.get("events") or [])[-20:]}


class ChaosClient:
    """timeout bounds reading the plane's state; action_timeout bounds an
    action, which the plane answers only after its Docker work is done."""

    def __init__(self, base_url: str, token: str = "", *, timeout: float = 3.0, action_timeout: float = 30.0,
                 session: requests.Session | None = None):
        self.base_url = base_url.rstrip("/")
        self.timeout, self.action_timeout = timeout, action_timeout
        self._token = token
        self._session = session or requests.Session()

    def _call(self, method: str, path: str, body: dict | None = None, *, timeout: float | None = None) -> dict:
        headers = {"x-control-token": self._token} if self._token else {}
        try:
            r = self._session.request(method, self.base_url + path, json=body, headers=headers,
                                      timeout=(1.5, timeout or self.timeout))
        except requests.RequestException as exc:
            log.info("chaos control plane unreachable: %s", exc.__class__.__name__)
            raise ChaosUnavailable("The chaos control plane did not answer.") from None
        try:
            payload = r.json()
        except ValueError:
            payload = {}
        if r.status_code < 400:
            return payload if isinstance(payload, dict) else {"result": payload}
        detail = payload.get("detail") if isinstance(payload, dict) else None
        if r.status_code == 409 and isinstance(detail, dict) and detail.get("needs_confirm"):
            raise ChaosConfirmRequired(str(detail.get("warning") or "This would make some data unavailable."))
        if r.status_code in (401, 403):
            raise ChaosError("The chaos control plane refused Banko's token; check CHAOS_CONTROL_TOKEN.")
        if r.status_code >= 500:
            raise ChaosUnavailable("The chaos control plane failed.")
        raise ChaosError(str(detail) if isinstance(detail, str) and detail else "The chaos control plane said no.")

    def state(self) -> dict:
        return summarize(self._call("GET", "/api/state"))

    def act(self, action: str, kind: str, value: str = "", *, confirm: bool = False) -> dict:
        if action not in ACTIONS:
            raise ValueError(f"unknown chaos action {action!r}")
        if kind not in KINDS:
            raise ValueError(f"unknown chaos target {kind!r}")
        return self._call("POST", "/api/chaos", {"action": action, "kind": kind, "value": value, "confirm": confirm},
                          timeout=self.action_timeout)

    def latency(self, setting: str) -> dict:
        if setting not in LATENCY:
            raise ValueError(f"unknown latency setting {setting!r}")
        return self._call("POST", "/api/cluster/latency", {"setting": setting}, timeout=self.action_timeout)


_clients: dict[tuple[str, str], ChaosClient] = {}
_clients_lock = threading.Lock()


def chaos_client() -> ChaosClient | None:
    """The configured plane, or None when CHAOS_CONTROL_URL is empty. One
    client per address and token, so its connections are reused."""
    cfg = get_config()
    if not cfg.chaos_control_url:
        return None
    key = (cfg.chaos_control_url, cfg.chaos_control_token)
    with _clients_lock:
        client = _clients.get(key)
        if client is None:
            client = _clients[key] = ChaosClient(cfg.chaos_control_url, cfg.chaos_control_token)
        return client
