"""The Platform page's API: what the load simulator, the app and the cluster
are doing, and the controls to start load and to break things.

/live answers from memory only, never the database, so it keeps answering
while a region is down. /cluster and /data answer from snapshots that a
background refresh keeps fresh while someone is looking: a slow source
makes a card stale, never a request slow. Every endpoint needs a signed-in
session; the writes also need a JSON body and PLATFORM_CONTROLS on.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

from flask import Blueprint, current_app, jsonify, request, session

from ..clients.chaos import ACTIONS, KINDS, ChaosConfirmRequired, ChaosError, ChaosUnavailable, chaos_client
from ..clients.crdb_status import StatusUnavailable, status_client
from ..config.settings import get_config
from ..observability.events import events
from ..observability.metrics import metrics
from ..services import integrity, load_users, platform
from ..workers.traffic import LoadConfig, LoadConfigError, SimulatorBusy, get_simulator

log = logging.getLogger("banko.platform")
platform_api = Blueprint("platform_api", __name__, url_prefix="/api/v1/platform")
_RECOVERING = ("recover", "start", "heal")


class _Snapshot:
    """A value refreshed in the background while it is being asked for.
    Readers never wait, except up to `wait` seconds for the very first one."""

    def __init__(self, fetch: Callable[[], dict], every: float,
                 on_change: Callable[[dict | None, dict], None] | None = None):
        self._fetch, self.every, self._on_change = fetch, every, on_change
        self._value: dict | None = None
        self._as_of = 0.0
        self._failed = False
        self._refreshing = False
        self._first = threading.Event()
        self._lock = threading.Lock()

    def _refresh(self) -> None:
        try:
            value, failed = self._fetch(), False
        except Exception:
            log.info("platform refresh failed", exc_info=True)
            value, failed = None, True
        with self._lock:
            previous = self._value
            if not failed:
                self._value, self._as_of = value, time.time()
            self._failed, self._refreshing = failed, False
        self._first.set()
        if not failed and self._on_change is not None:
            try:
                self._on_change(previous, value)
            except Exception:
                log.warning("platform change events failed", exc_info=True)

    def get(self, wait: float = 2.0) -> dict:
        with self._lock:
            due = not self._refreshing and time.time() - self._as_of >= self.every
            if due:
                self._refreshing = True
        if due:
            threading.Thread(target=self._refresh, name="banko-platform-refresh", daemon=True).start()
        if self._value is None:
            self._first.wait(wait)
        with self._lock:
            return {"value": self._value, "as_of": self._as_of or None,
                    "stale": self._failed or self._value is None or time.time() - self._as_of > 3 * self.every}

    def peek(self) -> dict | None:
        with self._lock:
            return self._value

    def refresh_soon(self) -> None:
        with self._lock:
            self._as_of = 0.0


class _State:
    def __init__(self) -> None:
        self.verified: dict[str, bool] = {}
        self.cluster = _Snapshot(lambda: _fetch_cluster(self), every=2.0, on_change=_topology_events)
        self.data = _Snapshot(lambda: _fetch_data(self), every=30.0)


_state_lock = threading.Lock()


def _state() -> _State:
    with _state_lock:
        return current_app.extensions.setdefault("banko_platform", _State())


# -- sources -----------------------------------------------------------------------------------


def _status_api(state: _State):
    """The HTTP API, when configured, or when the derived address turns out
    to be this database's cluster (same regions); a second local cluster on
    port 8080 is never mistaken for this one."""
    client = status_client()
    if client is None or get_config().crdb_admin_url:
        return client
    ok = state.verified.get(client.base_url)
    if ok is None:
        try:
            ok = set(platform.cluster_regions()) == {n["region"] for n in client.nodes() if n["region"]}
        except Exception:
            return None
        state.verified[client.base_url] = ok
        if not ok:
            log.info("not using %s: its nodes are another cluster's", client.base_url)
    return client if ok else None


def _fetch_cluster(state: _State) -> dict:
    chaos, topology = None, None
    plane = chaos_client()
    if plane is not None:
        try:
            chaos = plane.state()
            topology = {"source": "chaos", "regions": [r["name"] for r in chaos["regions"]], "nodes": [
                {"node_id": n["node_id"], "name": n["name"], "address": None, "region": n["region"], "zone": n["zone"],
                 "live": n["liveness"] == "live" and n["runtime"] == "running",
                 "status": n["liveness"] if n["runtime"] == "running" else n["runtime"],
                 "replicas": n["replicas"], "leases": n["leases"]} for n in chaos["nodes"]]}
        except ChaosError:
            chaos = None
    if topology is None:
        api = _status_api(state)
        if api is not None:
            try:
                nodes = api.nodes()
                topology = {"source": "status_api", "regions": sorted({n["region"] for n in nodes if n["region"]}),
                            "nodes": nodes}
            except StatusUnavailable:
                pass
    if topology is None:
        topology = platform.topology()
    return {"topology": topology, "chaos": chaos}


def _topology_events(previous: dict | None, current: dict) -> None:
    if not previous:
        return
    before = {n["node_id"]: n["live"] for n in previous["topology"]["nodes"] if n.get("live") is not None}
    for n in current["topology"]["nodes"]:
        was = before.get(n["node_id"])
        if was is None or n.get("live") is None or was == n["live"]:
            continue
        where = f" in {n['region']}" if n.get("region") else ""
        events.publish(f"Node {n['node_id']}{where} is {'back' if n['live'] else 'down'}",
                       level="good" if n["live"] else "bad", source="cluster")
    regions = {r["name"]: r["state"] for r in (previous.get("chaos") or {}).get("regions") or []}
    for r in (current.get("chaos") or {}).get("regions") or []:
        was = regions.get(r["name"])
        if was and was != r["state"]:
            events.publish(f"Region {r['name']} is {r['state']}", level="good" if r["state"] == "healthy" else "bad",
                           source="cluster")


def _node_leases(topology: dict | None) -> dict | None:
    nodes = (topology or {}).get("nodes") or []
    if not any(n.get("leases") is not None for n in nodes):
        return None
    by_region: dict[str, int] = {}
    for n in nodes:
        if n.get("leases") is not None:
            key = n.get("region") or "all"
            by_region[key] = by_region.get(key, 0) + int(n["leases"])
    return {"source": "nodes", "by_region": by_region}


def _fetch_data(state: _State) -> dict:
    out: dict = {}
    for key, fetch in (("users", platform.users_by_region), ("residency", platform.residency),
                       ("vectors", platform.vector_stats), ("cdc", platform.cdc_status)):
        try:
            out[key] = fetch()
        except Exception:
            log.info("platform %s card unavailable", key, exc_info=True)
            out[key] = None
    out["leases_sql"] = None
    if _node_leases((state.cluster.peek() or {}).get("topology")) is None:
        try:
            out["leases_sql"] = {"source": "sql", "by_region": platform.lease_distribution()["accounts"]}
        except Exception:
            log.info("lease distribution unavailable", exc_info=True)
    return out


# -- endpoints ---------------------------------------------------------------------------------


def _error(status: int, code: str, message: str):
    # As every v2 API: the plain sentence in "error" (the page shows it as
    # is), and a code beside it for pages that branch on the kind.
    return jsonify({"error": message, "code": code}), status


@platform_api.before_request
def _guard():
    if not session.get("user_id"):
        return _error(401, "sign_in", "Sign in to see the platform.")
    if request.method == "POST":
        if not get_config().platform_controls:
            return _error(403, "read_only", "The Platform page is read-only here.")
        if not request.is_json or not isinstance(request.get_json(silent=True), dict):
            return _error(400, "bad_request", "Send a JSON object.")
    return None


@platform_api.after_request
def _identify(response):
    response.headers["X-Banko-User"] = session.get("user_id") or ""
    return response


@platform_api.get("/live")
def live():
    sim = get_simulator()

    def window(name: str) -> dict:
        return metrics.summary(name, window_s=10)

    retries, failures = window("db.retries"), window("db.failures")
    return jsonify({
        "load": sim.status(),
        "summary": window("load"),
        "ops": {name.rsplit(".", 1)[1]: window(name) for name in metrics.names("load.op.")},
        "regions": {name.split(".", 2)[2]: window(name) for name in metrics.names("load.region.")},
        "series": metrics.timeline("load", seconds=300),
        "http": window("http"),
        "db": {"retries_s": retries["ops_s"], "failures_s": failures["ops_s"]},
        "events": events.since(request.args.get("since", type=int)),
        "controls": _controls(),
        "as_of": time.time(),
    })


def _controls() -> dict:
    cfg = get_config()
    return {"load": cfg.platform_controls, "chaos": chaos_client() is not None,
            "defaults": {"users": cfg.load_default_users, "rate": cfg.load_default_rate,
                         "workers": cfg.load_default_workers},
            "limits": {"users": cfg.load_max_users, "rate": cfg.load_max_rate, "workers": cfg.load_max_workers}}


@platform_api.get("/cluster")
def cluster():
    state = _state()
    snap = state.cluster.get()
    value = snap["value"] or {}
    topology = value.get("topology")
    leases = _node_leases(topology) or (state.data.peek() or {}).get("leases_sql")
    return jsonify({"topology": topology, "chaos": value.get("chaos"), "leases": leases,
                    "as_of": snap["as_of"], "stale": snap["stale"]})


@platform_api.get("/data")
def data():
    snap = _state().data.get(wait=6.0)
    value = {k: v for k, v in (snap["value"] or {}).items() if k != "leases_sql"}
    for key in ("users", "residency", "vectors", "cdc"):
        value.setdefault(key, None)
    report = integrity.latest_report()
    value["integrity"] = None if report is None else {
        "checked_at": report.checked_at.isoformat(), "accounts": report.accounts, "drift": len(report.drifted),
        "skipped": report.skipped, "ok": report.ok, "follower_reads": report.follower_reads}
    return jsonify({**value, "as_of": snap["as_of"], "stale": snap["stale"]})


@platform_api.post("/load")
def load():
    body = request.get_json()
    action, sim = body.get("action"), get_simulator()
    try:
        if action == "start":
            return jsonify(sim.start(LoadConfig.from_dict(body))), 202
        if action == "apply":
            return jsonify(sim.apply(LoadConfig.from_dict(body, base=sim.config)))
        if action == "stop":
            return jsonify(sim.stop())
    except LoadConfigError as exc:
        return _error(400, "bad_request", str(exc))
    except SimulatorBusy as exc:
        return _error(409, "busy", str(exc))
    return _error(400, "bad_request", "Pick start, apply or stop.")


@platform_api.post("/load/users")
def load_user_pool():
    body = request.get_json()
    action = body.get("action")
    if action == "prepare":
        try:
            count = LoadConfig.from_dict({"users": body.get("count")}).users
        except LoadConfigError as exc:
            return _error(400, "bad_request", str(exc))
        return jsonify({"users": len(load_users.prepare(count))})
    if action == "clear":
        if get_simulator().status()["state"] != "idle":
            return _error(409, "busy", "Stop the load before clearing its customers.")
        removed = load_users.clear()
        events.publish(f"Removed {removed} load customers and everything they owned", source="load")
        return jsonify({"removed": removed})
    return _error(400, "bad_request", "Pick prepare or clear.")


@platform_api.post("/chaos")
def chaos():
    plane = chaos_client()
    if plane is None:
        return _error(404, "not_configured", "Set CHAOS_CONTROL_URL to run chaos from here.")
    body = request.get_json()
    action, kind, value = body.get("action"), body.get("kind"), str(body.get("value") or "")
    if action not in ACTIONS or kind not in KINDS:
        return _error(400, "bad_request", "That is not an action the chaos control plane knows.")
    try:
        result = plane.act(action, kind, value, confirm=bool(body.get("confirm")))
    except ChaosConfirmRequired as exc:
        return jsonify({"error": exc.warning, "code": "needs_confirm", "needs_confirm": True,
                        "warning": exc.warning}), 409
    except ChaosUnavailable as exc:
        return _error(503, "unavailable", str(exc))
    except ChaosError as exc:
        return _error(502, "refused", str(exc))
    target = "the whole cluster" if kind == "all" else value
    events.publish(f"Chaos: {action} {target}", level="good" if action in _RECOVERING else "bad", source="chaos")
    _state().cluster.refresh_soon()
    return jsonify(result)


@platform_api.post("/chaos/latency")
def chaos_latency():
    plane = chaos_client()
    if plane is None:
        return _error(404, "not_configured", "Set CHAOS_CONTROL_URL to run chaos from here.")
    setting = request.get_json().get("setting")
    try:
        result = plane.latency(setting)
    except ValueError:
        return _error(400, "bad_request", "Pick off or realistic.")
    except ChaosUnavailable as exc:
        return _error(503, "unavailable", str(exc))
    except ChaosError as exc:
        return _error(502, "refused", str(exc))
    events.publish(f"Chaos: latency {setting}", source="chaos")
    return jsonify(result)


@platform_api.post("/integrity/check")
def integrity_check():
    report = integrity.run_check()
    return jsonify({"checked_at": report.checked_at.isoformat(), "accounts": report.accounts,
                    "drift": len(report.drifted), "skipped": report.skipped, "ok": report.ok,
                    "follower_reads": report.follower_reads})
