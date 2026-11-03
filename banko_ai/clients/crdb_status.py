"""CockroachDB's HTTP API, for the nodes behind the Platform page's
topology card: region, zone, liveness, replicas and leases. SQL cannot
say which nodes are live on CockroachDB v26 without crdb_internal, which
is restricted and unsupported in production; the DB Console's API is the
supported way. On a secure cluster the client logs in once through
/api/v2/login/ and reuses the session.
"""

from __future__ import annotations

import logging
import threading

import requests
from sqlalchemy.engine import make_url

from ..config.settings import get_config
from ..utils.db_retry import normalize_db_url

log = logging.getLogger("banko.crdb_status")

LIVENESS = {0: "unknown", 1: "dead", 2: "unavailable", 3: "live", 4: "decommissioning", 5: "decommissioned",
            6: "draining"}


class StatusUnavailable(Exception):
    """The HTTP API did not answer, refused the login, or failed."""


class StatusClient:
    def __init__(self, base_url: str, user: str = "", password: str = "", *, timeout: float = 3.0,
                 session: requests.Session | None = None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._user, self._password = user, password
        self._session = session or requests.Session()
        self._token: str | None = None
        self._lock = threading.Lock()

    def _request(self, method: str, path: str, **kw) -> requests.Response:
        try:
            return self._session.request(method, self.base_url + path, timeout=(1.5, self.timeout), **kw)
        except requests.RequestException as exc:
            log.info("CockroachDB HTTP API unreachable: %s", exc.__class__.__name__)
            raise StatusUnavailable("The CockroachDB HTTP API did not answer.") from None

    def _login(self) -> None:
        r = self._request("POST", "/api/v2/login/", data={"username": self._user, "password": self._password})
        try:
            token = r.json().get("session") if r.status_code < 400 else None
        except ValueError:
            token = None
        if not token:
            raise StatusUnavailable("The CockroachDB HTTP API refused the login; check CRDB_ADMIN_USER.")
        self._token = token

    def _get(self, path: str) -> dict:
        with self._lock:
            for attempt in range(2):
                headers = {"X-Cockroach-API-Session": self._token} if self._token else {}
                r = self._request("GET", path, headers=headers)
                if r.status_code == 401 and self._user and attempt == 0:
                    self._token = None
                    self._login()
                    continue
                if r.status_code >= 400:
                    raise StatusUnavailable(f"The CockroachDB HTTP API answered {r.status_code}.")
                try:
                    body: dict = r.json()
                except ValueError:
                    raise StatusUnavailable("The CockroachDB HTTP API did not answer with JSON.") from None
                return body
        raise StatusUnavailable("The CockroachDB HTTP API refused the login.")

    def nodes(self) -> list[dict]:
        out = []
        for n in self._get("/api/v2/nodes/").get("nodes") or []:
            tiers = {t.get("key"): t.get("value") for t in (n.get("locality") or {}).get("tiers") or []}
            stores = list((n.get("store_metrics") or {}).values())
            code = n.get("liveness_status")
            out.append({"node_id": n.get("node_id"), "address": (n.get("address") or {}).get("address_field"),
                        "region": tiers.get("region"), "zone": tiers.get("zone"),
                        "live": None if code is None else code == 3, "status": LIVENESS.get(code, "unknown"),
                        "replicas": int(sum(s.get("replicas", 0) for s in stores)),
                        "leases": int(sum(s.get("replicas.leaseholders", 0) for s in stores))})
        return sorted(out, key=lambda node: node["node_id"] or 0)


def _derived(database_url: str) -> str | None:
    """http://<database host>:8080 on an insecure cluster, where the DB
    Console usually is; a secure cluster needs CRDB_ADMIN_URL."""
    url = make_url(normalize_db_url(database_url))
    if url.query.get("sslmode") != "disable" or not url.host:
        return None
    return f"http://{url.host}:8080"


_clients: dict[tuple[str, str], StatusClient] = {}
_clients_lock = threading.Lock()


def status_client() -> StatusClient | None:
    """The HTTP API for this database (configured or derived), or None. One
    client per address, so a secure cluster logs in once."""
    cfg = get_config()
    base = cfg.crdb_admin_url or _derived(cfg.database_url)
    if not base:
        return None
    key = (base, cfg.crdb_admin_user)
    with _clients_lock:
        client = _clients.get(key)
        if client is None:
            client = _clients[key] = StatusClient(base, cfg.crdb_admin_user, cfg.crdb_admin_password)
        return client
