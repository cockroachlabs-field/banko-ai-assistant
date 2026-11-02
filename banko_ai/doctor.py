"""banko-ai doctor: is this machine ready to demo? One line per check, each
answering within a few seconds, so it doubles as the wifi-off check before
a talk. Errors name their class only: a message could echo a URL with a
password in it."""

from __future__ import annotations

import shutil
import socket
import subprocess
from collections.abc import Callable
from dataclasses import dataclass

import requests

from .config.settings import Settings, get_config

OK, FAIL, SKIP = "ok", "fail", "skip"
TIMEOUT = 3.0


@dataclass(frozen=True)
class Check:
    name: str
    state: str
    detail: str


def _database(cfg: Settings) -> Check:
    from sqlalchemy import text

    from .db.engine import get_engine
    try:
        with get_engine(cfg.database_url).connect() as conn:
            version = str(conn.execute(text("SELECT version()")).scalar() or "")
    except Exception as exc:
        return Check("database", FAIL, f"unreachable ({exc.__class__.__name__}); check DATABASE_URL")
    words = version.split()
    return Check("database", OK, " ".join(words[:3]) if words[:1] == ["CockroachDB"] else version[:60])


def _schema(cfg: Settings) -> Check:
    from .db.schema import schema_state
    state = schema_state(cfg.database_url)
    if state == "current":
        return Check("schema", OK, "current")
    if state == "missing":
        return Check("schema", FAIL, "not created yet; banko-ai migrate or the first boot creates it")
    return Check("schema", FAIL, "out of date; banko-ai migrate or the next boot upgrades it")


def _embedding_model(cfg: Settings) -> Check:
    from huggingface_hub import try_to_load_from_cache
    name = cfg.embedding_model
    repo = name if "/" in name else f"sentence-transformers/{name}"
    if isinstance(try_to_load_from_cache(repo, "config.json"), str):
        return Check("embedding model", OK, f"{name} cached")
    return Check("embedding model", FAIL, f"{name} not cached; the first boot downloads it, which needs the network")


def _provider(cfg: Settings) -> Check:
    from .ai_providers.factory import AIProviderFactory
    from .web.app import bounded_connection_test
    shown = f", shown as {cfg.ai_display_provider}" if cfg.ai_display_provider else ""
    try:
        provider = AIProviderFactory.create_provider(cfg.ai_service, cfg.get_ai_config()[cfg.ai_service])
    except Exception as exc:
        return Check("AI provider", FAIL, f"{cfg.ai_service} is not set up ({exc.__class__.__name__}){shown}")
    if bounded_connection_test(provider, seconds=TIMEOUT * 2):
        return Check("AI provider", OK, f"{cfg.ai_service} answers{shown}")
    return Check("AI provider", FAIL, f"{cfg.ai_service} did not answer; check its credentials and network{shown}")


def _ollama_model(cfg: Settings) -> Check:
    if cfg.ai_service != "ollama":
        return Check("Ollama model", SKIP, "AI_SERVICE is not ollama")
    try:
        models = requests.get(cfg.ollama_host.rstrip("/") + "/api/tags", timeout=TIMEOUT).json().get("models") or []
    except Exception:
        return Check("Ollama model", FAIL, f"no Ollama at {cfg.ollama_host}; start ollama serve")
    names = {m.get("name") for m in models}
    if {cfg.ollama_model, f"{cfg.ollama_model}:latest"} & names:
        return Check("Ollama model", OK, f"{cfg.ollama_model} present")
    return Check("Ollama model", FAIL, f"{cfg.ollama_model} missing; scripts/airgap/preload-models.sh pulls it")


def _tesseract(cfg: Settings) -> Check:
    path = shutil.which("tesseract")
    if not path:
        return Check("tesseract", FAIL, "not on PATH, so receipts read as Unknown; scripts/setup_system_deps.sh installs it")
    try:
        ran = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=TIMEOUT)
        version = ((ran.stdout or ran.stderr).splitlines() or ["tesseract"])[0]
    except (OSError, subprocess.SubprocessError):
        version = "tesseract"
    return Check("tesseract", OK, version)


def _port(cfg: Settings) -> Check:
    host = "127.0.0.1" if cfg.host in ("", "0.0.0.0") else cfg.host
    with socket.socket() as probe:
        try:
            probe.bind((host, cfg.port))
            return Check("port", OK, f"{cfg.port} is free")
        except OSError:
            pass
    try:
        ours = requests.get(f"http://{host}:{cfg.port}/api/health/live", timeout=TIMEOUT).ok
    except requests.RequestException:
        ours = False
    if ours:
        return Check("port", OK, f"Banko is already serving {cfg.port}")
    return Check("port", FAIL, f"{cfg.port} is taken by something else; set PORT to another")


def _status_api(cfg: Settings) -> Check:
    from .clients.crdb_status import StatusUnavailable, status_client
    client = status_client()
    if client is None:
        return Check("status API", SKIP, "not configured (CRDB_ADMIN_URL); the Platform page shows less")
    try:
        nodes = client.nodes()
    except StatusUnavailable as exc:
        if not cfg.crdb_admin_url:
            return Check("status API", SKIP, f"no DB Console at {client.base_url}; set CRDB_ADMIN_URL for node liveness")
        return Check("status API", FAIL, str(exc))
    if not cfg.crdb_admin_url:
        # Derived from DATABASE_URL: localhost:8080 can be another local cluster's DB Console.
        from .services.platform import cluster_regions
        if set(cluster_regions(url=cfg.database_url)) != {n["region"] for n in nodes if n["region"]}:
            return Check("status API", SKIP, f"{client.base_url} is another cluster's DB Console; set CRDB_ADMIN_URL")
    return Check("status API", OK, f"{sum(1 for n in nodes if n['live'])} of {len(nodes)} nodes live")


def _chaos(cfg: Settings) -> Check:
    from .clients.chaos import ChaosError, chaos_client
    client = chaos_client()
    if client is None:
        return Check("chaos control", SKIP, "not configured (CHAOS_CONTROL_URL)")
    try:
        client.state()
    except ChaosError:
        return Check("chaos control", FAIL, f"{cfg.chaos_control_url} did not answer")
    return Check("chaos control", OK, f"{cfg.chaos_control_url} answers")


CHECKS: tuple[Callable[[Settings], Check], ...] = (
    _database, _schema, _embedding_model, _provider, _ollama_model, _tesseract, _port, _status_api, _chaos)


def run_checks(cfg: Settings | None = None) -> list[Check]:
    """Every check in order; the schema check waits on the database's."""
    cfg = cfg or get_config()
    out: list[Check] = []
    for check in CHECKS:
        if check is _schema and out and out[0].state == FAIL:
            out.append(Check("schema", SKIP, "needs the database"))
            continue
        try:
            out.append(check(cfg))
        except Exception as exc:  # a broken check is a red line, never a traceback
            out.append(Check(check.__name__.strip("_").replace("_", " "), FAIL, exc.__class__.__name__))
    return out
