"""banko-ai doctor: is this machine ready to demo? One line per check, each
answering within a few seconds, so it doubles as the wifi-off check before
a talk. The provider check is the one that can take longer: it asks the
model a question, and a local model that is not loaded yet loads first.
Errors name their class or a fixed reason only: a message could echo a URL
with a password in it."""

from __future__ import annotations

import shutil
import socket
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass

import requests

from .config.settings import Settings, get_config

OK, FAIL, SKIP = "ok", "fail", "skip"
TIMEOUT = 3.0
# How long a hosted model gets to answer one question; it takes a second or two.
ANSWER_SECONDS = 20.0


@dataclass(frozen=True)
class Check:
    name: str
    state: str
    detail: str


def _reachable(url: str) -> bool:
    """A TCP connect bounded by TIMEOUT, before the pool's own connect
    timeout (ten seconds by default) gets a chance to hold the line."""
    from sqlalchemy.engine import make_url

    from .utils.db_retry import normalize_db_url
    try:
        u = make_url(normalize_db_url(url))
    except Exception:
        return True  # let the real connect explain it
    if not u.host:
        return True
    try:
        socket.create_connection((u.host, u.port or 26257), TIMEOUT).close()
        return True
    except OSError:
        return False


def _database(cfg: Settings) -> Check:
    from sqlalchemy import text

    from .db.engine import NO_SUCH_DATABASE, get_engine, missing_database
    if not _reachable(cfg.database_url):
        return Check("database", FAIL, "unreachable (no answer on its host and port); check DATABASE_URL")
    try:
        with get_engine(cfg.database_url).connect() as conn:
            version = str(conn.execute(text("SELECT version()")).scalar() or "")
            # CockroachDB accepts a connection to a database that does not
            # exist, and only a statement that touches it says so.
            there = conn.execute(text(
                "SELECT count(*) FROM [SHOW DATABASES] WHERE database_name = current_database()")).scalar()
            name = str(conn.execute(text("SELECT current_database()")).scalar() or "")
    except Exception as exc:
        missing = missing_database(exc, cfg.database_url)
        if missing is not None:
            return Check("database", FAIL, NO_SUCH_DATABASE.format(name=missing))
        return Check("database", FAIL, f"unreachable ({exc.__class__.__name__}); check DATABASE_URL")
    if not there:
        return Check("database", FAIL, NO_SUCH_DATABASE.format(name=name))
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


def _chat():
    """The model as every answer reaches it."""
    from .ai_providers.streaming import ProviderChat
    return ProviderChat()


_OUT_OF_QUOTA = ("quota", "resource_exhausted", "resourceexhausted")
_CREDENTIALS = ("invalid_api_key", "incorrect api key", "api key not valid", "unauthorized", "authenticationerror",
                "invalid_grant", "expiredtoken", "accessdenied", "permission_denied", "nocredentials")


def _why(exc: BaseException) -> str:
    """Why the model gave no answer, in words fixed here. The error's own
    text is searched and never shown: it can carry a key or a URL."""
    said = ""
    seen: BaseException | None = exc
    while seen is not None and len(said) < 20_000:
        said += f" {seen.__class__.__name__} {seen}".lower()
        seen = seen.__cause__ or seen.__context__
    if any(word in said for word in _OUT_OF_QUOTA):
        return "is out of quota; its account needs more before it answers again"
    if any(word in said for word in _CREDENTIALS):
        return "rejected the credentials"
    return f"did not answer ({exc.__class__.__name__}); check its credentials and network"


def _answer_seconds(cfg: Settings) -> float:
    """A local model may have to load into memory first, so it gets as long
    as the app itself waits for it."""
    return float(cfg.ollama_timeout_seconds) if cfg.ai_service == "ollama" else ANSWER_SECONDS


def _one_answer(seconds: float) -> str | None:
    """Ask the model one question. None when it replied, else why not."""
    outcome: list[str | None] = []

    def ask() -> None:
        try:
            reply = _chat().complete([{"role": "user", "content": "Reply with the single word OK."}], temperature=0.0)
            outcome.append(None if str(reply or "").strip() else "did not answer (an empty reply)")
        except Exception as exc:
            outcome.append(_why(exc))

    worker = threading.Thread(target=ask, name="banko-doctor-ask", daemon=True)
    worker.start()
    worker.join(seconds)
    return outcome[0] if outcome else f"did not answer within {seconds:.0f} s"


def _provider(cfg: Settings) -> Check:
    """A login token or a model list is not an answer: an account that is out
    of quota still hands those out, and a local model can be listed and fail
    to load. So this asks one real question, a few tokens each run."""
    from .ai_providers.factory import AIProviderFactory
    shown = f", shown as {cfg.ai_display_provider}" if cfg.ai_display_provider else ""
    try:
        AIProviderFactory.create_provider(cfg.ai_service, cfg.get_ai_config()[cfg.ai_service])
    except Exception as exc:
        return Check("AI provider", FAIL, f"{cfg.ai_service} is not set up ({exc.__class__.__name__}){shown}")
    why_not = _one_answer(_answer_seconds(cfg))
    if why_not is None:
        return Check("AI provider", OK, f"{cfg.ai_service} answers{shown}")
    return Check("AI provider", FAIL, f"{cfg.ai_service} {why_not}{shown}")


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
    from .db.engine import mask_url
    client = chaos_client()
    if client is None:
        return Check("chaos control", SKIP, "not configured (CHAOS_CONTROL_URL)")
    shown = mask_url(cfg.chaos_control_url)
    try:
        client.state()
    except ChaosError:
        return Check("chaos control", FAIL, f"{shown} did not answer")
    return Check("chaos control", OK, f"{shown} answers")


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
