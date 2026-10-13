"""
Main Flask application for Banko AI Assistant.

This module creates and configures the Flask application with all routes and functionality.
"""

import hashlib
import hmac
import json
import logging as _coach_log
import os
import re
import tempfile
import threading
import time
import uuid
from datetime import datetime

import requests
from flask import (
    Flask,
    flash,
    g,
    jsonify,
    redirect,
    render_template,
    render_template_string,
    request,
    session,
    url_for,
)
from flask_socketio import SocketIO
from sqlalchemy import text

from ..ai_providers.display import (
    NEUTRAL_FAILED,
    actual_identity,
    display_identity,
    display_label,
    display_models,
    neutral_error,
    ui_context,
)
from ..ai_providers.factory import AIProviderFactory
from ..coach.handler import SignalHandler
from ..coach.signals import SignalParseError, parse_changefeed_envelope
from ..config.settings import get_config
from ..db.engine import get_engine, mask_url
from ..db.regions import detect_database_regions
from ..db.schema import ensure_schema, run_pending_conversions
from ..utils.cache_manager import BankoCacheManager
from ..utils.db_retry import TRANSIENT_ERRORS, get_database_url
from ..utils.embeddings import ModelWarmingUp
from ..utils.request_trace import (
    current_trace,
    get_stored_plan,
    get_stored_trace,
    reset_trace,
    start_trace,
    store_trace,
    trace_elapsed_ms,
    trace_step,
)
from ..vector_search.generator import EnhancedExpenseGenerator
from ..vector_search.search import VectorSearchEngine
from ..workers.warmup import Readiness, start_warmup
from .auth import UserManager

PROVIDER_CHECK_SECONDS = 8.0
# Fixed forever: changing it would orphan every cookie-less caller's thread.
API_THREAD_NAMESPACE = uuid.UUID("ced3c92c-95fb-4c08-95d5-aa7a5b1f0334")


def api_thread_id(user_id: str) -> str:
    """The thread API calls without a signed-in session (curl demos) run on:
    one per user, the same on every request and restart, and never listed,
    since it has no chat_threads row."""
    return str(uuid.uuid5(API_THREAD_NAMESPACE, f"banko-api:{user_id}"))


def bounded_connection_test(provider, seconds: float = PROVIDER_CHECK_SECONDS) -> bool:
    """provider.test_connection(), given up on after `seconds`. Each provider
    already uses short timeouts; this is the backstop for SDK paths we don't
    control (an SSO refresh, a Vertex token fetch) on a black-holed network,
    so neither the ready line nor /api/health?refresh=1 waits on them."""
    result: list[bool] = []

    def check() -> None:
        try:
            result.append(bool(provider.test_connection()))
        except Exception:
            result.append(False)

    worker = threading.Thread(target=check, name="banko-provider-check", daemon=True)
    worker.start()
    worker.join(seconds)
    return bool(result and result[0])


class _TimedChat:
    """A chat page turn's model, counted and timed, so its trace names the
    model only when the turn really called it."""

    def __init__(self, chat):
        self.chat = chat
        self.calls = 0
        self.failed = 0
        self.ms = 0.0

    def _ran(self, t0: float, ok: bool) -> None:
        self.calls += 1
        self.failed += 0 if ok else 1
        self.ms += (time.perf_counter() - t0) * 1000

    def complete(self, messages, *, temperature: float = 0.3) -> str:
        t0, ok = time.perf_counter(), False
        try:
            text_out = self.chat.complete(messages, temperature=temperature)
            ok = True
            return text_out
        finally:
            self._ran(t0, ok)

    def stream(self, messages, *, temperature: float = 0.3):
        t0, ok = time.perf_counter(), False
        try:
            yield from self.chat.stream(messages, temperature=temperature)
            ok = True
        finally:
            self._ran(t0, ok)


def auto_setup_data_if_needed(database_url: str) -> str:
    """Generate demo data when the database has almost none. Runs in the
    warm-up thread, never before the server listens."""
    if get_config().banko_skip_autosetup:
        print("Skipping data generation (--no-data or BANKO_SKIP_AUTOSETUP)", flush=True)
        return "skipped (BANKO_SKIP_AUTOSETUP)"
    # Persona history does not count: it can land first, and must never stop
    # the sample rows the search demo needs from being generated.
    from ..config.catalog import CatalogError, get_catalog
    try:
        personas = [p.user_id for p in get_catalog().personas]
    except CatalogError:
        personas = []
    with get_engine(database_url).connect() as conn:
        rows = conn.execute(text("""
            SELECT count(*) FROM (SELECT 1 FROM expenses
                                  WHERE user_id IS NULL OR user_id <> ALL(CAST(:personas AS UUID[]))
                                  LIMIT 100)
        """), {"personas": personas}).scalar() or 0
        if rows >= 100:
            # Only for the narration; this is the warm-up thread, not boot.
            total = conn.execute(text("SELECT count(*) FROM expenses")).scalar()
    if rows >= 100:
        print(f"✅ Database already contains {total} expense records", flush=True)
        return f"{total} expenses present"
    print("🔍 Checking database setup...", flush=True)
    print(f"Using database: {mask_url(database_url)}", flush=True)
    count = get_config().sample_data_count
    print(f"Generating {count} sample expense records...", flush=True)
    from ..vector_search.generator import EnhancedExpenseGenerator
    generated = EnhancedExpenseGenerator(database_url).generate_and_save(count, user_id=None, clear_existing=False)
    print(f"✅ Successfully generated {generated} expense records", flush=True)
    return f"generated {generated} sample expenses"


def create_app() -> Flask:
    """Create and configure the Flask application."""
    # The coach and migration modules narrate what they do (handling
    # signal, nudge delivered, claimed, replayed, regional migration)
    # through the banko.* loggers at INFO. Python drops INFO on the
    # floor without a handler, which makes the CLI go silent exactly
    # when the interesting things happen. Wire just our namespace to
    # the console; third-party loggers keep their own defaults.
    _banko_log = _coach_log.getLogger("banko")
    if not _banko_log.handlers:
        _h = _coach_log.StreamHandler()
        _h.setFormatter(_coach_log.Formatter("%(asctime)s %(name)s: %(message)s",
                                             datefmt="%H:%M:%S"))
        _banko_log.addHandler(_h)
        _banko_log.setLevel(_coach_log.INFO)

    # Get the directory containing this file
    # Set up template and static directories
    # Use package-relative paths that work both in development and PyPI installation
    current_dir = os.path.dirname(os.path.abspath(__file__))
    package_dir = os.path.dirname(current_dir)  # Go up to banko_ai package root
    
    template_dir = os.path.join(package_dir, 'templates')
    static_dir = os.path.join(package_dir, 'static')
    
    # Ensure directories exist
    if not os.path.exists(template_dir):
        # Fallback for PyPI installation
        import banko_ai
        package_root = os.path.dirname(banko_ai.__file__)
        template_dir = os.path.join(package_root, 'templates')
        static_dir = os.path.join(package_root, 'static')
    
    app = Flask(__name__, 
                template_folder=template_dir,
                static_folder=static_dir)
    
    # Load configuration
    config = get_config()
    app.config['SECRET_KEY'] = config.secret_key
    app.config['DEBUG'] = config.debug
    app.config.update(
        MAX_CONTENT_LENGTH=config.max_upload_mb * 1024 * 1024,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=config.session_cookie_secure,
    )

    # Initialize components
    user_manager = UserManager()
    cache_manager = BankoCacheManager()
    search_engine = VectorSearchEngine(config.database_url, cache_manager)
    
    # Initialize AI provider
    print(f"🔧 Initializing AI Provider: {config.ai_service}")
    print(f"   Environment AI_SERVICE: {config.ai_service}")
    try:
        ai_config = config.get_ai_config()
        ai_provider = AIProviderFactory.create_provider(
            config.ai_service,
            ai_config[config.ai_service],
            cache_manager
        )
        print(f"✅ AI Provider initialized: {ai_provider.get_provider_name() if ai_provider else 'None'}")
    except Exception as e:
        print(f"Warning: Could not initialize AI provider: {e}")
        ai_provider = None

    def _actual():
        return actual_identity(config.ai_service, ai_provider.get_current_model() if ai_provider else "")

    def _shown():
        return display_identity(config, _actual(), selected_model=app.config.get("DISPLAY_MODEL_OVERRIDE"))

    def _provider_ui():
        return ui_context(_shown(), status="connected" if ai_provider else "disconnected")

    def _switch_model(model: str) -> tuple[dict, int]:
        """Switch the model, or the persona's shown model. Returns the JSON
        payload and status for POST /api/models and /api/v1/ai/model."""
        try:
            if not ai_provider:
                return {'success': False, 'error': 'AI provider not available'}, 500

            if config.ai_display_provider:
                # Settings guarantee a persona always has at least one model to show.
                if model not in display_models(config, None):
                    return {'success': False, 'error': f'Model {model} is not available'}, 400
                app.config["DISPLAY_MODEL_OVERRIDE"] = model
                return {'success': True, 'message': f'Switched to {model}', 'current_model': model}, 200

            if ai_provider.set_model(model):
                return {'success': True, 'message': f'Switched to {model}',
                        'current_model': ai_provider.get_current_model()}, 200
            return {'success': False, 'error': f'Model {model} is not available'}, 400

        except Exception as e:
            print(f"model switch failed: {e}")
            return {'success': False, 'error': neutral_error(e)}, 500

    app.extensions["banko_ai"] = {
        "shown": _shown,
        "models": lambda: (display_models(config, None) if config.ai_display_provider
                           else display_models(config, ai_provider.get_available_models() if ai_provider else [])),
        "switch_model": _switch_model,
        "current_model": lambda: getattr(ai_provider, "current_model", None),
    }

    if config.ai_display_provider:
        shown, actual = _shown(), _actual()
        _coach_log.getLogger("banko.boot").info(
            "AI persona: shown as %s · %s, served by %s · %s",
            shown.provider, shown.model, actual.provider, actual.model)
    
    # --- Boot: listen first, warm up in the background -------------------
    readiness = Readiness()
    app.extensions["banko_readiness"] = readiness
    # Tests set this to end the warm-up lanes; nothing in the app does.
    warmup_stop = threading.Event()
    app.extensions["banko_warmup_stop"] = warmup_stop
    boot_log = _coach_log.getLogger("banko.boot")

    def _with_backoff(step: str, fn):
        """Retry a gate step until it works, so a database that is late or
        blips once never leaves pages gated until a restart."""
        delay = 1.0
        while True:
            try:
                return fn()
            except Exception as e:
                readiness.set(step, "running", f"waiting for the database ({e.__class__.__name__})")
                boot_log.warning("%s: database not ready yet (%s); retrying in %.0fs", step, e, delay)
                if warmup_stop.wait(delay):
                    raise RuntimeError("warm-up stopped") from e
                delay = min(delay * 2, 30.0)

    def _schema_step() -> str:
        result = _with_backoff("schema", lambda: ensure_schema(
            config.database_url,
            on_progress=lambda detail: readiness.set("schema", "running", detail)))
        from ..db.rls import refresh_status
        try:
            status = refresh_status(config.database_url)
            if config.rls_mode == "off":
                readiness.set("rls", "ready", "off by choice (RLS_MODE=off)")
            else:
                readiness.set("rls", "ready" if status.enabled else "degraded", status.reason)
        except Exception as e:  # the schema is fine; RLS just stays off
            # Class name only: the message could echo a URL with a password.
            boot_log.warning("rls: status check failed (%s); row-level security stays off",
                             e.__class__.__name__)
            readiness.set("rls", "degraded", "row-level security status could not be checked")
        if not result.changed:
            return f"current ({result.fingerprint})"
        return f"applied {result.statements} statements in {result.seconds:.1f}s"

    def _provider_step(narrate: bool = True) -> str:
        # The startup lines the CLI printed before boot moved here.
        service = config.ai_service.upper()
        if ai_provider is None or not bounded_connection_test(ai_provider):
            if narrate:
                print(f"🔧 {service} Available: False", flush=True)
                print(f"⚠️  {service} running in demo mode", flush=True)
            if ai_provider is None:
                raise RuntimeError("no AI provider configured")
            raise RuntimeError(f"{config.ai_service} did not answer")
        if narrate:
            print(f"🔧 {service} Available: True", flush=True)
            if config.ai_display_provider:
                shown, actual = _shown(), _actual()
                print(f"✅ Active AI Service: {shown.provider} (served by {actual.provider})", flush=True)
            else:
                print(f"✅ Active AI Service: {service}", flush=True)
        return f"{config.ai_service} reachable"

    # The catalog is plain YAML: load it once, now, so a broken override
    # file fails the personas step with a reason instead of being retried
    # forever as if the database were late.
    from ..config.catalog import CatalogError, get_catalog
    catalog_error: str | None = None
    try:
        get_catalog()
    except CatalogError as e:
        catalog_error = str(e)
        boot_log.error("catalog: %s; the personas will not load until the file is fixed", catalog_error)

    def _personas_step() -> str:
        if catalog_error:
            raise CatalogError(catalog_error)
        from ..services.seeding import ensure_personas
        names = _with_backoff("personas", lambda: ensure_personas(config.database_url))
        return f"{', '.join(names)} present"

    def _expiry_step() -> str:
        from banko_ai.utils.migration import DatabaseMigration
        return f"{DatabaseMigration(config.database_url).expire_demo_users()} expired"

    def _model_step() -> str:
        from ..utils.embeddings import load_embedding_model
        load_embedding_model(wait_seconds=600)
        return f"{config.embedding_model} on {config.embedding_device}"

    def _coach_step() -> str:
        # Both helpers are closures defined further down in create_app. The
        # warm-up thread only starts at the very end of create_app (see
        # below), so they always exist by the time this runs. Kafka reports
        # its own state as the "kafka" component. The replay can call the
        # LLM for every stranded signal, minutes on a CPU Ollama, so it gets
        # its own thread instead of holding up locality and sample data.
        _maybe_start_kafka_consumer()
        threading.Thread(target=_replay_stranded_on_boot,
                         name="coach-stranded-replay", daemon=True).start()
        return "webhook active, kafka enabled" if config.coach_kafka_enabled else "webhook active"

    # /health covers /health/coach, which the pipeline polls during warm-up.
    _READINESS_EXEMPT = ("/api/health", "/health", "/static/", "/socket.io", "/favicon")
    _WARMING_UP_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="3"><title>Banko is getting ready</title>
<style>body{font-family:Inter,system-ui,sans-serif;background:#F6F5F2;color:#1F1D1A;display:grid;place-items:center;height:100vh;margin:0}
.card{background:#fff;border:1px solid #ECE9E2;border-radius:14px;padding:28px 32px;max-width:440px}
small{color:#7A756C}</style></head><body><div class="card"><h2>Banko is getting ready</h2>
<p>{{ detail }}</p><small>This page refreshes on its own.</small></div></body></html>"""

    # Pages wait for the schema and the three personas. Without the
    # personas, typing "maya" at /login would offer signup and create an
    # ordinary user that the backfill then never turns into the persona.
    _GATE = ("schema", "personas")

    def _gate_ready() -> bool:
        return all(readiness.is_ready(name) for name in _GATE)

    def _wait_for_gate(timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        return all(readiness.wait_for(name, max(0.0, deadline - time.monotonic())) for name in _GATE)

    def _gate_detail() -> str:
        for name in _GATE:
            component = readiness.get(name)
            if component is None or component.state != "ready":
                if component is not None and component.detail:
                    return component.detail
                return "Setting up the database." if name == "schema" else "Setting up the demo personas."
        return ""

    @app.before_request
    def _wait_for_schema():
        if request.path.startswith(_READINESS_EXEMPT) or _gate_ready():
            return None
        if _wait_for_gate(config.warmup_wait_seconds):
            return None
        detail = _gate_detail()
        if request.path.startswith("/api/"):
            return jsonify({"error": "warming_up", "detail": detail}), 503
        return render_template_string(_WARMING_UP_PAGE, detail=detail), 503

    @app.errorhandler(ModelWarmingUp)
    def _model_warming_up(e):
        detail = "The embedding model is still loading. Try again in a few seconds."
        if request.path.startswith("/api/"):
            return jsonify({"error": "warming_up", "detail": detail}), 503
        return render_template_string(_WARMING_UP_PAGE, detail=detail), 503

    @app.before_request
    def _set_request_scope():
        from ..utils.request_scope import set_user_scope
        g._banko_scope = set_user_scope(session.get('user_id') or config.coach_default_user_id)

    @app.teardown_request
    def _reset_request_scope(exc):
        from ..utils.request_scope import reset_user_scope
        token = g.pop('_banko_scope', None)
        if token is not None:
            reset_user_scope(token)

    @app.errorhandler(413)
    def too_large(error):
        if request.path == '/api/upload-receipt':
            msg = f'Receipts can be up to {config.max_upload_mb} MB'
        else:
            msg = 'Request body too large'
        return jsonify({'success': False, 'error': msg}), 413

    def _emit_seeding(user_id: str, done: int, total: int, state: str) -> None:
        socketio.emit("seeding", {"user_id": user_id, "done": done, "total": total, "state": state})

    # Personas seed in parallel; health shows their combined progress.
    persona_counts: dict[str, tuple[int, int]] = {}
    persona_counts_lock = threading.Lock()

    def _persona_progress(user_id: str, done: int, total: int, state: str) -> None:
        with persona_counts_lock:
            persona_counts[user_id] = (done, total)
            all_done = sum(d for d, _ in persona_counts.values())
            all_total = sum(t for _, t in persona_counts.values())
        if state == "running":
            readiness.set("persona_data", "running", f"seeding {all_done}/{all_total}")
        _emit_seeding(user_id, done, total, state)

    def _integrity(report) -> None:
        if report.skipped:
            readiness.set("integrity", "ready", f"skipped: {report.skipped}")
        elif report.drifted:
            readiness.set("integrity", "degraded", f"drift on {len(report.drifted)} of {report.accounts} accounts")
        else:
            readiness.set("integrity", "ready", f"0 drift across {report.accounts} accounts")

    def _persona_data_step() -> str:
        from ..services.integrity import IntegrityWorker
        if config.banko_skip_autosetup:
            # No seeding and no integrity sweeps: tests and --no-data runs.
            readiness.set("integrity", "ready", "skipped (BANKO_SKIP_AUTOSETUP)")
            return "skipped (BANKO_SKIP_AUTOSETUP)"
        try:
            if catalog_error:
                raise CatalogError(catalog_error)
            readiness.wait_for("personas", timeout=600)
            from ..services.seeding import seed_personas
            written = seed_personas(config.database_url, on_progress=_persona_progress)
            return ", ".join(f"{name} +{n}" for name, n in written.items())
        finally:
            IntegrityWorker(url=config.database_url, stop=warmup_stop, on_report=_integrity).start()

    def _assistant_step() -> str:
        # Builds the checkpointer once: the checkpoint tables and their TTL on
        # a first boot, a no-op afterwards, so the first question never pays for it.
        if config.banko_skip_autosetup:
            return "skipped (BANKO_SKIP_AUTOSETUP)"
        if not readiness.wait_for("schema", timeout=600):
            raise RuntimeError("the schema is not ready yet; the checkpointer builds on the first question")
        from ..assistant.checkpoint import get_saver
        get_saver()
        return "checkpointer ready"

    def _emit_welcome_signal(user: dict) -> None:
        """Post a welcome signal matched to the user's spending style.

        The POST goes back to this same server, so it runs on a background
        thread: a synchronous self-call inside the signup request would
        deadlock a single-threaded server."""
        secret = get_config().cdc_webhook_hmac_secret
        if not secret:
            print("welcome signal skipped: no webhook secret")
            return
        sig_type, severity, payload = {
            "subscriber": ("recurring_drift", "info",
                           {"subscription": "Netflix", "old_amount": 15.99,
                            "new_amount": 22.99, "pct_change": 0.44,
                            "merchant_id": str(uuid.uuid4())}),
        }.get(user["spending_style"],
              ("budget_threshold", "info",
               {"category": "dining", "pct_used": 0.55,
                "monthly_budget": 400.0, "spent_so_far": 220.0,
                "days_remaining": 12}))
        envelope = {"payload": [{"after": {
            "signal_id": str(uuid.uuid4()), "user_id": user["user_id"],
            "signal_type": sig_type, "severity": severity,
            "payload": payload,
            "idempotency_key": f"welcome:{user['user_id']}"},
            "updated": f"{time.time():.10f}"}]}
        body = json.dumps(envelope).encode()
        mac = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        # host_url needs the request context; resolve it before the thread.
        url = request.host_url.rstrip('/') + '/api/cdc/signals'

        def _post() -> None:
            try:
                requests.post(
                    url, data=body, timeout=5,
                    headers={"Content-Type": "application/json",
                             "X-Banko-Signature": mac})
            except Exception as e:
                print(f"welcome signal skipped: {e}")

        import threading
        threading.Thread(target=_post, daemon=True,
                         name="welcome-signal").start()

    @app.route('/')
    def index():
        """Main application page."""
        # First visit goes to the persona picker; everything downstream
        # scopes to the chosen persona.
        if not session.get('user_id'):
            return redirect(url_for('login'))
        current_user = {'id': session['user_id'],
                        'username': session.get('username', 'Demo')}

        # Get AI provider info for display
        ai_provider_display = _provider_ui()
        
        return render_template('index.html', 
                             user=current_user,
                             ai_provider=ai_provider_display)
    
    @app.route('/login', methods=['GET', 'POST'])
    def login():
        """Login for known users, reveal signup for unknown usernames."""
        if request.method == 'POST':
            username = request.form.get('username', '').strip()
            if not username:
                flash('Username required', 'error')
                return render_template('login.html', regions=[])

            um = UserManager(config.database_url)
            user = um.get_by_username(username)
            if user:
                session['user_id'] = user["user_id"]
                session['username'] = user["username"]
                return redirect(url_for('index'))

            # Offer only regions the database has: an operator-configured
            # database can carry fewer regions than the cluster, and rows
            # can never pin to a region outside its enum.
            regions = detect_database_regions(config.database_url)
            return render_template('login.html', username=username,
                                   regions=regions, show_signup=True)

        return render_template('login.html', regions=[])

    @app.route('/signup', methods=['POST'])
    def signup():
        """Create user, seed history, emit welcome signal, log in."""
        username = request.form.get('username', '').strip()
        spending_style = request.form.get('spending_style', '').strip()
        home_region = request.form.get('home_region', None)

        # Bound the username: it lands in headers, session cookies, and
        # SQL parameters on every page, so no novels and no control bytes.
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,32}", username):
            flash('Usernames are 1-32 characters: letters, digits, '
                  'dot, dash, underscore.', 'error')
            return render_template('login.html', username='',
                                   regions=detect_database_regions(config.database_url),
                                   show_signup=True)

        allowed_styles = ['diner', 'subscriber', 'saver', 'balanced']
        if spending_style not in allowed_styles:
            flash(f'Invalid spending style. Choose from: {", ".join(allowed_styles)}', 'error')
            return render_template('login.html', username=username,
                                   regions=detect_database_regions(config.database_url),
                                   show_signup=True)

        # Only accept a region the database actually has; the cluster can
        # know regions the database's enum does not, and anything else
        # (including any value on a single-region deployment) is dropped so
        # a crafted form post can never poison home_region.
        if home_region and home_region not in detect_database_regions(config.database_url):
            home_region = None

        um = UserManager(config.database_url)

        # A taken username is a sign-in, not an error: never touch the
        # existing account from the signup path.
        if um.get_by_username(username):
            flash(f'"{username}" already exists. Sign in below to continue.', 'info')
            return render_template('login.html', username=username,
                                   regions=detect_database_regions(config.database_url))

        user = None
        try:
            user = um.create(username, spending_style, home_region, demo_user=True)
            def _seed_then_welcome(new_user=user):
                from ..services.seeding import seed_user
                try:
                    seed_user(new_user["user_id"], on_progress=_emit_seeding)
                    _emit_welcome_signal(new_user)
                except Exception as exc:
                    print(f"⚠️  seeding {new_user['username']} failed: {exc}")
            threading.Thread(target=_seed_then_welcome, daemon=True,
                             name=f"seed-{user['username']}").start()

            session['user_id'] = user["user_id"]
            session['username'] = user["username"]
            return redirect(url_for('index'))
        except Exception as e:
            # Roll back only what this request created, including any
            # partially seeded history, so a failure never orphans rows
            # and a create-time race never deletes someone else's account.
            if user is not None:
                try:
                    with um.engine.connect() as conn:
                        conn.execute(text("DELETE FROM expenses WHERE user_id = :uid"),
                                     {"uid": user["user_id"]})
                        conn.commit()
                except Exception as cleanup_err:
                    print(f"signup rollback: expense cleanup failed: {cleanup_err}")
                um.delete_by_username(username)
            flash(f'Signup failed: {e}', 'error')
            return render_template('login.html', username=username,
                                   regions=detect_database_regions(config.database_url),
                                   show_signup=True)

    def current_demo_user() -> str:
        """The session persona, falling back to the coach default so API
        callers without a session still get scoped, sensible answers."""
        return session.get('user_id') or config.coach_default_user_id

    # UI pages require a persona so it is always clear whose data is on
    # screen. APIs, the CDC webhook, and health endpoints keep the default
    # persona fallback so curl demos and the pipeline never need a session.
    _SESSION_EXEMPT_PREFIXES = (
        '/login', '/logout', '/static/', '/api/', '/health', '/cache-stats',
        '/cache-cleanup', '/socket.io', '/favicon',
    )

    @app.before_request
    def _require_persona_for_pages():
        if request.method != 'GET':
            return None
        path = request.path
        if any(path.startswith(p) for p in _SESSION_EXEMPT_PREFIXES):
            return None
        if session.get('user_id'):
            return None
        return redirect(url_for('login'))

    @app.route('/logout')
    def logout():
        """Back to the persona picker."""
        session.pop('user_id', None)
        session.pop('username', None)
        user_manager.logout_user()
        return redirect(url_for('login'))

    @app.context_processor
    def inject_identity():
        """Who is signed in, for the header pill on every page. The region
        is the user's home region (where their rows live), shown only on
        regional deployments; the per-answer badge shows where each query
        was actually served."""
        username = session.get('username')
        region = None
        if username and session.get('user_id'):
            try:
                from ..db.regions import regional_tables_ready
                if regional_tables_ready(config.database_url):
                    from .auth import resolve_user_region
                    region = resolve_user_region(session['user_id'], config.database_url)
            except Exception:
                region = None
        return {'current_username': username, 'current_user_region': region}
    
    @app.route('/api/search', methods=['POST'])
    def api_search():
        """API endpoint for expense search."""
        try:
            data = request.get_json()
            query = data.get('query', '')
            limit = data.get('limit', 10)
            threshold = data.get('threshold', 0.7)
            # Use original simple logic - no user filtering
            results = search_engine.search_expenses(
                query=query,
                user_id=current_demo_user(),
                limit=limit,
                threshold=threshold
            )
            
            # Convert to serializable format
            search_results = []
            for result in results:
                search_results.append({
                    'expense_id': result.expense_id,
                    'user_id': result.user_id,
                    'description': result.description,
                    'merchant': result.merchant,
                    'amount': result.amount,
                    'date': result.date,
                    'similarity_score': result.similarity_score,
                    'metadata': result.metadata
                })
            
            return jsonify({
                'success': True,
                'results': search_results,
                'query': query,
                'user_id': None
            })
            
        except ModelWarmingUp:
            raise
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/rag', methods=['POST'])
    def api_rag():
        """Ask the assistant without the page (curl demos): the same graph as
        the chat page, on this session's thread."""
        try:
            data = request.get_json(silent=True) or {}
            query = str(data.get('query', '')).strip()
            if not query:
                return jsonify({'success': False, 'error': 'query is required'}), 400
            language = data.get('language', 'en')
            code = language if language in ('en-US', 'es-ES', 'fr-FR', 'de-DE', 'it-IT', 'pt-PT', 'ja-JP',
                                            'ko-KR', 'zh-CN', 'hi-IN') else None
            from ..assistant import answers
            from ..assistant.graph import run_turn
            from .assistant_api import _chat
            out = run_turn(current_demo_user(), _turn_thread(), query, language_code=code, chat=_chat())
            # The identity the persona shows, as this endpoint always reported it.
            shown = _shown()
            return jsonify({'success': True, 'response': answers.strip_citations(out['answer']),
                            'sources': out['citations'],
                            'metadata': {'route': (out.get('route') or {}).get('kind'), 'cached': out['cached'],
                                         'region': (out.get('meta') or {}).get('region'),
                                         'provider': shown.provider, 'model': shown.model}})
        except ModelWarmingUp:
            raise
        except Exception as e:
            print(f"RAG error: {e}")
            return jsonify({'success': False, 'error': neutral_error(e)}), 500
    
    # OLD ROUTE REMOVED - replaced by /api/generate-data with SocketIO support below
    
    # ── langchain-cockroachdb powered vectorstore search ──────────────────
    @app.route('/api/vectorstore-search', methods=['POST'])
    def api_vectorstore_search():
        """Semantic search via langchain-cockroachdb CockroachDBVectorStore."""
        try:
            data = request.get_json()
            query = data.get('query', '')
            limit = data.get('limit', 5)
            metadata_filter = data.get('filter')

            from banko_ai.vector_search.crdb_vectorstore import search_expenses_via_vectorstore
            results = search_expenses_via_vectorstore(query, limit=limit, metadata_filter=metadata_filter)

            return jsonify({'success': True, 'results': results, 'query': query, 'source': 'cockroachdb-vectorstore'})
        except ModelWarmingUp:
            raise
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

    # ── langchain-cockroachdb chat history ─────────────────────────────────
    @app.route('/api/chat-history/<session_id>', methods=['GET'])
    def api_chat_history(session_id):
        """Retrieve persistent chat history for a session."""
        try:
            from banko_ai.utils.crdb_chat_history import get_chat_history
            history = get_chat_history(session_id, database_url=config.database_url)
            messages = [
                {"role": m.type, "content": m.content}
                for m in history.messages
            ]
            return jsonify({'success': True, 'session_id': session_id, 'messages': messages})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/api/chat-history/<session_id>', methods=['DELETE'])
    def api_clear_chat_history(session_id):
        """Clear chat history for a session."""
        try:
            from banko_ai.utils.crdb_chat_history import get_chat_history
            history = get_chat_history(session_id, database_url=config.database_url)
            history.clear()
            return jsonify({'success': True, 'cleared': session_id})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/api/user-summary')
    def api_user_summary():
        """API endpoint for user spending summary. Served with a follower
        read: totals tolerate a few seconds of staleness."""
        try:
            # The old is_logged_in() died with the persona-era auth stub;
            # session scoping works like every other API route now.
            user_id = current_demo_user()
            summary = search_engine.get_user_spending_summary(user_id)
            
            return jsonify({
                'success': True,
                'summary': summary
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/ai-providers')
    def api_ai_providers():
        """API endpoint for available AI providers."""
        try:
            providers = AIProviderFactory.get_available_providers()
            current_provider = _shown().provider

            return jsonify({
                'success': True,
                'providers': providers,
                'current': current_provider
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/models')
    def api_models():
        """API endpoint for available models for current provider."""
        try:
            if not ai_provider:
                return jsonify({
                    'success': False,
                    'error': 'AI provider not available'
                }), 500

            shown = _shown()
            listed = display_models(config, None) if config.ai_display_provider else None
            models = listed or display_models(config, ai_provider.get_available_models())
            return jsonify({'success': True, 'models': models, 'current': shown.model, 'provider': shown.provider})

        except Exception as e:
            print(f"model list failed: {e}")
            return jsonify({
                'success': False,
                'error': neutral_error(e)
            }), 500
    
    @app.route('/api/models', methods=['POST'])
    def api_set_model():
        """API endpoint for switching models."""
        try:
            if not ai_provider:
                return jsonify({
                    'success': False,
                    'error': 'AI provider not available'
                }), 500

            data = request.get_json()
            model = data.get('model')

            if not model:
                return jsonify({
                    'success': False,
                    'error': 'Model name is required'
                }), 400

        except Exception as e:
            print(f"model switch failed: {e}")
            return jsonify({
                'success': False,
                'error': neutral_error(e)
            }), 500

        payload, status = _switch_model(model)
        return jsonify(payload), status

    def _db_ping() -> str | None:
        """None when the database answers, otherwise the error."""
        try:
            with get_engine(config.database_url).connect() as conn:
                conn.execute(text("SELECT 1"))
            return None
        except Exception as e:
            return f"database unreachable: {e.__class__.__name__}"

    @app.route('/api/health')
    def api_health():
        """Cheap health: database ping plus warm-up state. Pass ?refresh=1 to
        re-test the AI provider now."""
        if request.args.get("refresh") == "1":
            try:
                readiness.set("ai_provider", "ready", _provider_step(narrate=False))
            except Exception as e:
                readiness.set("ai_provider", "failed", str(e))
        db_error = _db_ping()
        db_ok = db_error is None
        snap = readiness.snapshot()
        provider_state = snap.get("ai_provider", {}).get("state", "unknown")
        body = {
            'success': db_ok,
            'database': 'connected' if db_ok else 'unreachable',
            'ai_provider': 'connected' if provider_state == 'ready' else provider_state,
            'ai_service': _shown().provider,
            'current_model': _shown().model,
            'ai_provider_available': ai_provider is not None,
            # components are engineer-facing (carry the actual provider by design); the dashboard doesn't render them
            'components': snap,
            'display': {'provider': _shown().provider, 'model': _shown().model},
            'actual': {'provider': _actual().provider, 'model': _actual().model},
        }
        if db_error:
            body['error'] = db_error
        return jsonify(body), (200 if db_ok else 503)

    @app.route('/api/health/live')
    def health_live():
        return jsonify({'status': 'alive'}), 200

    @app.route('/api/health/ready')
    def health_ready():
        ready = _gate_ready()
        snap = readiness.snapshot()
        return jsonify({'ready': ready, **{name: snap.get(name) for name in _GATE}}), (200 if ready else 503)

    @app.route('/api/upload-receipt', methods=['POST'])
    def upload_receipt():
        """The chat page's receipt upload: the assistant's receipt pipeline,
        answered in the shape the page reads. The receipt becomes the active
        receipt of the session's thread, so the next question can ask about it."""
        from pathlib import Path

        from ..assistant.graph import set_active_receipt
        from ..db.regions import agent_home_region
        from ..services import receipts as receipt_service
        file = request.files.get('receipt')
        if file is None or file.filename == '':
            return jsonify({'success': False, 'error': 'No receipt file provided'}), 400
        suffix = Path(file.filename).suffix.lower()
        if suffix not in receipt_service.ALLOWED_TYPES:
            return jsonify({'success': False, 'error': 'Upload a receipt image or PDF'}), 400
        handle = tempfile.NamedTemporaryFile(prefix="banko_receipt_", suffix=suffix, delete=False)
        handle.close()
        try:
            file.save(handle.name)
            hooks = receipt_service.default_hooks(getattr(ai_provider, 'current_model', None))
            # The dashboard shows "agent (region)", like every other agent_activity event.
            agent_region = agent_home_region(config.database_url)
            checks: dict[str, str] = {}
            final: dict = {}
            for event in receipt_service.process_receipt(current_demo_user(), handle.name, file.filename,
                                                         hooks=hooks):
                if event['type'] == 'step' and event['status'] != 'start':
                    checks[event['step']] = event['detail']
                    try:
                        socketio.emit('agent_activity', {
                            'agent_type': {'fraud': 'fraud', 'budget': 'budget'}.get(event['step'], 'receipt'),
                            'region': agent_region,
                            'status': 'completed' if event['status'] == 'done' else event['status'],
                            'message': f"{event['step']}: {event['detail']}",
                            'timestamp': datetime.now().isoformat()})
                    except Exception:
                        pass
                elif event['type'] in ('receipt', 'error'):
                    final = event
            if final.get('type') != 'receipt':
                message = final.get('message') or NEUTRAL_FAILED
                status = 422 if message in (receipt_service.TEMPLATE_ECHO, receipt_service.INVALID) else 500
                return jsonify({'success': False, 'error': message}), status
            try:
                set_active_receipt(_turn_thread(), {k: final.get(k) for k in (
                    'document_id', 'merchant', 'amount', 'date', 'category', 'items', 'expense_id')})
            except Exception as e:
                # The purchase has posted by now: report it, just without a follow-up thread.
                print(f"⚠️  Receipt {final.get('document_id')} posted but not attached to the chat: {e}")
            duplicate = final.get('status') == 'duplicate'
            return jsonify({
                'success': True, 'merchant': final.get('merchant'), 'amount': f"{float(final.get('amount') or 0):.2f}",
                'category': final.get('category'), 'date': final.get('date'), 'items': final.get('items') or [],
                'expense_id': final.get('expense_id'), 'document_id': final.get('document_id'),
                # A duplicate posts nothing, so its checks never run; the card says why instead.
                'fraud_status': checks.get('fraud') or (f"Duplicate: {checks.get('duplicate')}" if duplicate
                                                        else 'Not checked'),
                'budget_impact': checks.get('budget') or (checks.get('post') if duplicate else 'Not checked'),
                'duplicate': duplicate,
                'message': ('Duplicate receipt, nothing posted' if duplicate
                            else 'Receipt processed by Receipt, Fraud, and Budget agents')})
        except Exception as e:
            from werkzeug.exceptions import HTTPException
            if isinstance(e, HTTPException):
                raise
            print(f"❌ Receipt upload error: {e}")
            return jsonify({'success': False, 'error': NEUTRAL_FAILED}), 500
        finally:
            try:
                os.remove(handle.name)
            except OSError:
                pass
    
    def _trace_identity() -> str:
        """The model as the persona shows it, for LLM steps in the trace."""
        shown = _shown()
        return f"{shown.name} · {shown.model}"

    def _finish_trace(plan: str | None = None) -> str:
        """Close this request's trace and park it server side, with the
        answer's EXPLAIN plan. The chat history lives in the signed session
        cookie (4 KB, and a plan can be 20 KB), so the message keeps only
        the short id."""
        trace_step("total", "Total request time", ms=trace_elapsed_ms())
        return store_trace(current_trace(), plan)

    # Browsers silently ignore a cookie over 4 KB, and every session change
    # riding on it, a new thread id included. Each turn adds a few hundred
    # bytes, so a long session sheds its oldest turns first.
    _SESSION_COOKIE_BUDGET = 3800

    def _fit_chat_in_cookie() -> None:
        signer = getattr(app.session_interface, "get_signing_serializer", None)
        serializer = signer(app) if signer else None
        chat_log = session.get('chat') or []
        while (serializer is not None and len(chat_log) > 2
               and len(serializer.dumps(dict(session))) > _SESSION_COOKIE_BUDGET):
            del chat_log[:2]

    def _render_chat(ai_provider_display):
        """Render the chat page with each message's stored trace and plan
        attached for the template; the session itself keeps only the ids."""
        view = []
        for message in session.get('chat', []):
            meta = message.get('meta')
            if meta and meta.get('trace_id'):
                message = dict(message)
                message['meta'] = dict(meta, trace=get_stored_trace(meta['trace_id']) or [],
                                       explain=get_stored_plan(meta['trace_id']))
            view.append(message)
        return render_template('index.html',
                               chat=view,
                               ai_provider=ai_provider_display,
                               current_page='banko')

    @app.route('/banko', methods=['GET', 'POST'])
    def chat():
        """Main chat interface. Each request records its own trace, which
        the DB Ops panel shows for the answer it produced."""
        token = start_trace()
        try:
            return _chat_impl()
        finally:
            reset_trace(token)

    def _session_thread() -> str:
        """The chat page's thread: one per browser session and user."""
        from ..services import threads as thread_service
        user_id = current_demo_user()
        tid = session.get('thread_id')
        if not tid or thread_service.get_thread(user_id, tid) is None:
            tid = thread_service.create_thread(user_id)['thread_id']
            session['thread_id'] = tid
        return tid

    def _turn_thread() -> str:
        """A signed-in browser keeps its session's thread. Calls without a
        signed-in session share one stable thread per user, so curl demos
        never open a new thread per request."""
        if session.get('user_id'):
            return _session_thread()
        return api_thread_id(current_demo_user())

    def _chat_impl():
        if request.method == 'GET':
            session['chat'] = []
            session.pop('thread_id', None)
        elif 'chat' not in session:
            session['chat'] = []
        ai_provider_display = _provider_ui()
        if request.method != 'POST':
            return _render_chat(ai_provider_display)
        user_message = (request.form.get('user_input') or request.form.get('message') or '').strip()
        if not user_message:
            return _render_chat(ai_provider_display)
        session['chat'].append({'text': user_message, 'class': 'User'})
        # Flask only notices assignments to the session, not appends to the
        # list inside it; without this, only the first turn ever persisted.
        session.modified = True
        language = request.form.get('response_language', 'en-US')
        try:
            from ..assistant import answers
            from ..assistant.graph import run_turn
            from .assistant_api import _chat
            timed = _TimedChat(_chat())
            out = run_turn(current_demo_user(), _turn_thread(), user_message,
                           language_code=language, chat=timed)
            if timed.calls:
                # Cache hits, memory commands and English redirects never
                # reach the model, so their trace does not name one.
                detail = f"{_trace_identity()}, {timed.calls} call{'s' if timed.calls > 1 else ''}"
                trace_step("llm", "Model", detail + (f", {timed.failed} failed" if timed.failed else ""),
                           ms=timed.ms, status="warning" if timed.failed else "success")
            meta = out.get('meta') or {}
            # Only what the turn measured: the template prints any db_ms it is given.
            measured = {k: meta[k] for k in ('region', 'db_ms') if meta.get(k) is not None}
            session['chat'].append({'text': answers.strip_citations(out['answer']), 'class': 'Assistant',
                                    'meta': {**measured, 'trace_id': _finish_trace(meta.get('explain'))}})
        except Exception as e:
            if isinstance(e, ModelWarmingUp):
                error_message = "I'm still warming up my embedding model. Try that again in a few seconds."
                trace_detail = "The embedding model is still loading"
            elif isinstance(e, TRANSIENT_ERRORS):
                error_message = "Reconnecting to the database, try that again in a moment."
                trace_detail = "The database connection dropped; nothing was answered"
            else:
                print(f"Error with {config.ai_service}: {str(e)}")
                error_message = neutral_error(e)
                trace_detail = "The answer failed here; details are in the server log"
            trace_step("error", "Answer stopped", trace_detail, status="error")
            session['chat'].append({'text': error_message, 'class': 'Assistant',
                                    'meta': {'trace_id': _finish_trace()}})
        _fit_chat_in_cookie()
        return _render_chat(ai_provider_display)

    @app.route('/home')
    def dashboard():
        return render_template('dashboard.html', current_page='home')

    @app.route('/savings')
    def savings():
        return render_template('dashboard.html', current_page='savings')

    @app.route('/wallet')
    def wallet():
        return render_template('dashboard.html', current_page='wallet')

    @app.route('/credit-card')
    def credit_card():
        return render_template('dashboard.html', current_page='credit-card')

    @app.route('/statements')
    def statements():
        return render_template('dashboard.html', current_page='statements')

    @app.route('/benefits')
    def benefits():
        return render_template('dashboard.html', current_page='benefits')

    @app.route('/settings')
    def settings():
        ai_provider_display = _provider_ui()
        return render_template('dashboard.html',
                             current_page='settings',
                             ai_provider=ai_provider_display)

    @app.route('/cache-stats')
    def cache_stats():
        """Endpoint to view cache performance statistics"""
        if not cache_manager:
            return {'error': 'Cache manager not available'}, 503
        
        try:
            stats = cache_manager.get_cache_stats(hours=24)
            
            # Calculate overall hit rate
            total_requests = 0
            total_hits = 0
            
            for cache_type, cache_stats in stats.items():
                if cache_type != 'total_tokens_saved':
                    total_requests += cache_stats.get('hits', 0) + cache_stats.get('misses', 0)
                    total_hits += cache_stats.get('hits', 0)
            
            overall_hit_rate = (total_hits / total_requests) if total_requests > 0 else 0
            
            return {
                'success': True,
                'cache_enabled': True,
                'overall_hit_rate': overall_hit_rate,
                'total_requests': total_requests,
                'total_hits': total_hits,
                'total_tokens_saved': stats.get('total_tokens_saved', 0),
                'cache_details': stats
            }
        except Exception as e:
            return {'error': f'Failed to get cache stats: {str(e)}'}, 500

    @app.route('/cache-cleanup', methods=['POST'])
    def cache_cleanup():
        """Endpoint to manually trigger cache cleanup"""
        if not cache_manager:
            return {'error': 'Cache manager not available'}, 503
        
        try:
            cache_manager.cleanup_expired_cache()
            return {'message': 'Cache cleanup completed successfully'}
        except Exception as e:
            return {'error': f'Cache cleanup failed: {str(e)}'}, 500

    # Initialize SocketIO for real-time updates (needed before data generator routes)
    origins = [o.strip() for o in config.socketio_cors_origins.split(",") if o.strip()]
    socketio = SocketIO(app, cors_allowed_origins=origins or None, async_mode='threading')
    app.socketio = socketio

    # --- Coach v1 webhook receiver ---------------------------------------
    coach_log = _coach_log.getLogger("banko.coach.webhook")

    def _verify_signature(body: bytes, header_sig: str | None) -> bool:
        secret = get_config().cdc_webhook_hmac_secret
        if not secret:
            coach_log.warning("CDC_WEBHOOK_HMAC_SECRET not set; rejecting all "
                              "incoming webhooks")
            return False
        if not header_sig:
            return False
        expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, header_sig)

    def _socketio_emitter(event: str, payload: dict, room: str | None = None):
        if room:
            app.socketio.emit(event, payload, to=room)
        else:
            app.socketio.emit(event, payload)

    def _get_coach_handler() -> SignalHandler:
        handler = getattr(app, "_coach_handler", None)
        if handler is not None:
            return handler
        from ..coach.agent import CoachAgent, default_llm_invoker
        cfg = get_config()
        db_url = get_database_url()
        agent = CoachAgent(
            database_url=db_url,
            llm_invoker=default_llm_invoker,
            provider_name=cfg.ai_service,
            max_steps=cfg.coach_agent_max_steps,
        )
        handler = SignalHandler(
            coach=agent,
            emitter=type("E", (), {"emit": staticmethod(_socketio_emitter)})(),
            database_url=db_url,
            socketio_room_prefix=cfg.coach_socketio_room_prefix,
        )
        app._coach_handler = handler
        return handler

    def _maybe_start_kafka_consumer() -> None:
        """Start the flag-gated Kafka transport in a daemon thread. The
        webhook stays active either way; both transports share the handler,
        so idempotency dedups any event that arrives twice."""
        cfg = get_config()
        if not cfg.coach_kafka_enabled:
            return
        if getattr(app, "_coach_kafka_thread", None) is not None:
            return
        import threading

        bootstrap = cfg.kafka_bootstrap_servers
        topic = cfg.coach_kafka_topic
        # degraded, not running: kafka-python can take about 30 s to give
        # up on a missing broker, and the ready line must not wait for it.
        readiness.set("kafka", "degraded", f"connecting to {bootstrap}; webhook active")

        def _connect_and_run() -> None:
            # Kafka is an optional transport; an unreachable broker must
            # not take the app down, and starting the broker after the app
            # should just work. Keep retrying with backoff until the
            # brokers appear, then hand off to the consumer loop (which
            # handles reconnects itself once established).
            from ..coach.kafka_consumer import build_production_consumer
            delay, attempt = 5, 0
            while True:
                try:
                    consumer = build_production_consumer(
                        handler=_get_coach_handler(),
                        bootstrap_servers=bootstrap,
                        topic=topic,
                    )
                    break
                except Exception as e:
                    attempt += 1
                    readiness.set("kafka", "degraded",
                                  f"retrying: brokers unreachable at {bootstrap}; webhook active")
                    if attempt == 1 or attempt % 10 == 0:
                        print(f"⚠️  Coach Kafka brokers unreachable at "
                              f"{bootstrap} ({e}); retrying every {delay}s. "
                              f"The webhook transport remains active.", flush=True)
                    if warmup_stop.wait(delay):
                        readiness.set("kafka", "failed", "warm-up stopped")
                        return
                    delay = min(delay * 2, 30)
            readiness.set("kafka", "ready", f"connected (topic={topic})")
            print(f"📡 Coach Kafka consumer started (topic={topic}, "
                  f"brokers={bootstrap})", flush=True)
            consumer.run_forever()

        t = threading.Thread(target=_connect_and_run,
                             name="coach-kafka-consumer", daemon=True)
        t.start()
        app._coach_kafka_thread = t

    def _replay_stranded_on_boot() -> None:
        """A process dying mid nudge leaves its signal claimed with no
        nudge to show. Recover in the background so a demo-box restart
        never eats a welcome nudge."""
        try:
            from ..coach.handler import replay_stranded_signals
            n = replay_stranded_signals(_get_coach_handler(),
                                        config.database_url)
            if n:
                print(f"🔁 replayed {n} stranded coach signal(s) from "
                      f"before the last restart", flush=True)
        except Exception as e:
            print(f"stranded-signal replay skipped: {e}", flush=True)

    def _claim_signal(sig) -> bool:
        """Atomically claim a signal for processing by inserting its row
        into spending_signals. Returns True if this call claimed it
        (i.e. this is the first time we're seeing this idempotency_key),
        False if a previous call already claimed it (replay).

        This makes the webhook safely re-postable: the pipeline can fire
        the same envelope twice without producing duplicate nudges."""
        from sqlalchemy.exc import IntegrityError

        from banko_ai.db.regions import regional_tables_ready
        from banko_ai.web.auth import resolve_user_region as _resolve_region

        from ..db.engine import get_engine
        eng = get_engine(get_database_url())
        try:
            with eng.begin() as conn:
                cols = ["signal_id", "user_id", "signal_type", "severity",
                        "payload", "idempotency_key"]
                placeholders = [":sid", ":uid", ":stype", ":sev",
                               "CAST(:pl AS JSONB)", ":ik"]
                params = {
                    "sid": sig.signal_id,
                    "uid": sig.user_id,
                    "stype": sig.signal_type.value,
                    "sev": sig.severity,
                    "pl": json.dumps(sig.payload),
                    "ik": sig.idempotency_key,
                }

                if regional_tables_ready(get_database_url()):
                    user_region = _resolve_region(sig.user_id, get_database_url())
                    if user_region:
                        cols.append("crdb_region")
                        placeholders.append(":region")
                        params["region"] = user_region

                sql = f"""
                    INSERT INTO spending_signals ({", ".join(cols)})
                    VALUES ({", ".join(placeholders)})
                    ON CONFLICT (idempotency_key) DO NOTHING
                    RETURNING signal_id
                """
                row = conn.execute(text(sql), params).fetchone()
            return row is not None
        except IntegrityError:
            # PK collision on signal_id (or any other unique constraint) —
            # the signal already exists, so this is a replay.
            return False

    def _process_in_background(handler, sig):
        try:
            handler.handle(sig)
        except Exception:
            coach_log.exception("background handler failed",
                                extra={"signal_id": sig.signal_id})

    @app.route("/api/cdc/signals", methods=["POST"])
    def cdc_signals_webhook():
        body = request.get_data() or b""
        sig_header = request.headers.get("X-Banko-Signature")
        if not _verify_signature(body, sig_header):
            return jsonify({"error": "invalid signature"}), 401

        try:
            envelope = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            return jsonify({"error": "malformed payload", "detail": str(e)}), 400

        try:
            signals = parse_changefeed_envelope(envelope)
        except SignalParseError as e:
            return jsonify({"error": "invalid signal", "detail": str(e)}), 400

        if not signals:
            return jsonify({"status": "no_op",
                            "reason": "envelope contained no inserts"}), 200

        handler = _get_coach_handler()
        queued_ids = []
        replayed_ids = []
        for sig in signals:
            try:
                claimed = _claim_signal(sig)
            except Exception:
                coach_log.exception("signal claim failed; queuing anyway",
                                    extra={"signal_id": sig.signal_id})
                claimed = True
            if not claimed:
                replayed_ids.append(sig.signal_id)
                continue
            app.socketio.start_background_task(
                _process_in_background, handler, sig)
            queued_ids.append(sig.signal_id)

        if not queued_ids and replayed_ids:
            payload = {"status": "replayed", "replayed": True,
                       "replayed_signal_ids": replayed_ids}
            if len(replayed_ids) == 1:
                payload["signal_id"] = replayed_ids[0]
            return jsonify(payload), 200

        return jsonify({"status": "queued",
                        "queued_signal_ids": queued_ids,
                        "replayed_signal_ids": replayed_ids}), 202

    # --- Coach v1 UI + REST ----------------------------------------------
    @app.route("/coach")
    def coach_page():
        from ..config.settings import get_config
        cfg = get_config()
        user_id = session.get("user_id") or cfg.coach_default_user_id
        return render_template("coach.html", user_id=user_id)

    @app.route("/api/coach/nudges", methods=["GET"])
    def coach_list_nudges():

        from ..config.settings import get_config
        cfg = get_config()
        user_id = (request.args.get("user_id") or session.get("user_id")
                   or cfg.coach_default_user_id)
        limit = min(int(request.args.get("limit", "20")), 100)
        db_url = get_database_url()
        from ..db.engine import get_engine
        eng = get_engine(db_url)
        with eng.connect() as conn:
            rows = conn.execute(text("""
                SELECT n.nudge_id, n.message, n.provider_used, n.created_at,
                       s.signal_type, s.severity
                FROM coach_nudges n
                LEFT JOIN spending_signals s ON s.signal_id = n.signal_id
                WHERE n.user_id = :u
                ORDER BY n.created_at DESC
                LIMIT :l
            """), {"u": user_id, "l": limit}).fetchall()
        return jsonify({"nudges": [{
            "nudge_id": str(r[0]), "message": r[1],
            "provider_used": display_label(config, r[2]),
            "created_at": r[3].isoformat() if r[3] else None,
            "signal_type": r[4], "severity": r[5],
        } for r in rows]})

    @app.route("/api/coach/nudges/<nudge_id>", methods=["GET"])
    def coach_get_nudge(nudge_id: str):
        from ..coach.tools import explain_nudge
        result = explain_nudge(nudge_id=nudge_id,
                               database_url=get_database_url())
        if not result:
            return jsonify({"error": "not found"}), 404
        result["provider_used"] = display_label(config, result.get("provider_used"))
        return jsonify(result)

    @app.route("/api/coach/chat", methods=["POST"])
    def coach_chat():
        from ..coach.agent import CoachAgent, default_llm_invoker
        from ..config.settings import get_config
        cfg = get_config()
        body = request.get_json(silent=True) or {}
        message = (body.get("message") or "").strip()
        if not message:
            return jsonify({"error": "message is required"}), 400
        user_id = (body.get("user_id") or session.get("user_id")
                   or cfg.coach_default_user_id)
        thread_id = body.get("thread_id")
        context = ({"nudge_id": body["nudge_id"]}
                   if body.get("nudge_id") else None)

        agent = CoachAgent(
            database_url=get_database_url(),
            llm_invoker=default_llm_invoker,
            provider_name=cfg.ai_service,
            max_steps=cfg.coach_agent_max_steps,
        )
        try:
            reply = agent.converse(user_id=user_id, message=message,
                                   history=[], context=context,
                                   thread_id=thread_id)
        except ModelWarmingUp:
            raise
        except Exception as e:
            # A provider timeout or DB blip must come back as a retryable
            # coach message, not a raw 500 the UI renders as an error.
            print(f"coach chat failed: {e}")
            return jsonify({
                "message": "I could not reach the model just now. "
                           "Give it a moment and send that again.",
                "transient": True,
            }), 503
        reply["provider_used"] = display_label(config, reply.get("provider_used"))
        return jsonify(reply)

    @socketio.on("coach.join")
    def coach_join(data):
        from flask_socketio import join_room
        user_id = data.get("user_id")
        if user_id:
            join_room(f"{get_config().coach_socketio_room_prefix}{user_id}")

    @app.route("/health/coach", methods=["GET"])
    def health_coach():

        from ..config.settings import get_config
        cfg = get_config()
        db_url = get_database_url()
        last_nudge_at = None
        db_ok = False
        if db_url:
            try:
                from ..db.engine import get_engine
                eng = get_engine(db_url)
                with eng.connect() as conn:
                    row = conn.execute(text(
                        "SELECT max(created_at) FROM coach_nudges"
                    )).fetchone()
                    last_nudge_at = (row[0].isoformat()
                                     if row and row[0] else None)
                    db_ok = True
            except Exception as e:
                coach_log.warning("health DB check failed: %s", e)

        components = {
            "db_reachable": db_ok,
            "webhook_secret_configured": bool(cfg.cdc_webhook_hmac_secret),
            "kafka_enabled": cfg.coach_kafka_enabled,
            "active_provider": _shown().provider,
            "actual": _actual().provider,
            "last_nudge_at": last_nudge_at,
        }
        overall = ("green" if db_ok and
                   components["webhook_secret_configured"]
                   else "degraded")
        return jsonify({"status": overall, "components": components}), 200

    # Data Generator Routes
    generation_state = {'running': False, 'should_stop': False}
    
    @app.route('/data-generator')
    def data_generator_page():
        """Render data generator page."""
        return render_template('data_generator.html')
    
    @app.route('/api/generate-data', methods=['POST'])
    def start_generation():
        """Start data generation with real-time updates."""
        if generation_state['running']:
            return jsonify({'error': 'Generation already running'}), 400
        
        data = request.json
        count = data.get('count', 1000)
        clear_existing = data.get('clear_existing', False)
        continuous = data.get('continuous', False)
        # Clearing wipes every user's expenses, personas included, so the
        # request has to say so in words, not just tick a box.
        if clear_existing and data.get('confirm_clear') != 'all users':
            return jsonify({'success': False,
                            'error': 'Clearing deletes every user\'s expenses. Confirm with "all users".'}), 400
        
        def generate_in_background():
            # CRITICAL: Need Flask app context for SocketIO in background thread
            # Use app.socketio to ensure we have the right instance
            with app.app_context():
                generation_state['running'] = True
                generation_state['should_stop'] = False
                generator = EnhancedExpenseGenerator(config.database_url)
                sock = app.socketio  # Get socketio instance from app
                
                try:
                    print("🚀 Starting generation...")
                    sock.emit('generation_progress', {
                        'current': 0, 
                        'total': count, 
                        'message': 'Starting generation...'
                    })
                    
                    if clear_existing:
                        print("🗑️  Clearing existing data...")
                        sock.emit('generation_progress', {
                            'current': 0, 
                            'total': count, 
                            'message': 'Clearing data...'
                        })
                        generator.clear_expenses()


                    # Use same batch size as generator for consistency
                    batch_size = get_config().data_gen_batch_size
                    import time
                    
                    # Main generation loop - supports continuous mode
                    while not generation_state['should_stop']:
                        total_generated = 0
                        start_time = time.time()
                        
                        # Generate one batch of records
                        while total_generated < count and not generation_state['should_stop']:
                            batch = min(batch_size, count - total_generated)
                            generated = generator.generate_and_save(count=batch, clear_existing=False)
                            total_generated += generated
                            
                            elapsed = time.time() - start_time
                            speed = total_generated / elapsed if elapsed > 0 else 0
                            
                            progress_data = {
                                'current': total_generated,
                                'total': count,
                                'speed': round(speed, 1),
                                'message': f'Generated {total_generated:,} / {count:,} ({speed:.0f} rec/sec)'
                            }
                            print(f"📊 Progress: {total_generated}/{count} ({speed:.0f} rec/sec)")
                            sock.emit('generation_progress', progress_data)
                        
                        print(f"✅ Generation complete: {total_generated} records")
                        sock.emit('generation_complete', {
                            'total_generated': total_generated,
                            'elapsed': round(time.time() - start_time, 1),
                            'continuous': continuous
                        })
                        
                        # If not continuous mode or stopped, exit
                        if not continuous or generation_state['should_stop']:
                            break
                        
                        # Brief pause before next cycle in continuous mode
                        time.sleep(1)
                        print("🔄 Continuous mode: Restarting generation...")
                        sock.emit('generation_progress', {
                            'current': 0,
                            'total': count,
                            'message': 'Continuous mode: Restarting...'
                        })
                except Exception as e:
                    print(f"❌ Error during generation: {e}")
                    import traceback
                    traceback.print_exc()
                    sock.emit('generation_error', {'message': str(e)})
                finally:
                    generation_state['running'] = False

        import threading
        thread = threading.Thread(target=generate_in_background)
        thread.daemon = True
        thread.start()
        
        return jsonify({'status': 'started'})
    
    @app.route('/api/stop-generation', methods=['POST'])
    def stop_generation():
        """Stop data generation."""
        generation_state['should_stop'] = True
        return jsonify({'status': 'stopping'})
    
    @app.route('/api/reset-generation', methods=['POST'])
    def reset_generation():
        """Reset generation state."""
        generation_state['running'] = False
        generation_state['should_stop'] = False
        return jsonify({'status': 'reset'})
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({'error': 'Internal server error'}), 500
    
    # Register agent dashboard blueprint
    from .agent_dashboard import agent_dashboard
    app.register_blueprint(agent_dashboard)
    from .api_v1 import api_v1
    app.register_blueprint(api_v1)
    from .assistant_api import assistant_api
    app.register_blueprint(assistant_api)

    # Last, so every closure the steps call already exists.
    # Lane A needs no database. Lane B runs the gate (schema, personas)
    # first; locality conversions are online in CockroachDB, so they come
    # after the app is already usable.
    # The embedding model goes first: it needs nothing but the local disk,
    # while the provider check can wait on the network.
    start_warmup(readiness, [
        ("embedding_model", _model_step),
        ("ai_provider", _provider_step),
    ], stop=warmup_stop, name="banko-warmup-local")
    start_warmup(readiness, [
        ("schema", _schema_step),
        ("personas", _personas_step),
        ("demo_users", _expiry_step),
        ("coach", _coach_step),
        ("locality", lambda: f"{len(run_pending_conversions(config.database_url))} conversions"),
        ("sample_data", lambda: auto_setup_data_if_needed(config.database_url)),
    ], stop=warmup_stop, name="banko-warmup-db")
    # Lane C: persona history. Background, so the ready line never waits
    # for it; progress shows on /api/health and the "seeding" socket event.
    readiness.mark_background("persona_data")
    readiness.mark_background("integrity")
    readiness.mark_background("rls")
    start_warmup(readiness, [("persona_data", _persona_data_step)], stop=warmup_stop, name="banko-warmup-seed")
    # Lane D: the assistant's checkpointer, also background.
    readiness.mark_background("assistant")
    start_warmup(readiness, [("assistant", _assistant_step)], stop=warmup_stop, name="banko-warmup-assistant")
    return app
