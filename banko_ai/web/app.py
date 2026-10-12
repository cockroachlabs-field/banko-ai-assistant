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
    after_this_request,
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
    get_stored_trace,
    insert_step,
    reset_trace,
    serving_regions,
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


def auto_setup_data_if_needed(database_url: str) -> str:
    """Generate demo data when the database has almost none. Runs in the
    warm-up thread, never before the server listens."""
    if get_config().banko_skip_autosetup:
        print("Skipping data generation (--no-data or BANKO_SKIP_AUTOSETUP)", flush=True)
        return "skipped (BANKO_SKIP_AUTOSETUP)"
    with get_engine(database_url).connect() as conn:
        rows = conn.execute(text("SELECT count(*) FROM (SELECT 1 FROM expenses LIMIT 100)")).scalar()
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
            readiness.set("rls", "ready", status.reason)
        except Exception as e:  # the schema is fine; RLS just stays off
            boot_log.warning("rls: status check failed (%s); row-level security stays off", e)
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

    def _personas_step() -> str:
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
        try:
            if config.banko_skip_autosetup:
                return "skipped (BANKO_SKIP_AUTOSETUP)"
            readiness.wait_for("personas", timeout=600)
            from ..services.seeding import seed_personas
            written = seed_personas(config.database_url, on_progress=_persona_progress)
            return ", ".join(f"{name} +{n}" for name, n in written.items())
        finally:
            IntegrityWorker(url=config.database_url, stop=warmup_stop, on_report=_integrity).start()

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

    def _serving_regions(explain_text: str | None) -> str | None:
        """The region(s) that actually served a query, parsed from its
        EXPLAIN ANALYZE output. This is what makes the badge move during a
        region outage: the user's home region is where their rows live,
        but leaseholders fail over, and the plan names the survivors."""
        return serving_regions(explain_text)

    def _render_aggregation_markdown(agg) -> str:
        """Deterministic answer card for aggregation questions. Numbers come
        straight from SQL; nothing here passes through a model."""
        window = f"{agg.window_start} to {agg.window_end}"
        scope = agg.category or "all categories"
        if agg.operation == "count":
            headline = (f"You made **{agg.count} transactions** on {scope} "
                        f"between {window}.")
        elif agg.operation == "average":
            headline = (f"Your average {scope} transaction was "
                        f"**${agg.average:,.2f}** between {window} "
                        f"({agg.count} transactions, ${agg.total:,.2f} total).")
        else:
            headline = (f"You spent **${agg.total:,.2f}** on {scope} "
                        f"between {window} ({agg.count} transactions).")
        if not agg.count:
            return (f"No matching transactions on {scope} between {window}. "
                    f"Try a wider window or a different category.")
        lines = [headline, "", "| Date | Merchant | Amount |", "|---|---|---|"]
        for row in agg.rows:
            lines.append(f"| {row['date']} | {row['merchant']} "
                         f"| ${row['amount']:,.2f} |")
        return "\n".join(lines)

    def _aggregation_insights(agg, question: str) -> str | None:
        """One short LLM pass over the SQL result. The figures arrive
        pre-computed and the prompt forbids new arithmetic, so the model
        adds judgment, not math. Any failure just means no insights block."""
        try:
            from banko_ai.agents.llm_factory import get_llm_for_agent
            merchants = {}
            for row in agg.rows:
                merchants[row["merchant"]] = (
                    merchants.get(row["merchant"], 0) + row["amount"])
            top = sorted(merchants.items(), key=lambda kv: -kv[1])[:3]
            top_text = ", ".join(f"{m} (${v:,.2f})" for m, v in top)
            prompt = (
                "You are a personal finance coach. Exact figures, already "
                f"computed from the user's expense database: total "
                f"${agg.total:,.2f} across {agg.count} transactions on "
                f"{agg.category or 'all categories'} between "
                f"{agg.window_start} and {agg.window_end}. "
                f"Top merchants: {top_text}. "
                f"The user asked: \"{question}\". Write two or three short, "
                "specific observations or suggestions as markdown bullets. "
                "Use only the figures given; do not compute new totals or "
                "restate the question. No headings, bullets only.")
            llm = get_llm_for_agent(temperature=0.4)
            raw = llm.invoke(prompt)
            text = raw.content if hasattr(raw, "content") else str(raw)
            text = text.strip()
            return text or None
        except Exception as e:
            print(f"aggregation insights skipped: {e}")
            return None

    # Narrative cache for the aggregation branch: SQL reruns every time
    # (exact figures), only the LLM prose is reused. Keyed on user,
    # provider, language, and the figures themselves, so any change in
    # the data invalidates naturally.
    _agg_narrative_cache: dict = {}

    def _translate_answer(markdown_text: str, target_language: str) -> str | None:
        """Render a deterministic answer in the user's language. The
        figures were computed by SQL and must survive verbatim, so the
        model translates prose only; any failure keeps the English text."""
        try:
            from banko_ai.agents.llm_factory import get_llm_for_agent
            llm = get_llm_for_agent(temperature=0.1)
            raw = llm.invoke(
                f"Translate the following financial summary into "
                f"{target_language}. Keep every number, currency amount, "
                f"percentage, date, and merchant name exactly as written. "
                f"Keep the markdown formatting. Output only the "
                f"translation.\n\n{markdown_text}")
            text = raw.content if hasattr(raw, "content") else str(raw)
            return text.strip() or None
        except Exception as e:
            print(f"answer translation skipped: {e}")
            return None
    
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
                from ..utils.migration import regional_tables_ready
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
        """API endpoint for RAG responses."""
        try:
            if not ai_provider:
                return jsonify({
                    'success': False,
                    'error': 'AI provider not available'
                }), 500
            
            data = request.get_json()
            query = data.get('query', '')
            language = data.get('language', 'en')

            from banko_ai.utils.intent_classifier import REDIRECT_MESSAGE, is_financial_query
            if not is_financial_query(query):
                return jsonify({
                    'success': True,
                    'response': REDIRECT_MESSAGE,
                    'sources': [],
                    'metadata': {'cached': False, 'intent': 'off-topic'}
                })

            # Use original simple logic - no user filtering
            search_results = search_engine.search_expenses(
                query=query,
                user_id=current_demo_user(),
                limit=5,
                threshold=0.7
            )
            
            # Generate RAG response - use original simple logic
            rag_response = ai_provider.generate_rag_response(
                query=query,
                context=search_results,
                user_id=current_demo_user(),
                language=language
            )

            metadata = dict(rag_response.metadata) if rag_response.metadata else {}
            if config.ai_display_provider:
                shown = _shown()
                metadata['provider'] = shown.provider
                metadata['model'] = shown.model

            return jsonify({
                'success': True,
                'response': rag_response.response,
                'sources': [
                    {
                        'expense_id': result.expense_id,
                        'description': result.description,
                        'merchant': result.merchant,
                        'amount': result.amount,
                        'similarity_score': result.similarity_score
                    }
                    for result in rag_response.sources
                ],
                'metadata': metadata
            })

        except ModelWarmingUp:
            raise
        except Exception as e:
            print(f"RAG error: {e}")
            return jsonify({
                'success': False,
                'error': neutral_error(e)
            }), 500
    
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
        """Handle receipt upload and process with Agent system"""
        from pathlib import Path
        
        try:
            # Check if file was uploaded
            if 'receipt' not in request.files:
                return jsonify({
                    'success': False,
                    'error': 'No receipt file provided'
                }), 400
            
            file = request.files['receipt']
            
            if file.filename == '':
                return jsonify({
                    'success': False,
                    'error': 'No file selected'
                }), 400

            # Save file temporarily
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".pdf"}:
                return jsonify({'success': False, 'error': 'Upload a receipt image or PDF'}), 400
            handle = tempfile.NamedTemporaryFile(prefix="banko_receipt_", suffix=file_ext, delete=False)
            temp_path = handle.name
            handle.close()

            @after_this_request
            def _remove_temp_file(response):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
                return response

            file.save(temp_path)

            print(f"📄 Receipt uploaded: {file.filename} → {temp_path}")
            
            # One real region label for every agent in this pipeline: the
            # cluster's gateway region when multi-region, 'local' otherwise.
            # Hardcoded AWS names here used to reach the dashboard no
            # matter what cluster was underneath.
            from banko_ai.utils.migration import agent_home_region
            agent_region = agent_home_region(config.database_url)

            # Initialize Receipt Agent
            try:
                from pydantic import ValidationError

                from banko_ai.agents.llm_factory import get_embedding_model, get_llm_for_agent
                from banko_ai.agents.receipt_agent import ReceiptAgent
                from banko_ai.agents.receipt_extraction_schema import (
                    ReceiptExtraction,
                    is_placeholder_payload,
                )
                
                # Use centralized LLM factory with the currently selected model
                current_model = getattr(ai_provider, 'current_model', None)
                llm = get_llm_for_agent(temperature=0.7, model_override=current_model)
                embedding_model = get_embedding_model()
                
                receipt_agent = ReceiptAgent(
                    region=agent_region,
                    llm=llm,
                    database_url=config.database_url,
                    embedding_model=embedding_model
                )
                
                print(f"🤖 Receipt Agent created: {receipt_agent.agent_id[:8]}...")
                
                # Process document
                result = receipt_agent.process_document(
                    file_path=temp_path,
                    user_id=session.get('user_id', 'demo_user'),
                    document_type='receipt'
                )
                
                print(f"✅ Processing result: {result.get('success', False)}")
                
                # Check if processing actually succeeded
                if not result.get('success', False):
                    # Processing failed - return error with details
                    errors = result.get('errors', ['Unknown processing error'])
                    print(f"❌ Receipt processing failed: {errors}")
                    return jsonify({
                        'success': False,
                        'error': f"Receipt processing failed: {', '.join(errors)}",
                        'details': result
                    }), 500
                
                # Extract data for response (Receipt Agent returns 'extracted_fields')
                extracted = result.get('extracted_fields', {})

                print(f"📊 Extracted fields: {extracted}")

                # Some models (notably code-tuned ones like ibm/granite-8b-code-instruct)
                # echo the prompt scaffold instead of extracting values, which would land
                # in the DB as opaque driver errors.
                if is_placeholder_payload(extracted):
                    print(f"❌ Extraction returned prompt-template placeholders: {extracted}")
                    return jsonify({
                        'success': False,
                        'error': (
                            "The selected model returned the prompt template instead of "
                            "real receipt values. Switch to a chat/instruct model (not a "
                            "code model) and re-upload."
                        ),
                        'extracted_fields': extracted,
                    }), 422

                try:
                    validated = ReceiptExtraction(**extracted)
                except ValidationError as ve:
                    print(f"❌ Extraction failed schema validation: {ve}")
                    return jsonify({
                        'success': False,
                        'error': 'Extracted receipt fields failed validation.',
                        'validation_errors': ve.errors(),
                        'extracted_fields': extracted,
                    }), 422

                # Downstream code reads from `extracted`; replace with the validated
                # payload so the INSERT sees clean types and non-empty strings.
                extracted = validated.model_dump()

                # Emit real-time update: Receipt Agent completed
                try:
                    socketio.emit('agent_activity', {
                        'agent_type': 'receipt',
                        'region': agent_region,
                        'status': 'processing',
                        'message': f"Processing receipt from {extracted.get('merchant', 'Unknown')}",
                        'timestamp': datetime.now().isoformat()
                    })
                except Exception:
                    pass
                
                # Step 1: Add expense to expenses table
                expense_id = None
                try:
                    # Use official sqlalchemy-cockroachdb dialect (no conversion needed!)
                    from ..db.engine import get_engine
                    engine = get_engine(config.database_url)

                    expense_id = str(uuid.uuid4())
                    
                    # Get or create proper UUID for user
                    session_user_id = session.get('user_id', 'demo_user')
                    if isinstance(session_user_id, str) and not session_user_id.count('-') == 4:
                        # Not a UUID, create a deterministic one for demo_user
                        import hashlib
                        user_uuid = uuid.UUID(hashlib.md5(session_user_id.encode()).hexdigest())
                        user_id = str(user_uuid)
                    else:
                        user_id = session_user_id
                    
                    # Handle None values from extraction
                    amount = extracted.get('amount')
                    if amount is None or amount == 'None':
                        amount = 0.0
                    else:
                        amount = float(amount)
                    
                    # Handle missing date - use today as default
                    expense_date = extracted.get('date')
                    if expense_date is None or expense_date == 'None' or expense_date == '':
                        expense_date = datetime.now().strftime('%Y-%m-%d')
                    
                    # Handle missing merchant
                    merchant = extracted.get('merchant', 'Unknown')
                    if not merchant or merchant == 'None':
                        merchant = 'Unknown'
                    
                    # Generate embedding for expense using natural language
                    # This helps match conversational queries like "when did I go to X?"
                    merchant = extracted.get('merchant') or 'Unknown'
                    category = extracted.get('category') or 'Other'
                    expense_text = f"Spent ${amount} at {merchant} for {category} on {expense_date.strftime('%Y-%m-%d') if hasattr(expense_date, 'strftime') else expense_date}"
                    
                    from banko_ai.utils.embeddings import load_embedding_model
                    embedding_model = load_embedding_model()
                    embedding = embedding_model.encode(expense_text).tolist()
                    
                    # Get category and items for tags and description
                    category = extracted.get('category') or 'Other'
                    items = extracted.get('items', [])
                    
                    # Generate tags from merchant and category
                    tags = []
                    if merchant and merchant != 'Unknown':
                        # Add first word of merchant name (lowercase)
                        tags.append(merchant.lower().split()[0])
                    if category and category != 'general':
                        tags.append(category.lower())
                    
                    # Format better description
                    if items and len(items) > 0:
                        item_list = ', '.join(items)
                        description = f"Spent ${amount:.2f} at {merchant} for {item_list}."
                    else:
                        description = f"Spent ${amount:.2f} on {category} at {merchant}"
                    
                    with engine.connect() as conn:
                        # Format embedding as array literal for CockroachDB
                        embedding_str = '[' + ','.join(str(x) for x in embedding) + ']'

                        # Format tags array for CockroachDB
                        tags_str = '{' + ','.join(f'"{tag}"' for tag in tags) + '}' if tags else None

                        # Pin the row to the uploader's home region like every
                        # other expenses writer, or the region-pruned reads
                        # silently exclude uploaded receipts.
                        from banko_ai.utils.migration import regional_tables_ready

                        from .auth import resolve_user_region as _resolve_region
                        receipt_region = (_resolve_region(user_id, config.database_url)
                                          if regional_tables_ready(config.database_url) else None)
                        cols = """expense_id, user_id, expense_amount, shopping_type,
                                merchant, expense_date, description, payment_method,
                                tags, embedding"""
                        vals = """:expense_id, :user_id, :amount, :category,
                                :merchant, :date, :description, :payment_method,
                                :tags, CAST(:embedding AS VECTOR(384))"""
                        insert_params = {
                            'expense_id': expense_id,
                            'user_id': user_id,
                            'amount': amount,
                            'category': category,
                            'merchant': merchant,
                            'date': expense_date,
                            'description': description,
                            'payment_method': extracted.get('payment_method', 'unknown') if extracted.get('payment_method') else 'unknown',
                            'tags': tags_str,
                            'embedding': embedding_str
                        }
                        if receipt_region:
                            cols += ", crdb_region"
                            vals += ", :crdb_region"
                            insert_params['crdb_region'] = receipt_region
                        conn.execute(text(f"""
                            INSERT INTO expenses ({cols}) VALUES ({vals})
                        """), insert_params)
                        conn.commit()
                    
                    print(f"💰 Expense added to expenses table: {expense_id}")
                    print(f"   📝 Description: {description}")
                    print(f"   🏷️  Tags: {tags}")
                    
                    # Also index into langchain-cockroachdb vectorstore
                    try:
                        from banko_ai.vector_search.crdb_vectorstore import index_expense_document
                        index_expense_document(
                            expense_id=expense_id,
                            description=description,
                            metadata={
                                "user_id": user_id,
                                "merchant": merchant,
                                "shopping_type": category,
                                "expense_amount": amount,
                                "expense_date": str(expense_date),
                                "payment_method": extracted.get('payment_method', 'unknown'),
                            },
                        )
                        print("   🔍 Indexed in CockroachDB vectorstore")
                    except Exception as vs_err:
                        print(f"   ⚠️  Vectorstore indexing skipped: {vs_err}")
                    
                    # Emit update: Expense added
                    try:
                        socketio.emit('agent_activity', {
                            'agent_type': 'receipt',
                            'region': agent_region,
                            'status': 'completed',
                            'message': f"Added expense: {extracted.get('merchant', 'Unknown')} - ${amount}",
                            'timestamp': datetime.now().isoformat()
                        })
                    except Exception:
                        pass
                    
                except Exception as e:
                    print(f"⚠️  Failed to add expense to table: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Step 2: Trigger Fraud Agent
                fraud_result = "✅ No issues detected"
                try:
                    from banko_ai.agents.fraud_agent import FraudAgent
                    from banko_ai.agents.llm_factory import get_embedding_model, get_llm_for_agent
                    
                    # Use centralized LLM factory based on configured provider
                    fraud_llm = get_llm_for_agent(temperature=0.7)
                    fraud_embedding_model = get_embedding_model()
                    
                    fraud_agent = FraudAgent(
                        region=agent_region,
                        llm=fraud_llm,
                        database_url=config.database_url,
                        embedding_model=fraud_embedding_model,
                        fraud_threshold=0.7,
                        duplicate_window_days=config.fraud_duplicate_window_days
                    )
                    
                    print("🕵️  Running fraud check...")
                    
                    # Emit update: Fraud Agent started
                    try:
                        socketio.emit('agent_activity', {
                            'agent_type': 'fraud',
                            'region': agent_region,
                            'status': 'processing',
                            'message': 'Scanning for suspicious patterns...',
                            'timestamp': datetime.now().isoformat()
                        })
                    except Exception:
                        pass
                    
                    # Analyze the newly created expense for fraud
                    fraud_check = fraud_agent.analyze_expense(expense_id)
                    
                    if fraud_check.get('fraud_detected', False):
                        confidence = fraud_check.get('confidence', 0)
                        signals = fraud_check.get('signals', [])
                        dup_signals = [s for s in signals if s.get('type') == 'duplicate']
                        if dup_signals:
                            fraud_result = f"⚠️  Duplicate detected: {dup_signals[0]['details']}"
                        else:
                            fraud_result = f"⚠️  Suspicious transaction detected (confidence: {confidence:.0%})"
                    else:
                        fraud_result = "✅ No issues detected"
                    
                    print(f"   {fraud_result}")
                    
                    # Emit update: Fraud Agent completed
                    try:
                        socketio.emit('agent_activity', {
                            'agent_type': 'fraud',
                            'region': agent_region,
                            'status': 'completed',
                            'message': fraud_result,
                            'timestamp': datetime.now().isoformat()
                        })
                    except Exception:
                        pass
                    
                except Exception as e:
                    print(f"⚠️  Fraud check failed: {e}")
                    fraud_result = "⚠️  Check failed"
                
                # Step 3: Trigger Budget Agent
                budget_result = "Budget updated"
                try:
                    from banko_ai.agents.budget_agent import BudgetAgent
                    from banko_ai.agents.llm_factory import get_llm_for_agent
                    
                    # Use centralized LLM factory based on configured provider
                    budget_llm = get_llm_for_agent(temperature=0.7)
                    
                    budget_agent = BudgetAgent(
                        region=agent_region,
                        llm=budget_llm,
                        database_url=config.database_url,
                        alert_threshold=0.8
                    )
                    
                    print("📊 Running budget check...")
                    
                    # Emit update: Budget Agent started
                    try:
                        socketio.emit('agent_activity', {
                            'agent_type': 'budget',
                            'region': agent_region,
                            'status': 'processing',
                            'message': 'Analyzing budget impact...',
                            'timestamp': datetime.now().isoformat()
                        })
                    except Exception:
                        pass
                    
                    # Get budget from config (can be set via MONTHLY_BUDGET_DEFAULT env var)
                    from ..config.settings import get_config
                    app_config = get_config()
                    
                    budget_check = budget_agent.check_budget_status(
                        user_id=user_id,
                        monthly_budget=app_config.monthly_budget_default
                    )
                    
                    status = budget_check.get('status', 'unknown')
                    if status == 'over_budget':
                        budget_result = "⚠️  Over budget!"
                    elif status == 'on_pace_to_exceed':
                        budget_result = "⚠️  On pace to exceed"
                    else:
                        budget_result = "✅ Within budget"
                    
                    print(f"   {budget_result}")
                    
                    # Emit update: Budget Agent completed
                    try:
                        socketio.emit('agent_activity', {
                            'agent_type': 'budget',
                            'region': agent_region,
                            'status': 'completed',
                            'message': budget_result,
                            'timestamp': datetime.now().isoformat()
                        })
                    except Exception:
                        pass
                    
                except Exception as e:
                    print(f"⚠️  Budget check failed: {e}")
                    budget_result = "⚠️  Check failed"
                
                return jsonify({
                    'success': True,
                    'merchant': extracted.get('merchant', 'Unknown'),
                    'amount': str(extracted.get('amount', '0.00')),
                    'category': extracted.get('category', 'Unknown'),
                    'date': extracted.get('date', 'Unknown'),
                    'items': extracted.get('items', []),
                    'expense_id': expense_id,
                    'fraud_status': fraud_result,
                    'budget_impact': budget_result,
                    'document_id': result.get('document_id'),
                    'message': 'Receipt processed by Receipt, Fraud, and Budget agents'
                })
                
            except Exception as agent_error:
                print(f"⚠️  Agent processing error: {agent_error}")
                import traceback
                traceback.print_exc()

                # Return error (not fake success)
                return jsonify({
                    'success': False,
                    'error': neutral_error(agent_error),
                    'message': 'Receipt uploaded but processing failed. Check server logs.'
                }), 500
        
        except Exception as e:
            from werkzeug.exceptions import HTTPException
            if isinstance(e, HTTPException):
                raise
            print(f"❌ Receipt upload error: {e}")
            return jsonify({
                'success': False,
                'error': NEUTRAL_FAILED
            }), 500
    
    def _trace_identity() -> str:
        """The model as the persona shows it, for LLM steps in the trace."""
        shown = _shown()
        return f"{shown.name} · {shown.model}"

    def _finish_trace() -> str:
        """Close this request's trace and park it server side. The chat
        history lives in the signed session cookie, so the message keeps
        only the short id."""
        trace_step("total", "Total request time", ms=trace_elapsed_ms())
        return store_trace(current_trace())

    def _render_chat(ai_provider_display):
        """Render the chat page with each message's stored trace attached
        for the template; the session itself keeps only the ids."""
        view = []
        for message in session.get('chat', []):
            meta = message.get('meta')
            if meta and meta.get('trace_id'):
                message = dict(message)
                message['meta'] = dict(meta, trace=get_stored_trace(meta['trace_id']) or [])
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

    def _chat_impl():
        # Clear chat history on GET request (fresh start)
        if request.method == 'GET':
            session['chat'] = []
        elif 'chat' not in session:
            session['chat'] = []

        # Get AI provider info for display
        ai_provider_display = _provider_ui()
        
        if request.method == 'POST':
            # One import for every branch below: a local import inside a
            # single branch makes the name function-local everywhere and
            # the other branches crash on it. Gated on the deployment being
            # regional so a failed RBR migration (tables without
            # crdb_region) can never poison the read path.
            from ..utils.migration import regional_tables_ready
            from .auth import resolve_user_region as _resolve_region_raw

            def resolve_user_region(user_id, database_url):
                if not regional_tables_ready(database_url):
                    return None
                return _resolve_region_raw(user_id, database_url)

            # Handle both 'message' and 'user_input' field names for compatibility
            user_message = request.form.get('user_input') or request.form.get('message')
            response_language = request.form.get('response_language', 'en-US')
            
            if user_message:
                session['chat'].append({'text': user_message, 'class': 'User'})
                prompt = user_message
                
                # Persist user message in CockroachDB chat history
                try:
                    from langchain_core.messages import HumanMessage

                    from banko_ai.utils.crdb_chat_history import get_chat_history
                    chat_session_id = session.get('session_id', session.sid if hasattr(session, 'sid') else 'default')
                    crdb_history = get_chat_history(chat_session_id, database_url=config.database_url)
                    crdb_history.add_message(HumanMessage(content=user_message))
                except Exception:
                    pass
                
                # Map language codes to language names for AI prompt
                language_map = {
                    'en-US': 'English',
                    'es-ES': 'Spanish', 
                    'fr-FR': 'French',
                    'de-DE': 'German',
                    'it-IT': 'Italian',
                    'pt-PT': 'Portuguese',
                    'ja-JP': 'Japanese',
                    'ko-KR': 'Korean',
                    'zh-CN': 'Chinese',
                    'hi-IN': 'Hindi'
                }
                
                target_language = language_map.get(response_language, 'English')

                from banko_ai.utils.intent_classifier import REDIRECT_MESSAGE, is_financial_query
                t_route = time.perf_counter()
                if not is_financial_query(user_message):
                    trace_step("route", "Intent routing",
                               "Not a finance question: redirected, no database work",
                               ms=(time.perf_counter() - t_route) * 1000)
                    session['chat'].append({'text': REDIRECT_MESSAGE, 'class': 'Assistant',
                                            'meta': {'trace_id': _finish_trace()}})
                    return _render_chat(ai_provider_display)

                # Agentic routing: aggregation questions go to SQL so the
                # number is exact and identical on every provider. The LLM
                # never does the arithmetic.
                from banko_ai.utils.intent_classifier import classify_aggregation, extract_window
                agg_intent = classify_aggregation(user_message)
                route_ms = (time.perf_counter() - t_route) * 1000
                if agg_intent is not None:
                    trace_step("route", "Intent routing",
                               f"Aggregation, answered by SQL: {agg_intent.operation} on "
                               f"'{agg_intent.subject or 'all spending'}', "
                               f"{agg_intent.window_start} to {agg_intent.window_end}",
                               ms=route_ms)
                    # Same transient posture as the RAG branch below: a DB
                    # blip during a fault demo must read as reconnecting,
                    # never as a 500.
                    try:
                        from banko_ai.utils.aggregations import explain_aggregation, run_aggregation
                        print("\n🧮 AGGREGATION (agentic routing):")
                        print(f"1. Intent: {agg_intent.operation} on "
                              f"'{agg_intent.subject or 'all spending'}' | window "
                              f"{agg_intent.window_start} to {agg_intent.window_end}")
                        user_region = resolve_user_region(current_demo_user(), config.database_url)
                        if user_region:
                            print(f"   Region-pruned to {user_region} (REGIONAL BY ROW)")
                            trace_step("region", "Region pruning",
                                       f"Scoped to {user_region} (REGIONAL BY ROW)")
                        t0 = time.perf_counter()
                        agg = run_aggregation(agg_intent, current_demo_user(),
                                              config.database_url, region=user_region)
                        db_ms = int((time.perf_counter() - t0) * 1000)
                        print(f"2. SQL answered in {db_ms}ms: ${agg.total:,.2f} "
                              f"across {agg.count} transactions "
                              f"(category: {agg.category or 'all'})")
                        print("   Figures computed by CockroachDB; the model "
                              "never does the arithmetic")
                        trace_step("sql", "Aggregation SQL",
                                   f"${agg.total:,.2f} across {agg.count} transactions "
                                   f"(category: {agg.category or 'all'}), computed by CockroachDB",
                                   ms=db_ms, status="success")
                        # The SQL always runs (exact, fast); only the model
                        # work is cached. Same user, same figures, same
                        # language: reuse the narrative instead of paying
                        # for insights plus translation again, which on a
                        # local model turns a repeat question from seconds
                        # into milliseconds.
                        cache_key = (current_demo_user(), config.ai_service,
                                     target_language, agg.operation,
                                     agg.category, str(agg.window_start),
                                     str(agg.window_end), agg.total, agg.count)
                        cached_text = _agg_narrative_cache.get(cache_key)
                        if cached_text:
                            est = int(len(cached_text.split()) * 1.3)
                            print(f"3. ✅ Narrative cache HIT "
                                  f"(est. {est} tokens saved)")
                            trace_step("cache", "Narrative cache",
                                       f"Hit: reused the narrative, est. {est} tokens saved",
                                       status="hit")
                            agg_text = cached_text
                        else:
                            print("3. ❌ Narrative cache MISS, generating")
                            trace_step("cache", "Narrative cache", "Miss: generating the narrative",
                                       status="miss")
                            agg_text = _render_aggregation_markdown(agg)
                            if agg.count:
                                t_llm = time.perf_counter()
                                insights = _aggregation_insights(agg, user_message)
                                llm_ms = (time.perf_counter() - t_llm) * 1000
                                if insights:
                                    agg_text += "\n\n**Insights**\n\n" + insights
                                    print(f"   Insights narrated by {config.ai_service} "
                                          f"around the pinned figures")
                                    trace_step("llm", "LLM insights",
                                               f"{_trace_identity()}, around the pinned figures",
                                               ms=llm_ms, status="success")
                                else:
                                    trace_step("llm", "LLM insights",
                                               f"{_trace_identity()}, no insights returned",
                                               ms=llm_ms, status="warning")
                            # SQL computed the figures; the model only
                            # renders them in the user's language. English
                            # answers on a Hindi session were this path
                            # skipping the language setting the RAG path
                            # always honored.
                            if target_language != 'English':
                                t_llm = time.perf_counter()
                                translated = _translate_answer(agg_text, target_language)
                                llm_ms = (time.perf_counter() - t_llm) * 1000
                                agg_text = translated or agg_text
                                print(f"   Rendered in {target_language}, figures preserved")
                                trace_step("llm", "LLM translation",
                                           f"{_trace_identity()}, into {target_language}"
                                           if translated else
                                           f"{_trace_identity()}, translation skipped, kept English",
                                           ms=llm_ms, status="success" if translated else "warning")
                            if len(_agg_narrative_cache) > 256:
                                _agg_narrative_cache.clear()
                            _agg_narrative_cache[cache_key] = agg_text
                        t_explain = time.perf_counter()
                        explain_text = explain_aggregation(agg_intent, current_demo_user(),
                                                           config.database_url, region=user_region)
                        served = _serving_regions(explain_text) or user_region
                        trace_step("explain", "EXPLAIN ANALYZE",
                                   f"Plan captured{f', served by {served}' if served else ''}",
                                   ms=(time.perf_counter() - t_explain) * 1000)
                        session['chat'].append({
                            'text': agg_text,
                            'class': 'Assistant',
                            'meta': {
                                'region': served,
                                'db_ms': db_ms,
                                'explain': explain_text,
                                'trace_id': _finish_trace(),
                            }
                        })
                        try:
                            from langchain_core.messages import AIMessage
                            crdb_history.add_message(AIMessage(content=agg_text))
                        except Exception:
                            pass
                    except Exception as e:
                        if isinstance(e, TRANSIENT_ERRORS):
                            error_message = "Reconnecting to the database, try that again in a moment."
                            trace_step("error", "Answer stopped",
                                       "The database connection dropped; nothing was answered",
                                       status="error")
                        else:
                            print(f"Aggregation error: {str(e)}")
                            error_message = NEUTRAL_FAILED
                            trace_step("error", "Answer stopped",
                                       "The aggregation failed here; details are in the server log",
                                       status="error")
                        session['chat'].append({'text': error_message, 'class': 'Assistant',
                                                'meta': {'trace_id': _finish_trace()}})
                    return _render_chat(ai_provider_display)

                try:
                    # Retrieval questions: vector RAG scoped to the session
                    # persona, with a date predicate when the question names
                    # a window. The cache layers live inside the engine.
                    window = extract_window(user_message)
                    trace_step("route", "Intent routing",
                               "Retrieval, answered by vector search (RAG)"
                               + (f", {window[0]} to {window[1]}" if window else ""),
                               ms=route_ms)
                    t0 = time.perf_counter()
                    search_result = search_engine.search_expenses(
                        query=prompt,
                        user_id=current_demo_user(),
                        limit=10,
                        date_start=window[0] if window else None,
                        date_end=window[1] if window else None,
                        region=resolve_user_region(current_demo_user(), config.database_url),
                        capture_explain=True,
                    )
                    db_ms = int((time.perf_counter() - t0) * 1000)

                    if isinstance(search_result, tuple):
                        search_results, explain_text = search_result
                    else:
                        search_results = search_result
                        explain_text = ""

                    print(f"Using {config.ai_service} for response generation in {target_language}")

                    # Convert SearchResult objects to dictionaries if needed
                    if search_results and hasattr(search_results[0], 'description'):
                        # Convert SearchResult objects to dict format - MUST include ALL fields!
                        search_results_dict = []
                        for result in search_results:
                            search_results_dict.append({
                                'expense_id': result.expense_id,
                                'user_id': result.user_id,
                                'description': result.description,
                                'merchant': result.merchant,
                                'expense_amount': result.amount,
                                'expense_date': result.date,  # ← WAS MISSING!
                                'shopping_type': result.metadata.get('shopping_type', 'Unknown'),
                                'payment_method': result.metadata.get('payment_method', 'Unknown'),  # ← WAS MISSING!
                                'similarity_score': result.similarity_score
                            })
                        search_results = search_results_dict

                    # Generate RAG response with language preference
                    steps_before = len(current_trace())
                    t_llm = time.perf_counter()
                    if hasattr(ai_provider, 'simple_rag_response'):
                        rag_response_text = ai_provider.simple_rag_response(
                            user_message, search_results, language=target_language
                        )
                    else:
                        rag_response = ai_provider.generate_rag_response(
                            user_message, search_results, None, response_language
                        )
                        rag_response_text = rag_response.response if hasattr(rag_response, 'response') else str(rag_response)

                    llm_ms = (time.perf_counter() - t_llm) * 1000
                    print(f"Response from {config.ai_service}: {rag_response_text}")
                    # A response cache hit inside the provider means no
                    # model call happened, so there is no LLM step to show.
                    inner = current_trace()[steps_before:]
                    served_from_cache = any(
                        s["label"] == "Response cache" and s["status"] == "hit" for s in inner)
                    if not served_from_cache:
                        # The provider's cache lookup and write are traced
                        # on their own; the model time is what is left, and
                        # it ran right after the cache miss.
                        gen_ms = max(llm_ms - sum(s["ms"] or 0 for s in inner), 0.0)
                        misses = [i for i, s in enumerate(inner) if s["label"] == "Response cache"]
                        detail = (f"{_trace_identity()}, grounded on {len(search_results or [])} expenses"
                                  + (f", in {target_language}" if target_language != 'English' else ""))
                        if misses:
                            at = misses[-1]
                            insert_step(steps_before + at + 1, "llm", "LLM answer", detail,
                                        ms=gen_ms, status="success", t=inner[at]["t"] + gen_ms)
                        else:
                            trace_step("llm", "LLM answer", detail, ms=llm_ms, status="success")

                    session['chat'].append({
                        'text': rag_response_text,
                        'class': 'Assistant',
                        'meta': {
                            'region': _serving_regions(explain_text)
                                or resolve_user_region(current_demo_user(), config.database_url),
                            'db_ms': db_ms,
                            'explain': explain_text,
                            'trace_id': _finish_trace(),
                        }
                    })

                    # Persist assistant response in CockroachDB chat history
                    try:
                        from langchain_core.messages import AIMessage
                        crdb_history.add_message(AIMessage(content=rag_response_text))
                    except Exception:
                        pass
                    
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

        from banko_ai.utils.migration import regional_tables_ready
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
    return app
