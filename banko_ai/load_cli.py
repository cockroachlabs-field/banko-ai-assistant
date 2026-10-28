"""banko-ai load: generate banking traffic.

    banko-ai load run --rate 100 --users 200 --duration 5m    run it here, headless
    banko-ai load start --url http://localhost:5000 --rate 100 drive a running app (its Platform page shows it)
    banko-ai load stop --url http://localhost:5000
    banko-ai load status --url http://localhost:5000
    banko-ai load users prepare 500                            create load customers ahead of a run
    banko-ai load users clear                                  remove them and everything they own
"""

from __future__ import annotations

import re
import sys
import time

import click

_sleep = time.sleep
_PROFILES = ("balanced", "payments", "browsing", "search", "custom")
_UNITS = {"": 1, "s": 1, "m": 60, "h": 3600}


def _seconds(value: str | None, name: str) -> int | None:
    """90, 90s, 5m or 1h; 0 means no limit."""
    if value is None:
        return None
    m = re.fullmatch(r"\s*(\d+)\s*([smh]?)\s*", value)
    if not m:
        raise click.BadParameter(f"{value!r} is not a duration like 90, 90s, 5m or 1h", param_hint=name)
    return int(m.group(1)) * _UNITS[m.group(2)]


def _weights(value: str | None) -> dict[str, int] | None:
    if value is None:
        return None
    out = {}
    for part in filter(None, (p.strip() for p in value.split(","))):
        op, _, weight = part.partition("=")
        if not weight.strip().isdigit():
            raise click.BadParameter(f"{part!r} is not operation=weight, for example purchase=60", param_hint="--weights")
        out[op.strip()] = int(weight)
    return out


def _body(profile, users, rate, workers, duration, ramp, weights) -> dict:
    """The settings given on the command line; the rest come from the
    server's or this machine's LOAD_DEFAULT_* settings."""
    body = {"profile": profile, "users": users, "rate": rate, "workers": workers,
            "duration_s": _seconds(duration, "--duration"), "ramp_s": _seconds(ramp, "--ramp"),
            "weights": _weights(weights)}
    return {k: v for k, v in body.items() if v is not None}


def _load_options(fn):
    for option in reversed((
        click.option("--profile", type=click.Choice(_PROFILES), help="What customers do (default balanced)."),
        click.option("--users", type=int, help="Simulated customers."),
        click.option("--rate", type=float, help="Operations per second."),
        click.option("--workers", type=int, help="Operations in flight at once."),
        click.option("--duration", help="How long: 90, 90s, 5m, 1h; 0 or none runs until stopped."),
        click.option("--ramp", help="Seconds to reach the full rate (default 10)."),
        click.option("--weights", help='With --profile custom: "purchase=60,search=40".'),
    )):
        fn = option(fn)
    return fn


def _ms(v) -> str:
    if v is None:
        return "-"
    return f"{v / 1000:.1f} s" if v >= 1000 else f"{v:.1f} ms" if v < 10 else f"{round(v)} ms"


def _line(s: dict, *, columns: bool = True) -> str:
    """One reading; in columns for a run's scrolling output."""
    w = (8, 7) if columns else (0, 0)
    return (f"{s['ops_s']:{w[0]}.1f} ops/s   p50 {_ms(s['p50']):>{w[1]}}   p95 {_ms(s['p95']):>{w[1]}}   "
            f"p99 {_ms(s['p99']):>{w[1]}}   errors {s['errors']}   declined {s['declined']}   dropped {s['dropped']}")


def _simulator():
    """A simulator in this process, on its own banko-load pool, through the
    app role when row-level security is on, as the app's own would be."""
    from .config.settings import get_config
    from .db import rls
    from .db.engine import with_application_name
    from .workers.traffic import TrafficSimulator
    cfg = get_config()
    try:
        rls.refresh_status(cfg.database_url)
    except Exception:  # an unreachable database fails the run's preparation with a plain reason
        pass
    url = with_application_name(rls.app_url() or cfg.database_url, "banko-load")
    return TrafficSimulator(url=url, pool_size=cfg.load_max_workers + 2)


def _http():
    import requests
    return requests.Session()


def _prepare(count: int):
    from .services.load_users import prepare
    return prepare(count)


def _clear() -> int:
    from .services.load_users import clear
    return clear()


@click.group("load")
def load():
    """Generate banking traffic: here, headless, or in a running app."""


@load.command("run")
@_load_options
@click.option("--max-error-rate", type=float, help="Exit 1 when errors pass this share of operations, e.g. 0.01.")
def run(profile, users, rate, workers, duration, ramp, weights, max_error_rate):
    """Run the load simulator in this terminal; Ctrl-C stops it."""
    from .workers.traffic import LoadConfig, LoadConfigError
    try:
        config = LoadConfig.from_dict(_body(profile, users, rate, workers, duration, ramp, weights))
    except LoadConfigError as exc:
        raise click.UsageError(str(exc)) from None
    sim = _simulator()
    sim.start(config)
    click.echo(f"Load: {config.describe()}")
    started = time.monotonic()
    try:
        while True:
            _sleep(2)
            state = sim.status()["state"]
            if state == "idle":
                break
            if state == "running":
                click.echo("  " + _line(sim.metrics.summary("load", window_s=2)))
            elif state == "preparing":
                click.echo("  preparing customers")
    except KeyboardInterrupt:
        click.echo("\nStopping")
        sim.stop()
    error = sim.status().get("error")
    if error:
        click.echo(error)
        sys.exit(1)
    span = int(time.monotonic() - started) + 3
    s = sim.metrics.summary("load", window_s=span, now=sim.metrics.clock() + 1)
    done = s["ok"] + s["declined"]
    click.echo(f"Summary: {done:,} operations ({s['declined']:,} declined), {s['errors']:,} errors, "
               f"{s['dropped']:,} dropped; p50 {_ms(s['p50'])}, p95 {_ms(s['p95'])}, p99 {_ms(s['p99'])}")
    attempted = done + s["errors"]
    if max_error_rate is not None and attempted and s["errors"] / attempted > max_error_rate:
        click.echo(f"Error rate {s['errors'] / attempted:.2%} is above {max_error_rate:.2%}")
        sys.exit(1)


def _remote(url: str, persona: str):
    http = _http()
    r = http.post(url + "/login", data={"username": persona}, timeout=10, allow_redirects=False)
    if r.status_code >= 400:
        raise click.ClickException(f"Could not sign in to {url} as {persona}")
    return http


def _answer(r) -> dict:
    try:
        body: dict = r.json()
    except ValueError:
        body = {}
    if r.status_code >= 400:
        click.echo(body.get("error") or f"The app answered {r.status_code}")
        sys.exit(1)
    return body


@load.command("start")
@click.option("--url", default="http://localhost:5000", show_default=True, help="The running app.")
@click.option("--persona", default="maya", show_default=True, help="Who to sign in as.")
@_load_options
def start(url, persona, profile, users, rate, workers, duration, ramp, weights):
    """Start the load simulator in a running app."""
    url = url.rstrip("/")
    body = {"action": "start", **_body(profile, users, rate, workers, duration, ramp, weights)}
    status = _answer(_remote(url, persona).post(url + "/api/v1/platform/load", json=body, timeout=30))
    click.echo(f"Load is {status.get('state', 'starting')} in {url}; watch it on {url}/platform")


@load.command("stop")
@click.option("--url", default="http://localhost:5000", show_default=True)
@click.option("--persona", default="maya", show_default=True)
def stop(url, persona):
    """Stop the load simulator in a running app."""
    url = url.rstrip("/")
    status = _answer(_remote(url, persona).post(url + "/api/v1/platform/load", json={"action": "stop"}, timeout=30))
    click.echo(f"Load is {status.get('state', 'stopped')} in {url}")


@load.command("status")
@click.option("--url", default="http://localhost:5000", show_default=True)
@click.option("--persona", default="maya", show_default=True)
def status(url, persona):
    """What the load simulator in a running app is doing."""
    url = url.rstrip("/")
    live = _answer(_remote(url, persona).get(url + "/api/v1/platform/live", timeout=10))
    load_state = live["load"]
    line = load_state["state"]
    if load_state["state"] == "running" and load_state.get("config"):
        c = load_state["config"]
        line += f" for {int(load_state['elapsed_s'])} s: {c['profile']}, {c['users']} customers, {c['rate']:g} ops/s"
    click.echo(line)
    click.echo(_line(live["summary"], columns=False))


@load.group("users")
def users():
    """The load simulator's customers."""


@users.command("prepare")
@click.argument("count", type=click.IntRange(1, None))
def prepare_users(count):
    """Create load customers up to COUNT (existing ones are kept)."""
    click.echo(f"{len(_prepare(count)):,} load customers ready")


@users.command("clear")
@click.option("--yes", is_flag=True, help="Do not ask.")
def clear_users(yes):
    """Remove every load customer and everything they own."""
    if not yes and not click.confirm("Remove every load customer and everything they own?"):
        click.echo("Aborted.")
        return
    click.echo(f"Removed {_clear():,} load customers")
