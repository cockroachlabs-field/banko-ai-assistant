"""
Command-line interface for Banko AI Assistant.

This module provides CLI commands for running the application and managing data.
"""


import click

from .config.settings import get_config, set_config
from .db.engine import mask_url
from .load_cli import load as load_group


@click.group()
@click.pass_context
def cli(ctx):
    """Banko AI Assistant: the CockroachDB banking demo.

    \b
    Quick start:
      banko-ai doctor    is this machine ready?
      banko-ai run       start the app on http://localhost:5000

    The README covers setup per provider; banko-ai config --docs lists every setting.
    """
    pass


@cli.command()
@click.option('--host', default=None, help='Host to bind to')
@click.option('--port', default=None, type=int, help='Port to bind to')
@click.option('--debug', is_flag=True, help='Enable debug mode')
@click.option('--no-data', is_flag=True, help='Start without seeding the personas')
@click.option('--background', is_flag=True, help='Run in background mode (suppress Flask output)')
@click.option('--generate-data', hidden=True, default=None)
@click.option('--clear-data', hidden=True, is_flag=True)
def run(host, port, debug, no_data, background, generate_data, clear_data):
    """Run the Banko AI Assistant web application."""
    if generate_data is not None:
        raise click.UsageError("there is no sample data any more: each user's history is seeded. See banko-ai seed.")
    if clear_data:
        raise click.UsageError("to start a user over, use banko-ai reset.")
    config = get_config()
    host = host or config.host
    port = port or config.port
    click.echo("🏦 === Banko AI Assistant Starting === 🏦")
    if config.ai_display_provider:
        click.echo(f"🤖 AI: shown as {config.ai_display_provider} · {config.ai_display_model or 'default'}, "
                   f"served by {config.ai_service}")
    else:
        click.echo(f"🤖 AI Service: {config.ai_service}")
    click.echo("=" * 44)

    if no_data:
        # The warm-up's seeding steps read this setting.
        set_config(config.model_copy(update={"banko_skip_autosetup": True}))

    from .web.app import create_app
    from .web.server import serve

    app = create_app()
    app.debug = bool(debug or config.debug)  # no reloader: serve() runs in this process
    readiness = app.extensions["banko_readiness"]

    def announce_when_warm():
        # The ready line means ready: it waits for every warm-up step.
        import time
        while not readiness.finished():
            time.sleep(0.5)
        click.echo("🎉 Banko AI is ready to help with your finances!")
        snap = readiness.snapshot()
        failed = [name for name, c in snap.items() if c["state"] == "failed"]
        if failed:
            click.echo(f"⚠️  Not ready: {', '.join(failed)} (details at /api/health)")
        for name, c in snap.items():
            if c["state"] == "degraded":
                click.echo(f"⚠️  {name}: {c['detail']}")

    def announce(urls):
        import threading
        click.echo(f"🚀 Starting server on http://localhost:{port}")
        click.echo("=" * 44)
        for url in urls:
            click.echo(f" * Running on {url}")
        if background:
            click.echo("💡 Use 'banko-ai doctor' to check it")
            click.echo("🛑 Use 'pkill -f banko-ai' to stop the background process")
        threading.Thread(target=announce_when_warm, name="banko-ready-line", daemon=True).start()

    if background:
        import logging
        logging.getLogger('werkzeug').setLevel(logging.ERROR)
    serve(app, host, port, on_ready=announce)


def _moved(name: str, replacement: str, why: str):
    """A 1.x command kept as a hidden pointer to what replaced it."""
    @cli.command(name, hidden=True, context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
    def moved():
        raise click.UsageError(f"{why} Use {replacement}.")
    moved.__doc__ = f"Moved: use {replacement}."
    return moved


_moved("generate-data", "banko-ai seed", "there is no sample data any more: each user's history is seeded.")
_moved("clear-data", "banko-ai reset", "expenses belong to users now.")


@cli.command()
@click.option('--agents', is_flag=True, help='Clear agent tables (agent_state, agent_decisions, agent_memory, agent_tasks)')
@click.option('--cache', is_flag=True, help='Clear cache tables (vector_search_cache, query_cache)')
@click.option('--all', 'clear_all', is_flag=True, help='Clear both agents and cache')
@click.confirmation_option(prompt='Are you sure you want to clear this data?')
def clear_agents(agents, cache, clear_all):
    """Clear agent state and cache to start fresh.
    
    This is useful when:
    - Testing new agent features
    - Demoing to different audiences
    - Resetting agent learning
    
    Examples:
      banko-ai clear-agents --agents        # Clear only agent tables
      banko-ai clear-agents --cache         # Clear only cache
      banko-ai clear-agents --all           # Clear everything
    """
    from sqlalchemy import text

    from .db.engine import get_engine

    config = get_config()
    engine = get_engine(config.database_url)
    
    tables_cleared = []
    
    try:
        with engine.connect() as conn:
            # Clear agent tables
            if agents or clear_all:
                click.echo("🗑️  Clearing agent tables...")
                
                # Clear in order to avoid foreign key constraints
                agent_tables = [
                    'agent_tasks',
                    'agent_memory', 
                    'agent_decisions',
                    'agent_state'
                ]
                
                for table in agent_tables:
                    try:
                        result = conn.execute(text(f"DELETE FROM {table}"))
                        rows_deleted = result.rowcount
                        conn.commit()
                        tables_cleared.append(f"{table} ({rows_deleted} rows)")
                        click.echo(f"  ✅ Cleared {table}: {rows_deleted} rows")
                    except Exception as e:
                        click.echo(f"  ⚠️  Could not clear {table}: {e}")
            
            # Clear cache tables
            if cache or clear_all:
                click.echo("🗑️  Clearing cache tables...")
                
                cache_tables = [
                    'vector_search_cache',
                    'query_cache'
                ]
                
                for table in cache_tables:
                    try:
                        result = conn.execute(text(f"DELETE FROM {table}"))
                        rows_deleted = result.rowcount
                        conn.commit()
                        tables_cleared.append(f"{table} ({rows_deleted} rows)")
                        click.echo(f"  ✅ Cleared {table}: {rows_deleted} rows")
                    except Exception as e:
                        click.echo(f"  ⚠️  Could not clear {table}: {e}")

        if tables_cleared:
            click.echo(f"\n✅ Successfully cleared: {', '.join(tables_cleared)}")
            click.echo("\n💡 Tip: Agent tables will be repopulated when you upload a new receipt")
        else:
            click.echo("\n⚠️  No tables were cleared. Use --agents, --cache, or --all")

    except Exception as e:
        click.echo(f"\n❌ Error clearing data: {e}")


_moved("status", "banko-ai doctor", "status folded into doctor.")


@cli.command()
@click.option('--host', default=None, help='Host to bind to')
@click.option('--port', default=None, type=int, help='Port to bind to')
@click.option('--no-data', is_flag=True, help='Start without seeding the personas')
@click.pass_context
def start(ctx, host, port, no_data):
    """Start Banko AI in background mode (equivalent to 'run --background')."""
    ctx.invoke(run, host=host, port=port, debug=False, no_data=no_data, background=True,
               generate_data=None, clear_data=False)


@cli.command()
def help():
    """Where to find setup, settings and the demo guide."""
    click.echo("""Banko AI Assistant

  banko-ai doctor          check the database, provider, models and ports
  banko-ai run             start the app (PORT and HOST honored)
  banko-ai config          every setting in effect and where it came from
  banko-ai config --docs   the full settings reference as Markdown
  banko-ai seed / reset    give users their history, or start them over
  banko-ai load run        generate banking traffic

Setup per provider, the demo guide and upgrading from 1.x are in the README:
https://github.com/cockroachlabs-field/banko-ai-assistant""")


@cli.command()
@click.argument('query')
@click.option('--user', 'username', default='maya', show_default=True, help='Whose transactions to search')
@click.option('--limit', default=10, show_default=True, help='Maximum number of results')
def search(query, username, limit):
    """Search a user's purchases by meaning (their own vector index)."""
    from .services import transactions
    from .services.users import user_ids_by_name

    user_id = user_ids_by_name([username]).get(username)
    if user_id is None:
        raise click.ClickException(f"no user called {username}")
    found = transactions.semantic_search(user_id, query, limit=limit)["items"]
    if not found:
        click.echo("No results found")
        return
    click.echo(f"{len(found)} of {username}'s purchases nearest to '{query}':")
    for item in found:
        amount = f"${abs(item['amount']):,.2f}"
        click.echo(f"  {item['posted_at']}  {item['merchant'] or '-':<28} {amount:>11}  {item.get('category') or ''}")


_MARKS = {"ok": "✅", "fail": "❌", "skip": "➖"}


@cli.command()
def doctor():
    """Is this machine ready to demo? One line per check; exits 1 if any fail."""
    from . import doctor as checks

    results = checks.run_checks()
    width = max(len(c.name) for c in results)
    for c in results:
        click.echo(f"{_MARKS.get(c.state, '?')} {c.name:<{width}}  {c.detail}")
    if any(c.state == "fail" for c in results):
        raise SystemExit(1)


def _progress():
    """Seeding progress, one line per user when they finish."""
    def report(user_id: str, done: int, total: int, state: str) -> None:
        if state != "running":
            click.echo(f"  {user_id[:8]}  {done:,} of {total:,} entries, {state}")
    return report


@cli.command()
@click.argument('usernames', nargs=-1)
@click.option('--yes', is_flag=True, help='Do not ask')
def reset(usernames, yes):
    """Start users over: delete their money rows and reseed their history.

    With no names, resets the personas. Coach signals and nudges stay.
    """
    from .config.catalog import get_catalog
    from .services import seeding
    from .services.users import user_ids_by_name

    if usernames:
        names = list(usernames)
        found = user_ids_by_name(names)
    else:
        # The personas by id: someone who signed up as "maya" first is not the persona.
        personas = get_catalog().personas
        names, found = [p.username for p in personas], {p.username: p.user_id for p in personas}
    missing = [n for n in names if n not in found]
    if missing:
        click.echo(f"No user called {', '.join(missing)}; nothing was reset.")
        raise SystemExit(1)
    if not yes and not click.confirm(f"Reset {', '.join(names)}? Their history is deleted and reseeded."):
        click.echo("Aborted.")
        return
    busy = []
    for name in names:
        written = seeding.reset_user(found[name], on_progress=_progress())
        click.echo(f"{name}: {written:,} entries seeded" if written else f"{name}: busy seeding, try again shortly")
        if not written:
            busy.append(name)
    if busy:
        raise SystemExit(1)


@cli.command()
@click.option('--users', type=click.IntRange(1, None), default=None, help='Create this many new demo users')
@click.option('--months', type=click.IntRange(1, 36), default=None, help='Months of history for new users')
def seed(users, months):
    """Seed the personas' history, or create new demo users with theirs."""
    from .services import seeding

    if months and not users:
        raise click.UsageError("--months sets the history of new users; use it with --users.")
    if users:
        made = seeding.seed_generated_users(users, months=months, on_progress=_progress())
        click.echo(f"{len(made)} demo users created and seeded")
        return
    written = seeding.seed_personas(on_progress=_progress())
    click.echo(", ".join(f"{name} +{n:,}" for name, n in written.items()) or "personas already seeded")


@cli.command()
def migrate():
    """Apply the schema registry now instead of on the next boot."""
    from .db import schema

    result = schema.ensure_schema(force=True, on_progress=click.echo)
    click.echo(f"Schema current: {result.statements} statements in {result.seconds:.1f} s")
    done = schema.run_pending_conversions()
    click.echo(f"{len(done)} locality conversion{'' if len(done) == 1 else 's'}")


@cli.command("config")
@click.option("--docs", is_flag=True, help="Print the README configuration reference instead.")
def config_cmd(docs):
    """Show the effective configuration and where each value came from."""
    from .config.settings import GROUPS, render_config_docs

    if docs:
        click.echo(render_config_docs(), nl=False)
        return
    current = None
    for info in get_config().describe():
        if info.group != current:
            current = info.group
            click.echo(f"\n[{GROUPS[info.group]}]")
        click.echo(f"  {info.env:<32} {info.value:<56} ({info.source})")


@cli.command()
@click.option('--yes', is_flag=True, help='Skip confirmation prompt')
def clear_demo_users(yes):
    """Clear all demo users and every row they own (keeps the personas).

    Signed-up, generated and load users go, with their accounts, ledger,
    transfers, goals, budgets, expenses, receipts, conversations, memories,
    caches, signals and nudges. The personas maya, sam and riley stay.
    """
    if not yes:
        if not click.confirm('Delete all demo users and their data?'):
            click.echo('Aborted.')
            return

    config = get_config()
    from .web.auth import clear_demo_users as clear_fn

    click.echo('Clearing demo users...')
    counts = clear_fn(config.database_url)

    click.echo(f'Deleted {counts["users"]} demo users and every row they owned.')


cli.add_command(load_group)


def main():
    """Main CLI entry point."""
    cli()


if __name__ == '__main__':
    main()
