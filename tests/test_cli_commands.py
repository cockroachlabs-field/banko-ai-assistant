"""The v2 CLI: reset, seed, migrate and search run on the same services the
app uses; help is short; status points to doctor."""

import os
from datetime import date
from decimal import Decimal

import pytest
from click.testing import CliRunner

from banko_ai import cli
from banko_ai.config.catalog import get_catalog

DB = os.getenv("DATABASE_URL")
needs_db = pytest.mark.skipif(not DB, reason="DATABASE_URL not set")


def _invoke(*args, input=None):
    return CliRunner().invoke(cli.cli, list(args), input=input)


@pytest.fixture
def personas():
    from banko_ai.services.seeding import ensure_personas
    ensure_personas(DB)
    return {p.username: p.user_id for p in get_catalog().personas}


@needs_db
def test_user_ids_by_name_finds_users_and_leaves_out_strangers(personas):
    from banko_ai.services.users import user_ids_by_name
    found = user_ids_by_name(["maya", "nobody-xyz"], url=DB)
    assert found == {"maya": personas["maya"]}


@needs_db
def test_reset_defaults_to_the_personas_and_asks_first(personas, monkeypatch):
    reset = []
    monkeypatch.setattr("banko_ai.services.seeding.reset_user", lambda uid, **kw: reset.append(uid) or 10)
    declined = _invoke("reset", input="n\n")
    assert "Aborted" in declined.output and reset == []
    accepted = _invoke("reset", input="y\n")
    assert accepted.exit_code == 0, accepted.output
    assert sorted(reset) == sorted(personas.values())


@needs_db
def test_reset_names_users_it_cannot_find_and_resets_nobody(personas, monkeypatch):
    reset = []
    monkeypatch.setattr("banko_ai.services.seeding.reset_user", lambda uid, **kw: reset.append(uid) or 10)
    result = _invoke("reset", "maya", "nobody-xyz", "--yes")
    assert result.exit_code == 1 and "nobody-xyz" in result.output and reset == []


def test_seed_writes_the_personas_by_default_and_new_users_on_request(monkeypatch):
    calls = []
    monkeypatch.setattr("banko_ai.services.seeding.seed_personas",
                        lambda url=None, on_progress=None: calls.append("personas") or {"maya": 5})
    monkeypatch.setattr("banko_ai.services.seeding.seed_generated_users",
                        lambda count, months=None, url=None, on_progress=None:
                        calls.append(("users", count, months)) or ["a", "b", "c"])
    assert _invoke("seed").exit_code == 0
    result = _invoke("seed", "--users", "3", "--months", "2")
    assert result.exit_code == 0, result.output
    assert calls == ["personas", ("users", 3, 2)]
    assert "3" in result.output


def test_migrate_applies_the_registry_and_reports_what_ran(monkeypatch):
    from banko_ai.db.schema import SchemaResult
    seen = {}

    def ensure(url=None, *, force=False, primary_region=None, on_progress=None):
        seen["force"] = force
        return SchemaResult(changed=True, fingerprint="abc", statements=7, seconds=1.25)

    monkeypatch.setattr("banko_ai.db.schema.ensure_schema", ensure)
    monkeypatch.setattr("banko_ai.db.schema.run_pending_conversions", lambda url=None: ["expenses"])
    result = _invoke("migrate")
    assert result.exit_code == 0, result.output
    assert seen["force"] is True
    assert "7 statements" in result.output and "1 locality conversion" in result.output


@needs_db
def test_search_runs_the_users_own_vector_search(personas, monkeypatch):
    asked = {}

    def search(user_id, q, *, limit=20, **kw):
        asked.update(user_id=user_id, q=q, limit=limit)
        return {"items": [{"merchant": "Blue Bottle", "amount": Decimal("-4.50"), "posted_at": date(2026, 10, 1),
                           "category": "Coffee"}], "next_cursor": None}

    monkeypatch.setattr("banko_ai.services.transactions.semantic_search", search)
    result = _invoke("search", "coffee", "--user", "maya", "--limit", "5")
    assert result.exit_code == 0, result.output
    assert asked == {"user_id": personas["maya"], "q": "coffee", "limit": 5}
    assert "Blue Bottle" in result.output and "4.50" in result.output


def test_help_is_a_short_pointer():
    result = _invoke("help")
    assert result.exit_code == 0
    assert len(result.output.strip().splitlines()) <= 15
    assert "README" in result.output and "banko-ai config --docs" in result.output


def test_status_points_to_doctor():
    result = _invoke("status")
    assert result.exit_code == 2 and "banko-ai doctor" in result.output


def test_the_command_list_is_the_v2_set():
    listed = _invoke("--help").output
    for name in ("doctor", "reset", "seed", "migrate", "search", "config", "load", "run", "clear-demo-users"):
        assert f"  {name} " in listed, name
    for gone in ("status", "generate-data", "clear-data"):
        assert f"  {gone} " not in listed, gone
