"""banko-ai doctor answers every check within seconds, on a machine that is
ready and on one that has nothing, and exits 1 when anything is red."""

import os
import socket
import time

import pytest
from click.testing import CliRunner
from sqlalchemy import text
from sqlalchemy.engine import make_url

from banko_ai import cli
from banko_ai.config.settings import reset_config

DB = os.getenv("DATABASE_URL")
needs_db = pytest.mark.skipif(not DB, reason="DATABASE_URL not set")
NOWHERE = "postgresql://root@127.0.0.1:1/defaultdb?sslmode=require&connect_timeout=2"


@pytest.fixture
def env(monkeypatch):
    """Settings from the environment, reloaded around each test."""
    for name in ("CRDB_ADMIN_URL", "CHAOS_CONTROL_URL", "AI_DISPLAY_PROVIDER", "AI_DISPLAY_MODEL"):
        monkeypatch.delenv(name, raising=False)

    def set_env(**values):
        for name, value in values.items():
            monkeypatch.setenv(name, value)
        reset_config()

    yield set_env
    monkeypatch.undo()
    reset_config()


def _by_name(checks):
    return {c.name: c for c in checks}


def test_a_bare_machine_gets_an_answer_for_every_check(env, monkeypatch):
    from banko_ai import doctor
    env(DATABASE_URL=NOWHERE, AI_SERVICE="ollama", OLLAMA_HOST="http://127.0.0.1:1")
    monkeypatch.setattr(doctor.shutil, "which", lambda name: None)
    started = time.monotonic()
    found = _by_name(doctor.run_checks())
    assert time.monotonic() - started < 30
    assert set(found) == {"database", "schema", "embedding model", "AI provider", "Ollama model", "tesseract",
                          "port", "status API", "chaos control"}
    assert found["database"].state == "fail" and "DATABASE_URL" in found["database"].detail
    assert "127.0.0.1:1" not in found["database"].detail
    assert found["schema"].state == "skip"
    assert found["Ollama model"].state == "fail" and "ollama serve" in found["Ollama model"].detail
    assert found["tesseract"].state == "fail"
    assert found["status API"].state == "skip" and found["chaos control"].state == "skip"


def test_a_check_that_breaks_is_a_red_line_not_a_traceback(env, monkeypatch):
    from banko_ai import doctor

    def broken(cfg):
        raise KeyError("boom")

    env(DATABASE_URL=NOWHERE)
    monkeypatch.setattr(doctor, "CHECKS", (doctor.CHECKS[0], broken))
    out = doctor.run_checks()
    assert out[-1].state == "fail" and out[-1].detail == "KeyError"


def test_the_port_check_tells_a_taken_port_from_a_free_one(env):
    from banko_ai import doctor
    with socket.socket() as held:
        held.bind(("127.0.0.1", 0))
        held.listen()
        port = held.getsockname()[1]
        env(PORT=str(port), HOST="127.0.0.1")
        taken = doctor._port(doctor.get_config())
    env(PORT=str(port))
    free = doctor._port(doctor.get_config())
    assert taken.state == "fail" and "PORT" in taken.detail
    assert free.state == "ok"


def test_a_persona_is_named_next_to_the_real_provider(env, monkeypatch):
    from banko_ai import doctor
    from banko_ai.ai_providers.factory import AIProviderFactory

    class Answers:
        def test_connection(self):
            return True

    env(AI_SERVICE="openai", OPENAI_API_KEY="sk-test", AI_DISPLAY_PROVIDER="watsonx", AI_DISPLAY_MODEL="granite")
    monkeypatch.setattr(AIProviderFactory, "create_provider", staticmethod(lambda service, config: Answers()))
    line = doctor._provider(doctor.get_config())
    assert line.state == "ok" and "openai" in line.detail and "shown as watsonx" in line.detail


@needs_db
def test_schema_state_reads_missing_current_and_out_of_date():
    from banko_ai.db.engine import get_engine
    from banko_ai.db.schema import ensure_schema, schema_state
    admin = get_engine(DB)
    url = make_url(DB).set(database="banko_doctor_test").render_as_string(hide_password=False)
    with admin.connect() as c:
        c.execution_options(isolation_level="AUTOCOMMIT")
        c.execute(text("DROP DATABASE IF EXISTS banko_doctor_test CASCADE"))
        c.execute(text("CREATE DATABASE banko_doctor_test"))
    try:
        assert schema_state(url) == "missing"
        ensure_schema(url)
        assert schema_state(url) == "current"
        with get_engine(url).begin() as c:
            c.execute(text("UPDATE banko_meta SET value = jsonb_set(value::JSONB, '{fingerprint}', '\"old\"')::STRING "
                           "WHERE key = 'schema_fingerprint'"))
        assert schema_state(url) == "outdated"
    finally:
        with admin.connect() as c:
            c.execution_options(isolation_level="AUTOCOMMIT")
            c.execute(text("DROP DATABASE IF EXISTS banko_doctor_test CASCADE"))


@pytest.mark.parametrize("states, code", [(("ok", "skip"), 0), (("ok", "fail"), 1)])
def test_doctor_prints_a_line_per_check_and_exits_1_on_red(monkeypatch, states, code):
    from banko_ai import doctor
    checks = [doctor.Check(f"check {i}", s, f"detail {i}") for i, s in enumerate(states)]
    monkeypatch.setattr(doctor, "run_checks", lambda cfg=None: checks)
    result = CliRunner().invoke(cli.cli, ["doctor"])
    assert result.exit_code == code
    assert "check 0" in result.output and "detail 1" in result.output


@pytest.mark.parametrize("db_regions, state", [([], "skip"), (["us-east-1"], "ok")])
def test_a_derived_status_api_counts_only_when_it_is_this_databases_cluster(env, monkeypatch, db_regions, state):
    """localhost:8080 can be another local cluster's DB Console."""
    from banko_ai import doctor

    class OtherConsole:
        base_url = "http://localhost:8080"

        def nodes(self):
            return [{"node_id": 1, "region": "us-east-1", "live": True}]

    env(DATABASE_URL="postgresql://root@localhost:26258/defaultdb?sslmode=disable")
    monkeypatch.setattr("banko_ai.clients.crdb_status.status_client", lambda: OtherConsole())
    monkeypatch.setattr("banko_ai.services.platform.cluster_regions", lambda url=None: db_regions)
    line = doctor._status_api(doctor.get_config())
    assert line.state == state
    if state == "skip":
        assert "another cluster" in line.detail and "CRDB_ADMIN_URL" in line.detail


@pytest.mark.parametrize("configured, state", [("", "skip"), ("http://db-console:8080", "fail")])
def test_a_silent_status_api_is_red_only_when_someone_configured_it(env, monkeypatch, configured, state):
    from banko_ai import doctor
    from banko_ai.clients.crdb_status import StatusUnavailable

    class Silent:
        base_url = configured or "http://localhost:8080"

        def nodes(self):
            raise StatusUnavailable("The CockroachDB HTTP API did not answer.")

    env(CRDB_ADMIN_URL=configured)
    monkeypatch.setattr("banko_ai.clients.crdb_status.status_client", lambda: Silent())
    assert doctor._status_api(doctor.get_config()).state == state


def test_the_chaos_url_is_shown_without_its_credentials(env, monkeypatch):
    from banko_ai import doctor
    from banko_ai.clients.chaos import ChaosUnavailable

    class Down:
        def state(self):
            raise ChaosUnavailable("down")

    env(CHAOS_CONTROL_URL="http://admin:hunter2@chaos:8088")
    monkeypatch.setattr("banko_ai.clients.chaos.chaos_client", lambda: Down())
    line = doctor._chaos(doctor.get_config())
    assert line.state == "fail" and "hunter2" not in line.detail and "chaos:8088" in line.detail


def test_an_unreachable_database_host_answers_within_seconds(env):
    """A black-holed address (venue wifi half up) must not hold the line for
    the pool's ten-second connect timeout."""
    from banko_ai import doctor
    env(DATABASE_URL="postgresql://root@10.255.255.1:26257/defaultdb?sslmode=require")
    started = time.monotonic()
    line = doctor._database(doctor.get_config())
    assert line.state == "fail" and time.monotonic() - started < doctor.TIMEOUT + 2
