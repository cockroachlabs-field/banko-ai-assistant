"""`banko-ai run --generate-data N` reaches the warm-up's sample-data step."""

from click.testing import CliRunner

from banko_ai import cli
from banko_ai.config.settings import get_config


class _StubApp:
    debug = False
    extensions = {"banko_readiness": None}


def _run(monkeypatch, *args):
    seen = {}

    def fake_create_app():
        seen["count"] = get_config().sample_data_count
        return _StubApp()

    monkeypatch.setattr("banko_ai.web.app.create_app", fake_create_app)
    monkeypatch.setattr("banko_ai.web.server.serve", lambda *a, **k: None)
    result = CliRunner().invoke(cli.cli, ["run", *args])
    assert result.exit_code == 0, result.output
    return seen["count"]


def test_generate_data_flag_overrides_the_setting(monkeypatch):
    assert _run(monkeypatch, "--generate-data", "123") == 123


def test_without_the_flag_the_setting_wins(monkeypatch):
    monkeypatch.setenv("SAMPLE_DATA_COUNT", "777")
    assert _run(monkeypatch) == 777


def test_the_sample_data_step_generates_the_configured_count(monkeypatch):
    from banko_ai.web import app as app_module

    class _Result:
        def scalar(self):
            return 0

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute(self, *args, **kwargs):
            return _Result()

    class _Engine:
        def connect(self):
            return _Conn()

    asked = {}

    class _Generator:
        def __init__(self, url):
            pass

        def generate_and_save(self, count, **kwargs):
            asked["count"] = count
            return count

    monkeypatch.setenv("BANKO_SKIP_AUTOSETUP", "0")
    monkeypatch.setenv("SAMPLE_DATA_COUNT", "42")
    monkeypatch.setattr(app_module, "get_engine", lambda url: _Engine())
    monkeypatch.setattr("banko_ai.vector_search.generator.EnhancedExpenseGenerator", _Generator)
    assert app_module.auto_setup_data_if_needed("postgresql://unused") == "generated 42 sample expenses"
    assert asked["count"] == 42


class _SpyConn:
    def __init__(self, calls, value=0):
        self.calls, self.value = calls, value

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, statement, params=None):
        self.calls.append((str(statement), params or {}))
        value = self.value

        class _Result:
            def scalar(self):
                return value

        return _Result()


def _persona_ids():
    from banko_ai.config.catalog import get_catalog
    return {p.user_id for p in get_catalog().personas}


def test_the_sample_data_count_ignores_persona_rows(monkeypatch):
    from banko_ai.web import app as app_module

    calls = []

    class _Engine:
        def connect(self):
            return _SpyConn(calls, value=500)

    monkeypatch.setenv("BANKO_SKIP_AUTOSETUP", "0")
    monkeypatch.setattr(app_module, "get_engine", lambda url: _Engine())
    app_module.auto_setup_data_if_needed("postgresql://unused")
    sql, params = calls[0]
    assert "LIMIT 100" in sql
    assert set(params.get("personas", [])) == _persona_ids()


def test_clearing_sample_data_keeps_persona_expenses(monkeypatch):
    from banko_ai.vector_search.generator import EnhancedExpenseGenerator

    calls = []

    class _Engine:
        def begin(self):
            return _SpyConn(calls)

    monkeypatch.setattr(EnhancedExpenseGenerator, "engine", property(lambda self: _Engine()))
    gen = EnhancedExpenseGenerator.__new__(EnhancedExpenseGenerator)
    assert gen.clear_expenses() is True
    sql, params = calls[0]
    assert sql.startswith("DELETE FROM expenses") and "WHERE" in sql
    assert set(params.get("personas", [])) == _persona_ids()
