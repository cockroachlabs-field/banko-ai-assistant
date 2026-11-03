"""v2 seeds every user's own history. The 1.x generator of 5,000 sample
expenses that belonged to nobody is gone, and its knobs point to what
replaced it instead of failing quietly."""

import os

import pytest
from click.testing import CliRunner

from banko_ai import cli
from banko_ai.config import settings as S

needs_db = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")


@pytest.mark.parametrize("flag, pointer", [(["--generate-data", "5"], "banko-ai seed"),
                                           (["--clear-data"], "banko-ai reset")])
def test_run_points_old_data_flags_to_their_replacements(flag, pointer, monkeypatch):
    def no_server(*a, **k):
        raise AssertionError("run went on to start the app")

    monkeypatch.setattr("banko_ai.web.app.create_app", no_server)
    monkeypatch.setattr("banko_ai.web.server.serve", no_server)
    result = CliRunner().invoke(cli.cli, ["run", *flag])
    assert result.exit_code == 2
    assert pointer in result.output


def test_the_sample_data_settings_are_retired():
    assert {"SAMPLE_DATA_COUNT", "DATA_GEN_BATCH_SIZE"} <= set(S.REMOVED_SETTINGS)
    assert "seed" in S.REMOVED_SETTINGS["SAMPLE_DATA_COUNT"]
    assert "use SAMPLE_DATA_COUNT" not in S.REMOVED_SETTINGS["DEFAULT_RECORD_COUNT"]
    fields = {S.field_meta(f).env for f in S.Settings.model_fields.values()}
    assert not fields & {"SAMPLE_DATA_COUNT", "DATA_GEN_BATCH_SIZE"}


@needs_db
def test_boot_generates_no_sample_data_and_the_generator_routes_are_gone():
    from banko_ai.web.app import create_app
    app = create_app()
    try:
        routes = {r.rule for r in app.url_map.iter_rules()}
        assert not routes & {"/api/generate-data", "/api/stop-generation", "/api/reset-generation"}
        assert "sample_data" not in app.extensions["banko_readiness"].snapshot()
    finally:
        app.extensions["banko_warmup_stop"].set()
