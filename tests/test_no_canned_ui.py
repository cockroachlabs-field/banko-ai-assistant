"""The chat page never fabricates activity: no random cache hits, no
hardcoded provider names in the DB Ops panel."""

from pathlib import Path

INDEX = Path(__file__).resolve().parents[1] / "banko_ai" / "templates" / "index.html"


def test_db_ops_panel_has_no_simulated_operations():
    html = INDEX.read_text()
    assert "simulateDatabaseOperations" not in html
    assert "OpenAI/gpt-4o-mini" not in html
    assert "cacheScenarios" not in html
