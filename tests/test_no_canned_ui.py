"""No page fabricates activity: no random numbers, no simulated database
operations, no hardcoded provider names."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "banko_ai"
BANNED = ("simulateDatabaseOperations", "OpenAI/gpt-4o-mini", "cacheScenarios", "Math.random")


def test_no_ui_file_fabricates_activity():
    files = [*(ROOT / "templates").rglob("*.html"), *(ROOT / "static" / "js").rglob("*.js")]
    assert files
    for path in files:
        text = path.read_text()
        for banned in BANNED:
            assert banned not in text, f"{banned} in {path.relative_to(ROOT)}"
