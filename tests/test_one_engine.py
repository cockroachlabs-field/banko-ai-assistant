"""Nothing outside banko_ai/db/engine.py builds or disposes an engine.

Every create_engine() call built a fresh connection pool (or, with NullPool,
a fresh TCP connection per query), and every dispose() threw a pool away.
Under the chaos cluster's realistic latency that churn is most of a
request's database time."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "banko_ai"
ENGINE_MODULE = ROOT / "db" / "engine.py"


def _offenders(pattern: str) -> list[str]:
    rx = re.compile(pattern)
    found = []
    for path in sorted(ROOT.rglob("*.py")):
        if path == ENGINE_MODULE:
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if rx.search(line.split("#", 1)[0]):
                found.append(f"{path.relative_to(ROOT.parent)}:{lineno}: {line.strip()}")
    return found


def test_nothing_builds_its_own_engine():
    assert _offenders(r"\bcreate_engine\(") == []


def test_nothing_disposes_the_shared_engine():
    assert _offenders(r"\.dispose\(\)") == []
