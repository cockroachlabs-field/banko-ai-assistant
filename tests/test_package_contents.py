"""The wheel ships what the app uses and nothing else."""

from importlib.metadata import version
from pathlib import Path

import banko_ai

STATIC = Path(banko_ai.__file__).parent / "static"

MAX_STATIC_BYTES = 3 * 1024 * 1024  # static/ is about 2.3 MB after the cleanup


def test_dead_assets_are_gone():
    for name in ("creditcard_files", "creditcard.png", "Anallytics.png", "profilepic.jpeg",
                 "banko-ai-architecture.png", "banko-ai-assistant-watsonx.gif"):
        assert not (STATIC / name).exists(), name


def test_static_assets_stay_small():
    total = sum(p.stat().st_size for p in STATIC.rglob("*") if p.is_file())
    assert total < MAX_STATIC_BYTES, f"static/ is {total / 1e6:.1f} MB"


def test_version_matches_the_package():
    assert banko_ai.__version__ == version("banko-ai-assistant")
