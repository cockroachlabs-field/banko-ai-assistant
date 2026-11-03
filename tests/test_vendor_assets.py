"""Every vendored asset is local, listed, licensed and unmodified."""

import hashlib
import json
import re
from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "banko_ai" / "static"
MANIFEST = STATIC / "vendor" / "manifest.json"
REQUIRED = {"alpinejs", "chart.js", "socket.io-client", "marked", "dompurify", "inter", "lucide-static"}


def _entries():
    return json.loads(MANIFEST.read_text())


def test_manifest_lists_every_required_library():
    names = {e["name"] for e in _entries()}
    assert REQUIRED <= names


def test_files_exist_and_match_their_hash():
    for e in _entries():
        path = STATIC / e["file"]
        assert path.is_file(), e["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == e["sha256"], e["file"]


def _v(text):
    return tuple(int(part) for part in text.split(".")[:3])


def test_versions_meet_the_floors():
    """Nothing older than the releases OSV showed clean on 2026-10-13."""
    versions = {e["name"]: _v(e["version"]) for e in _entries()}
    assert versions["dompurify"] >= (3, 4, 17)
    assert (3, 17, 4) <= versions["alpinejs"] < (4,)
    assert (4, 8, 4) <= versions["socket.io-client"] < (5,)
    assert versions["marked"] >= (18, 1, 1)
    assert (4, 5, 1) <= versions["chart.js"] < (5,)
    assert versions["lucide-static"] >= (1, 56, 0)


def test_every_library_has_a_license_file():
    for e in _entries():
        assert (STATIC / "vendor" / "LICENSES" / f"{e['name']}.txt").is_file(), e["name"]


def test_sprite_has_every_listed_icon():
    import importlib.util
    spec = importlib.util.spec_from_file_location("sprite", Path(__file__).resolve().parents[1] / "scripts" / "build_icon_sprite.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sprite = (STATIC / "vendor" / "lucide-sprite.svg").read_text()
    ids = set(re.findall(r'<symbol id="i-([a-z0-9-]+)"', sprite))
    assert set(mod.ICONS) == ids
