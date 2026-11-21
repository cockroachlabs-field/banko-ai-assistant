"""The README is also the PyPI page, and PyPI keeps a release's text for good.

The 1.3.2 page linked two images on main. The clean-up for 2.0 deleted them,
and the page has shown two broken images since: a published description
cannot be edited. So an image the README shows has to be in the repository,
and a release pins its images to its own tag before it is built."""

import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
REPO = "cockroachlabs-field/banko-ai-assistant"
PIN = ROOT / "scripts" / "pin_readme_images.py"
# An address into this repository's files at a branch or tag: raw, or the page for a file.
REPO_FILE = re.compile(rf"https://(?:raw\.githubusercontent\.com/{REPO}|github\.com/{REPO}/blob)/([^/\s)]+)/([^\s)\"']+)")


def test_the_badges_say_whether_ci_passes_and_how_often_it_is_downloaded():
    top = README.read_text().split("# Banko AI Assistant", 1)[0]
    assert f"https://github.com/{REPO}/actions/workflows/ci.yml/badge.svg" in top
    assert f"(https://github.com/{REPO}/actions/workflows/ci.yml)" in top
    assert "https://static.pepy.tech/badge/banko-ai-assistant" in top
    assert "(https://pepy.tech/projects/banko-ai-assistant" in top
    for kept in ("img.shields.io/pypi/v/banko-ai-assistant", "img.shields.io/pypi/pyversions/banko-ai-assistant",
                 "img.shields.io/pypi/l/banko-ai-assistant", "img.shields.io/docker/pulls/virag/banko-ai-assistant"):
        assert kept in top, kept
    assert (ROOT / ".github" / "workflows" / "ci.yml").is_file()      # the workflow the badge reports on


def test_every_file_the_readme_links_in_this_repository_is_there():
    linked = REPO_FILE.findall(README.read_text())
    assert len(linked) >= 13                                          # the screenshots, light and dark, and the diagram
    for ref, path in linked:
        assert ref == "main", f"{path} is linked at {ref}; the README in the repository links main"
        assert (ROOT / path).is_file(), f"the README links {path}, which is not in the repository"


def _pin(readme: Path, version: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(PIN), version, str(readme)], capture_output=True, text=True)


def test_a_release_pins_what_the_readme_shows_to_its_own_tag(tmp_path):
    copy = tmp_path / "README.md"
    shutil.copy(README, copy)
    done = _pin(copy, "9.9.9")
    assert done.returncode == 0, done.stderr
    pinned = REPO_FILE.findall(copy.read_text())
    assert len(pinned) == len(REPO_FILE.findall(README.read_text()))
    assert {ref for ref, _ in pinned} == {"v9.9.9"}                   # every one, raw files and file pages alike
    # Nothing else moved: the badges and the links to the repository itself still point where they did.
    before = REPO_FILE.sub("", README.read_text())
    assert REPO_FILE.sub("", copy.read_text()) == before
    assert f"github.com/{REPO}/actions/workflows/ci.yml" in copy.read_text()


def test_pinning_says_so_when_there_was_nothing_to_pin(tmp_path):
    """A README whose links were changed to something this does not know
    must not go to PyPI as if it had been pinned."""
    empty = tmp_path / "README.md"
    empty.write_text("# Banko\n\nNo pictures here.\n")
    done = _pin(empty, "9.9.9")
    assert done.returncode != 0 and "nothing" in (done.stdout + done.stderr).lower()
    assert _pin(empty, "not a version").returncode != 0


def test_the_release_pins_the_images_before_it_builds_the_package():
    steps = (ROOT / ".github" / "workflows" / "release.yml").read_text()
    pin, build = steps.index("scripts/pin_readme_images.py"), steps.index("python -m build")
    assert pin < build
    assert "${{ inputs.version }}" in steps[pin:steps.index("\n", pin)]
