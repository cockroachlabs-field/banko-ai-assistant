"""The README is also the PyPI page, and PyPI keeps a release's text for good.

The 1.3.2 page linked two images on main. The clean-up for 2.0 deleted them,
and the page has shown two broken images since: a published description
cannot be edited. So an image the README shows has to be in the repository,
and a release pins its images to its own tag before it is built.

It is Docker Hub's overview as well. That page is not part of the image, and
nothing kept it current: after 2.0.0 it still showed the 1.x README with the
same two broken images. The Docker workflow publishes the README there."""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
DOCKER_WORKFLOW = ROOT / ".github" / "workflows" / "docker.yml"
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



def test_the_readme_has_no_link_that_only_github_can_follow():
    """GitHub resolves a relative link against the repository. PyPI and
    Docker Hub resolve it against themselves: on the 2.0.0 page the three
    links to the docs lead nowhere."""
    targets = re.findall(r"\]\(([^)\s]+)", README.read_text())
    assert len(targets) > 20                                          # the scan found the links
    relative = sorted({t for t in targets if not t.startswith(("https://", "#"))})
    assert relative == []


def test_the_readme_fits_docker_hubs_overview(tmp_path):
    """Docker Hub cuts an overview off at 25,000 bytes, and the step that
    publishes it only says so in a log nobody reads."""
    copy = tmp_path / "README.md"
    shutil.copy(README, copy)
    assert _pin(copy, "10.20.30").returncode == 0                     # as published: the pinned links are longer
    assert copy.stat().st_size <= 25_000


def _docker_workflow() -> dict:
    return yaml.safe_load(DOCKER_WORKFLOW.read_text())


def test_docker_hubs_overview_is_the_readme_of_the_release():
    """After the image is pushed, the README goes to Docker Hub as the
    overview, with what it links pinned to the release's tag, which has to
    exist by then."""
    overview = _docker_workflow()["jobs"]["overview"]
    steps = overview["steps"]
    kinds = [str(step.get("uses") or step.get("run") or "") for step in steps]
    checkout = next(i for i, kind in enumerate(kinds) if kind.startswith("actions/checkout@"))
    pin = next(i for i, kind in enumerate(kinds) if "scripts/pin_readme_images.py" in kind)
    publish = next(i for i, kind in enumerate(kinds) if kind.startswith("peter-evans/dockerhub-description@"))
    assert checkout < pin < publish
    # The pin moves links to the tag v<version>: without that tag every picture on the page would be broken.
    assert "ls-remote" in kinds[pin] and "refs/tags/v" in kinds[pin] and "exit 1" in kinds[pin]
    assert steps[pin]["env"] == {"VERSION": "${{ inputs.version }}"}  # the input reaches the shell as data, not as script
    assert "${{" not in kinds[pin]
    settings = steps[publish]["with"]
    assert settings["username"] == "${{ secrets.DOCKERHUB_USERNAME }}"
    assert settings["password"] == "${{ secrets.DOCKERHUB_TOKEN }}"
    assert settings["repository"] == "${{ secrets.DOCKERHUB_USERNAME }}/banko-ai-assistant"   # where the image job pushes
    assert settings.get("readme-filepath", "./README.md") == "./README.md"
    assert 0 < len(settings["short-description"].encode()) <= 100     # Docker Hub's limit, in bytes


def test_the_overview_can_be_refreshed_without_building_the_image():
    """The page of a release that is already out has to be correctable: the
    2.0.0 image was pushed before anything published the README."""
    workflow = _docker_workflow()
    inputs = (workflow.get("on") or workflow[True])["workflow_dispatch"]["inputs"]   # YAML 1.1 reads `on` as true
    assert inputs["overview_only"]["type"] == "boolean" and inputs["overview_only"]["default"] is False
    jobs = workflow["jobs"]
    assert jobs["docker"]["if"] == "${{ !inputs.overview_only }}"
    # The overview follows the image, and still runs when the image job was skipped on purpose.
    assert jobs["overview"]["needs"] == "docker"
    condition = jobs["overview"]["if"]
    assert "!cancelled()" in condition and "needs.docker.result == 'success'" in condition
    assert "needs.docker.result == 'skipped'" in condition
