"""Known vulnerabilities in every version uv.lock pins.

pip-audit takes one version per package and checks only the requirements
whose environment markers match its own interpreter, so a version locked
for another Python version (the lock forks by Python version) was never
audited. This reads uv.lock directly and audits every PyPI package version
in it, every extra, group and Python version included, in batches with one
version per package. It asks the OSV feed, which carries the GitHub
advisories Dependabot reports; PyPI's own feed lacks some of them.

    uv run --no-project python scripts/audit_lock.py   (or: make audit)
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import tomllib

LOCK = Path(__file__).resolve().parents[1] / "uv.lock"


def batches(lock_text: str) -> list[list[str]]:
    """name==version lines for every PyPI package version in the lock, split so
    no batch names a package twice. The project itself and git or path sources
    are left out: there is nothing on PyPI to look them up by."""
    versions: dict[str, set[str]] = {}
    for p in tomllib.loads(lock_text).get("package", []):
        if "registry" not in p.get("source", {}):
            continue
        versions.setdefault(p["name"], set()).add(p["version"])
    out: list[list[str]] = []
    for name in sorted(versions):
        for i, version in enumerate(sorted(versions[name])):
            if len(out) <= i:
                out.append([])
            out[i].append(f"{name}=={version}")
    return out


def _pip_audit(requirements: str) -> int:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(requirements)
    try:
        return subprocess.run(["uvx", "pip-audit", "-r", f.name, "--no-deps", "--disable-pip",
                               "--progress-spinner", "off", "--vulnerability-service", "osv"]).returncode
    finally:
        Path(f.name).unlink(missing_ok=True)


def audit(lock_text: str, run: Callable[[str], int] = _pip_audit) -> int:
    """Audit every batch; non-zero when any batch reports a vulnerability or fails."""
    status = 0
    for batch in batches(lock_text):
        status = run("\n".join(batch) + "\n") or status
    return status


if __name__ == "__main__":
    sys.exit(audit(LOCK.read_text()))
