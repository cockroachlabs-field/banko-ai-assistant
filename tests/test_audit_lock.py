"""The lock audit checks every version uv.lock pins, the ones locked for
other Python versions included, in batches pip-audit accepts: one version
per package per batch, the project itself and non-PyPI sources left out."""

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_lock.py"
_spec = importlib.util.spec_from_file_location("audit_lock", _SCRIPT)
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)

LOCK = '''
version = 1
requires-python = ">=3.11"

[[package]]
name = "banko-ai-assistant"
version = "2.0.0"
source = { editable = "." }

[[package]]
name = "networkx"
version = "3.4.2"
source = { registry = "https://pypi.org/simple" }
resolution-markers = ["python_full_version < '3.12'"]

[[package]]
name = "networkx"
version = "3.7"
source = { registry = "https://pypi.org/simple" }
resolution-markers = ["python_full_version >= '3.12'"]

[[package]]
name = "pillow"
version = "13.0.0"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "somelib"
version = "0.1.0"
source = { git = "https://github.com/example/somelib?rev=abc#abc" }
'''


def test_every_locked_version_is_audited_once_and_no_batch_names_a_package_twice():
    batches = audit.batches(LOCK)
    assert sorted(line for b in batches for line in b) == ["networkx==3.4.2", "networkx==3.7", "pillow==13.0.0"]
    for b in batches:
        names = [line.split("==")[0] for line in b]
        assert len(names) == len(set(names)), b


def test_a_failing_batch_fails_the_audit_and_every_batch_still_runs():
    seen = []

    def run(requirements):
        seen.append(requirements)
        return 1 if "networkx==3.7" in requirements else 0

    assert audit.audit(LOCK, run=run) == 1
    assert len(seen) == 2
