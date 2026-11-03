"""Settings is the only reader of environment variables.

If a module reads os.getenv directly, its knob is invisible to
`banko-ai config`, missing from the README reference, and ignores the YAML
file. This test lists every offender with its line."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "banko_ai"
ALLOWED = {ROOT / "config" / "settings.py"}
ENV_READ = re.compile(r"\bos\.(getenv|environ)\b")


def test_every_environment_read_goes_through_settings():
    offenders = []
    for path in sorted(ROOT.rglob("*.py")):
        if path in ALLOWED:
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            code = line.split("#", 1)[0]
            if ENV_READ.search(code):
                offenders.append(f"{path.relative_to(ROOT.parent)}:{lineno}: {line.strip()}")
    assert not offenders, "Read these through get_config():\n" + "\n".join(offenders)
