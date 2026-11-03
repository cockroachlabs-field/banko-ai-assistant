"""The vendored-library audit asks OSV about every npm package in the
manifest and reports each advisory it gets back."""

import importlib.util
import json
import urllib.error
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_vendor.py"
_spec = importlib.util.spec_from_file_location("audit_vendor", _SCRIPT)
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)

ENTRIES = [{"name": "dompurify", "version": "3.0.0"}, {"name": "inter", "version": "4.1"},
           {"name": "alpinejs", "version": "3.17.4"}]


def test_the_font_is_not_looked_up_on_npm():
    assert [q["package"]["name"] for q in audit.queries(ENTRIES)] == ["dompurify", "alpinejs"]


def test_each_advisory_names_the_package_and_version():
    def fake_post(body):
        assert body["queries"][0] == {"package": {"ecosystem": "npm", "name": "dompurify"}, "version": "3.0.0"}
        return {"results": [{"vulns": [{"id": "GHSA-0000-1111-2222"}]}, {}]}

    assert audit.findings(ENTRIES, post=fake_post) == ["dompurify 3.0.0: GHSA-0000-1111-2222"]


def test_every_vendored_npm_package_is_audited():
    names = {q["package"]["name"] for q in audit.queries(json.loads(audit.MANIFEST.read_text()))}
    assert names == {"alpinejs", "chart.js", "socket.io-client", "marked", "dompurify", "lucide-static"}


def test_network_failures_exit_cleanly(monkeypatch, capsys):
    def failing_post(body):
        raise urllib.error.URLError("Network unreachable")

    monkeypatch.setattr(audit, "_post", failing_post)
    assert audit.main() == 1
    captured = capsys.readouterr()
    assert "Could not reach the OSV database" in captured.err
    assert "vendored-library audit needs the network" in captured.err
    assert "Traceback" not in captured.err
