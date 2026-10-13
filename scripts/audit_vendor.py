"""Check the vendored browser libraries against the OSV vulnerability database.

    uv run --no-project --with truststore python scripts/audit_vendor.py

Reads banko_ai/static/vendor/manifest.json and asks api.osv.dev about each
npm package at its vendored version. Dependabot cannot see these files, so
this is the only thing watching them. Needs the network; `make audit` runs it."""

from __future__ import annotations

import json
import ssl
import sys
import urllib.request
from pathlib import Path

MANIFEST = Path(__file__).resolve().parents[1] / "banko_ai" / "static" / "vendor" / "manifest.json"
OSV_BATCH = "https://api.osv.dev/v1/querybatch"
NOT_ON_NPM = {"inter"}  # the font comes from its GitHub releases


def queries(entries: list[dict]) -> list[dict]:
    return [{"package": {"ecosystem": "npm", "name": e["name"]}, "version": e["version"]}
            for e in entries if e["name"] not in NOT_ON_NPM]


def _context() -> ssl.SSLContext:
    try:  # the OS trust store: python.org builds on macOS ship without CA certificates
        import truststore
        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except ImportError:
        return ssl.create_default_context()


def _post(body: dict) -> dict:
    req = urllib.request.Request(OSV_BATCH, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30, context=_context()) as resp:
        return json.load(resp)


def findings(entries: list[dict], post=None) -> list[str]:
    asked = queries(entries)
    if not asked:
        return []
    results = (post or _post)({"queries": asked})["results"]
    return [f'{q["package"]["name"]} {q["version"]}: {v["id"]}'
            for q, r in zip(asked, results) for v in r.get("vulns") or []]


def main() -> int:
    entries = json.loads(MANIFEST.read_text())
    found = findings(entries)
    for line in found:
        print(line)
    print(f"{len(found)} known vulnerabilities in {len(queries(entries))} vendored npm packages")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
