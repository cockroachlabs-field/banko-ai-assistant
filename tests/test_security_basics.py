"""Small security fixes: no secrets on open endpoints, bounded uploads,
temp files cleaned up, sane cookies, same-origin sockets, no account ids."""

import io
import os
import re
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "banko_ai"


def test_aws_provider_prints_no_account_ids_or_internal_profiles():
    source = (ROOT / "ai_providers" / "aws_provider.py").read_text()
    assert not re.search(r"\b\d{12}\b", source)
    assert "get_caller_identity" not in source
    assert "CRLRevenue" not in source


needs_db = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    from banko_ai.web.app import create_app
    from banko_ai.web.auth import UserManager
    app = create_app()
    app.config["TESTING"] = True
    c = app.test_client()
    app.extensions["banko_readiness"].wait_for("schema", 60)
    # The warm-up backfills personas asynchronously; the cookie test signs in as maya now.
    UserManager(os.environ["DATABASE_URL"]).backfill_personas()
    return c


@needs_db
def test_debug_endpoints_are_gone(client):
    for path in ("/ai-status", "/diagnostics/watsonx", "/test-ai-connection"):
        # Deleted routes return 404 or redirect (302) to login, either confirms they're gone
        status = client.get(path).status_code
        assert status in (302, 404), f"{path} returned {status}"


@needs_db
def test_oversized_uploads_are_refused(client):
    data = {"receipt": (io.BytesIO(b"x" * (2 * 1024 * 1024)), "big.png")}
    response = client.post("/api/upload-receipt", data=data, content_type="multipart/form-data")
    assert response.status_code == 413
    assert response.get_json()["success"] is False


@needs_db
def test_unsupported_upload_types_are_refused(client):
    data = {"receipt": (io.BytesIO(b"MZ"), "tool.exe")}
    response = client.post("/api/upload-receipt", data=data, content_type="multipart/form-data")
    assert response.status_code == 400


@needs_db
def test_receipt_temp_files_are_removed(client, monkeypatch):
    import banko_ai.agents.receipt_agent as receipt_agent

    def explode(self, *args, **kwargs):
        raise RuntimeError("agent down")

    monkeypatch.setattr(receipt_agent.ReceiptAgent, "process_document", explode)
    before = set(Path(tempfile.gettempdir()).glob("banko_receipt_*"))
    data = {"receipt": (io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"0" * 64), "r.png")}
    client.post("/api/upload-receipt", data=data, content_type="multipart/form-data")
    after = set(Path(tempfile.gettempdir()).glob("banko_receipt_*"))
    assert after <= before


@needs_db
def test_session_cookie_flags(client):
    response = client.post("/login", data={"username": "maya"})
    cookie = response.headers.get("Set-Cookie", "")
    assert "HttpOnly" in cookie
    assert "SameSite=Lax" in cookie


@needs_db
def test_socketio_allows_same_origin_only_by_default(client):
    server = client.application.socketio.server
    assert server.eio.cors_allowed_origins != "*"


@needs_db
def test_receipt_temp_file_cleaned_even_when_save_fails(client, monkeypatch):
    """Temp file cleanup is registered before file.save, so a failed save doesn't leak."""
    import tempfile
    from pathlib import Path

    def fail_save(self, *args, **kwargs):
        raise OSError("disk full")

    from werkzeug.datastructures import FileStorage
    monkeypatch.setattr(FileStorage, "save", fail_save)
    before = set(Path(tempfile.gettempdir()).glob("banko_receipt_*"))
    data = {"receipt": (io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"0" * 64), "r.png")}
    client.post("/api/upload-receipt", data=data, content_type="multipart/form-data")
    after = set(Path(tempfile.gettempdir()).glob("banko_receipt_*"))
    assert after <= before


@needs_db
def test_413_on_non_receipt_paths_uses_generic_message(client):
    """The CDC webhook shares MAX_CONTENT_LENGTH; 413 there should be generic."""
    data = b"x" * (2 * 1024 * 1024)
    response = client.post("/api/cdc/signals", data=data, content_type="application/json",
                          headers={"X-Banko-Signature": "fake"})
    assert response.status_code == 413
    json_response = response.get_json()
    assert json_response["success"] is False
    assert "Request body too large" in json_response["error"]
    assert "Receipt" not in json_response["error"]
