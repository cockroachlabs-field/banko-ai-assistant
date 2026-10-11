"""Boot does no slow work up front."""

import os
import subprocess
import sys
import threading
import time
import urllib.request

from flask import Flask

from banko_ai.web import server


def test_importing_the_app_touches_no_database_and_no_provider_sdk():
    code = (
        "import sys, time; t = time.time(); import banko_ai.web.app, banko_ai.cli; "
        "print(round(time.time() - t, 2)); "
        "print('google.genai' in sys.modules, 'openai' in sys.modules, 'sentence_transformers' in sys.modules)"
    )
    env = dict(os.environ, DATABASE_URL="postgresql://root@203.0.113.1:26257/defaultdb?sslmode=disable&connect_timeout=2")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=60)
    assert out.returncode == 0, out.stderr
    seconds, loaded = out.stdout.strip().splitlines()[-2:]
    print(f"import took {seconds}s")
    assert float(seconds) < 5.0
    assert loaded == "False False False"


def test_server_starts_without_a_reverse_dns_lookup(monkeypatch):
    import socket

    def boom(*args, **kwargs):
        raise AssertionError("socket.getfqdn was called")

    monkeypatch.setattr(socket, "getfqdn", boom)
    app = Flask("boot-test")
    app.add_url_rule("/ping", "ping", lambda: "pong")
    sock = server.bind("127.0.0.1", 0)
    srv = server.make_app_server(app, sock)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        port = sock.getsockname()[1]
        body = urllib.request.urlopen(f"http://127.0.0.1:{port}/ping", timeout=5).read()
    finally:
        srv.shutdown()
        sock.close()
    assert body == b"pong"


def test_display_urls_never_need_dns():
    urls = server.display_urls("0.0.0.0", 5000)
    assert urls[0] == "http://127.0.0.1:5000"
