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
    # CPU time, not wall time: the import does no I/O worth waiting on, and
    # a loaded machine (the full suite, a busy laptop) stretches wall time
    # without the import doing any more work.
    code = (
        "import resource, sys, time; t = time.time(); import banko_ai.web.app, banko_ai.cli; "
        "u = resource.getrusage(resource.RUSAGE_SELF); "
        "print(round(u.ru_utime + u.ru_stime, 2), round(time.time() - t, 2)); "
        "print('google.genai' in sys.modules, 'openai' in sys.modules, 'sentence_transformers' in sys.modules)"
    )
    env = dict(os.environ, DATABASE_URL="postgresql://root@203.0.113.1:26257/defaultdb?sslmode=disable&connect_timeout=2")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=60)
    assert out.returncode == 0, out.stderr
    timing, loaded = out.stdout.strip().splitlines()[-2:]
    cpu, wall = (float(x) for x in timing.split())
    print(f"import took {cpu}s of CPU, {wall}s wall")
    assert cpu < 5.0
    assert wall < 30.0
    assert loaded == "False False False"


def test_the_assistant_api_loads_no_langgraph_before_the_server_listens():
    """create_app registers the assistant blueprint before the server
    listens. LangGraph costs about half a second and loads on the first turn."""
    code = ("import sys; import banko_ai.web.app, banko_ai.web.assistant_api; "
            "print('langgraph' in sys.modules, 'banko_ai.assistant.graph' in sys.modules)")
    env = dict(os.environ, DATABASE_URL="postgresql://root@203.0.113.1:26257/defaultdb?sslmode=disable&connect_timeout=2")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "False False"


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
