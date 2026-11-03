"""Serve the app for `banko-ai run`.

Werkzeug's dev server binds through http.server.HTTPServer.server_bind,
which calls socket.getfqdn() on the bind address. On macOS that reverse
lookup of 0.0.0.0 times out after 5 seconds, on every boot. Binding the
socket here and handing werkzeug the file descriptor skips server_bind
entirely. Docker runs gunicorn and never comes through this module.
"""

from __future__ import annotations

import socket
from collections.abc import Callable

from werkzeug.serving import BaseWSGIServer, make_server


def bind(host: str, port: int) -> socket.socket:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(128)
    sock.set_inheritable(True)
    return sock


def make_app_server(app, sock: socket.socket) -> BaseWSGIServer:
    host, port = sock.getsockname()[:2]
    return make_server(host, port, app, threaded=True, fd=sock.fileno())


def _lan_ip() -> str | None:
    """The outbound interface address: connect a UDP socket (no packet is
    sent and no DNS is involved) and read its local address."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            probe.connect(("10.255.255.255", 1))
            return str(probe.getsockname()[0])
        except OSError:
            return None


def display_urls(host: str, port: int) -> list[str]:
    if host in ("0.0.0.0", "::"):
        urls = [f"http://127.0.0.1:{port}"]
        lan = _lan_ip()
        if lan and lan != "127.0.0.1":
            urls.append(f"http://{lan}:{port}")
        return urls
    return [f"http://{host}:{port}"]


def serve(app, host: str, port: int, *, on_ready: Callable[[list[str]], None] | None = None) -> None:
    sock = bind(host, port)
    try:
        server = make_app_server(app, sock)
        if on_ready:
            on_ready(display_urls(host, sock.getsockname()[1]))
        server.serve_forever()
    finally:
        sock.close()
