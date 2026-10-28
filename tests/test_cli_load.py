"""banko-ai load: the simulator headless in a terminal, a running app driven
from the command line, and the load customers prepared or cleared."""

from urllib.parse import urlparse

from click.testing import CliRunner

from banko_ai import load_cli as L
from banko_ai.cli import cli
from banko_ai.observability.metrics import Metrics


class FakeSim:
    def __init__(self, ok=100, errors=0, polls=3, error=None):
        self.metrics, self.started, self.stopped = Metrics(), None, False
        self._ok, self._errors, self._polls, self._error = ok, errors, polls, error

    def start(self, cfg):
        self.started = cfg
        for _ in range(self._ok):
            self.metrics.record("load", ms=50)
        for _ in range(self._errors):
            self.metrics.record("load", outcome="error", error_class="40001")
        return self.status()

    def status(self):
        self._polls -= 1
        state = "running" if self._polls > 0 and not self.stopped else "idle"
        return {"state": state, "users_ready": 0, "error": self._error, "config": None, "serving": {}}

    def stop(self, timeout=5.0):
        self.stopped = True
        return self.status()

    def totals(self):
        return {"ok": self._ok, "error": self._errors}


def _run(monkeypatch, sim, *args):
    monkeypatch.setattr(L, "_simulator", lambda: sim)
    monkeypatch.setattr(L, "_sleep", lambda seconds: None)
    return CliRunner().invoke(cli, ["load", "run", *args])


def test_run_prints_lines_and_a_summary(monkeypatch):
    sim = FakeSim()
    r = _run(monkeypatch, sim, "--rate", "40", "--users", "20", "--duration", "5s")
    assert r.exit_code == 0, r.output
    assert (sim.started.rate, sim.started.users, sim.started.duration_s) == (40, 20, 5)
    assert "ops/s" in r.output and "100 operations" in r.output and "Summary" in r.output


def test_run_exits_1_above_the_error_limit(monkeypatch):
    r = _run(monkeypatch, FakeSim(ok=90, errors=10), "--duration", "5s", "--max-error-rate", "0.01")
    assert r.exit_code == 1 and "error rate" in r.output.lower()


def test_a_run_that_cannot_prepare_says_why_and_exits_1(monkeypatch):
    sim = FakeSim(ok=0, polls=1, error="Load users could not be prepared: check that the database is reachable.")
    r = _run(monkeypatch, sim, "--duration", "5s")
    assert r.exit_code == 1 and "could not be prepared" in r.output


def test_bad_settings_are_usage_errors(monkeypatch):
    for args in (["--duration", "5x"], ["--rate", "0"], ["--profile", "custom"], ["--weights", "purchase=lots"]):
        r = _run(monkeypatch, FakeSim(), *args)
        assert r.exit_code == 2, (args, r.output)


def test_custom_weights_parse(monkeypatch):
    sim = FakeSim()
    r = _run(monkeypatch, sim, "--profile", "custom", "--weights", "purchase=60,search=40", "--duration", "1m")
    assert r.exit_code == 0, r.output
    assert sim.started.weights == {"purchase": 60, "search": 40} and sim.started.duration_s == 60


def test_ctrl_c_stops_the_run_and_still_summarizes(monkeypatch):
    sim = FakeSim(polls=100)

    def interrupt(seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(L, "_simulator", lambda: sim)
    monkeypatch.setattr(L, "_sleep", interrupt)
    r = CliRunner().invoke(cli, ["load", "run"])
    assert r.exit_code == 0, r.output
    assert sim.stopped and "Summary" in r.output


class _Resp:
    def __init__(self, status, body):
        self.status_code, self._body = status, body

    def json(self):
        return self._body


class FakeHttp:
    def __init__(self, replies=None):
        self.calls, self.replies = [], replies or {}

    def _answer(self, method, url, kw):
        self.calls.append((method, url, kw))
        return self.replies.get((method, urlparse(url).path), _Resp(200, {"state": "running"}))

    def post(self, url, **kw):
        return self._answer("POST", url, kw)

    def get(self, url, **kw):
        return self._answer("GET", url, kw)


def test_start_and_stop_drive_a_running_app(monkeypatch):
    http = FakeHttp()
    monkeypatch.setattr(L, "_http", lambda: http)
    r = CliRunner().invoke(cli, ["load", "start", "--url", "http://app:5000/", "--rate", "80", "--users", "30"])
    assert r.exit_code == 0, r.output
    assert http.calls[0][:2] == ("POST", "http://app:5000/login") and http.calls[0][2]["data"] == {"username": "maya"}
    assert http.calls[1][1] == "http://app:5000/api/v1/platform/load"
    assert http.calls[1][2]["json"] == {"action": "start", "rate": 80.0, "users": 30}
    r = CliRunner().invoke(cli, ["load", "stop", "--url", "http://app:5000", "--persona", "sam"])
    assert r.exit_code == 0 and http.calls[-1][2]["json"] == {"action": "stop"} and http.calls[-2][2]["data"] == {
        "username": "sam"}


def test_a_refused_remote_start_prints_the_reason_and_exits_1(monkeypatch):
    http = FakeHttp({("POST", "/api/v1/platform/load"): _Resp(409, {"error": "A load run is already going; stop it first",
                                                                   "code": "busy"})})
    monkeypatch.setattr(L, "_http", lambda: http)
    r = CliRunner().invoke(cli, ["load", "start", "--url", "http://app:5000"])
    assert r.exit_code == 1 and "already going" in r.output


def test_status_prints_the_state_and_the_numbers(monkeypatch):
    live = {"load": {"state": "running", "elapsed_s": 65.0, "users_ready": 100, "error": None,
                     "config": {"profile": "balanced", "users": 100, "rate": 50, "workers": 16}},
            "summary": {"ops_s": 48.2, "p50": 61.5, "p95": 140.2, "p99": 310.7, "errors": 0, "declined": 2,
                        "dropped": 0}}
    http = FakeHttp({("GET", "/api/v1/platform/live"): _Resp(200, live)})
    monkeypatch.setattr(L, "_http", lambda: http)
    r = CliRunner().invoke(cli, ["load", "status", "--url", "http://app:5000"])
    assert r.exit_code == 0 and "running" in r.output and "48.2 ops/s" in r.output and "p95 140 ms" in r.output


def test_users_prepare_and_clear(monkeypatch):
    calls = []
    monkeypatch.setattr(L, "_prepare", lambda n: calls.append(("prepare", n)) or [object()] * n)
    monkeypatch.setattr(L, "_clear", lambda: calls.append(("clear",)) or 7)
    first = CliRunner().invoke(cli, ["load", "users", "prepare", "25"])
    second = CliRunner().invoke(cli, ["load", "users", "clear", "--yes"])
    assert calls == [("prepare", 25), ("clear",)]
    assert "25 load customers" in first.output and "7 load customers" in second.output
