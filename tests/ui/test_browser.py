"""Browser pass over every page, as every persona, in Daylight and Midnight.

    uv sync --all-extras && uv run playwright install chromium
    BANKO_UI_TESTS=1 DATABASE_URL="postgresql://root@localhost:26258/defaultdb?sslmode=disable" \
        uv run pytest tests/ui -q -p no:cacheprovider

Boots the app on port 5073 with AI_SERVICE=ollama. No page may log a console
error; every page is screenshotted to artifacts/ui/."""

import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.getenv("BANKO_UI_TESTS") != "1", reason="set BANKO_UI_TESTS=1 to run the browser pass")
playwright = pytest.importorskip("playwright.sync_api")

PORT = 5073
BASE = f"http://127.0.0.1:{PORT}"
OUT = Path(__file__).resolve().parents[2] / "artifacts" / "ui"
PAGES = ["/", "/accounts", "/transactions", "/cashflow", "/budget", "/recurring", "/goals", "/assistant",
         "/coach", "/agents", "/settings"]
PERSONAS = ["maya", "sam", "riley"]


@pytest.fixture(scope="module")
def server():
    env = {**os.environ, "AI_SERVICE": "ollama", "COACH_KAFKA_ENABLED": "false",
           "CDC_WEBHOOK_HMAC_SECRET": "dev-only-secret", "PYTHONUNBUFFERED": "1"}
    assert ":26258/" in env.get("DATABASE_URL", ""), "the browser pass runs against the test database only"
    # `python -m banko_ai` ignores its arguments, so call the click group directly.
    proc = subprocess.Popen([sys.executable, "-c", "from banko_ai.cli import cli; cli()",
                             "run", "--host", "127.0.0.1", "--port", str(PORT)],
                            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 300
    while time.time() < deadline:
        try:
            if urllib.request.urlopen(BASE + "/api/health/ready", timeout=2).status == 200:
                break
        except Exception:
            time.sleep(2)
    else:
        proc.kill()
        pytest.fail("the app never became ready on port 5073")
    yield BASE
    proc.terminate()
    proc.wait(timeout=20)


@pytest.fixture(scope="module")
def browser():
    with playwright.sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


def _page(browser, theme):
    ctx = browser.new_context(color_scheme=theme, viewport={"width": 1360, "height": 900})
    page = ctx.new_page()
    errors = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: errors.append(str(e)))
    return ctx, page, errors


def _login(page, base, persona):
    page.goto(base + "/login")
    page.locator(f'form:has(input[name="username"][value="{persona}"]) button').click()
    page.wait_for_url(base + "/")


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("persona", PERSONAS)
def test_every_page_renders_without_console_errors(server, browser, persona, theme):
    OUT.mkdir(parents=True, exist_ok=True)
    ctx, page, errors = _page(browser, theme)
    try:
        _login(page, server, persona)
        for path in PAGES:
            page.goto(server + path)
            page.wait_for_load_state("networkidle")
            assert page.locator("[data-page]").count() == 1, path
            page.screenshot(path=str(OUT / f"{persona}-{theme}-{path.strip('/') or 'dashboard'}.png"), full_page=True)
        assert not errors, errors
    finally:
        ctx.close()


def test_stream_parser_handles_error_and_abort(server, browser):
    ctx, page, errors = _page(browser, "light")
    try:
        _login(page, server, "maya")
        parsed = page.evaluate("""() => {
            const a = Banko.parseSSE('event: token\\ndata: {"text": "Hi"}\\n\\nevent: error\\ndata: {"message": "x"}\\n\\nevent: tok');
            return { kinds: a.events.map(e => e.event), rest: a.rest };
        }""")
        assert parsed == {"kinds": ["token", "error"], "rest": "event: tok"}

        page.route("**/api/v1/assistant/threads/failing/messages", lambda route: route.fulfill(
            status=500, content_type="application/json",
            body='{"error": "Something went wrong on our side. Please try again."}'))
        failed = page.evaluate("""async () => {
            try { await Banko.stream("/api/v1/assistant/threads/failing/messages", { body: { message: "hi" }, onEvent: () => {} }); return "finished"; }
            catch (e) { return e.message; }
        }""")
        assert failed == "Something went wrong on our side. Please try again."

        page.route("**/api/v1/assistant/threads/hanging/messages", lambda route: None)  # never answers
        stopped = page.evaluate("""async () => {
            const c = new AbortController();
            setTimeout(() => c.abort(), 300);
            try { await Banko.stream("/api/v1/assistant/threads/hanging/messages", { body: { message: "hi" }, signal: c.signal, onEvent: () => {} }); return "finished"; }
            catch (e) { return e.name; }
        }""")
        assert stopped == "AbortError"
    finally:
        ctx.close()


def test_assistant_markup_is_sanitized(server, browser):
    ctx, page, errors = _page(browser, "light")
    try:
        _login(page, server, "maya")
        html = page.evaluate("""() => DOMPurify.sanitize(marked.parse('<img src=x onerror=alert(1)><script>alert(2)</script>**ok**'))""")
        assert "onerror" not in html and "<script" not in html and "<strong>ok</strong>" in html
        plain = page.evaluate("""() => Banko.stripCitations('Lunch [txn:abc] and [ txn : 1234 ] done')""")
        assert plain == "Lunch and done"
    finally:
        ctx.close()


def test_expired_session_goes_to_login(server, browser):
    # Not maya: a request without a session acts as maya, the default persona.
    ctx, page, errors = _page(browser, "light")
    try:
        _login(page, server, "sam")
        page.goto(server + "/transactions")
        page.wait_for_load_state("networkidle")
        ctx.clear_cookies()
        page.locator("button.b-f").first.click()
        page.wait_for_url(server + "/login", timeout=10000)
    finally:
        ctx.close()


def test_a_fresh_signup_shows_progress_not_zeros(server, browser):
    ctx, page, errors = _page(browser, "light")
    try:
        page.goto(server + "/login")
        page.fill('#name-form input[name="username"]', f"ui-{int(time.time())}")
        page.locator("#name-form button[type=submit]").click()
        page.wait_for_selector("#signup-section", state="visible")
        page.locator("#name-form button[type=submit]").click()
        page.wait_for_url(server + "/")
        page.wait_for_selector("text=Setting up your accounts", state="visible", timeout=15000)
        assert page.locator('section[x-data="dashboardPage"] > .b-grid').first.is_hidden()
        assert not errors, errors
    finally:
        ctx.close()
