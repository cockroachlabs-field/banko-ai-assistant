"""Live smoke of a running Banko against whatever provider it serves: the
release check, run once per provider. It signs in as a persona and drives the
real API, so the model, the database and the checkpointer all answer for real.
Exit code 0 when every check passes.

    uv run python scripts/smoke.py --url http://localhost:5000
    # then restart the app and prove the conversation survived it:
    uv run python scripts/smoke.py --url http://localhost:5000 --resume <thread id printed by the first run>

Each run first posts a receipt with a fresh total. The new purchase moves
the persona's data version, so no answer is served from a cache an earlier
run (or another provider) filled; only the deliberate repeat is.

The coach check fires signals at the webhook, so it needs the app's
CDC_WEBHOOK_HMAC_SECRET and DATABASE_URL in this shell; without them it is
skipped and says so.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path

import requests

TIMEOUT = 240  # seconds per request: a local model can take a while on a cold start
SPANISH = (" el ", " la ", " en ", " de ", " los ", " las ", "gast", " del ")


def parse_sse(lines: Iterable[str]) -> list[tuple[str, dict]]:
    """(event, data) pairs from a Server-Sent Events stream."""
    out, event, data = [], None, []
    for raw in lines:
        line = raw.rstrip("\r")
        if not line:
            if event is not None:
                out.append((event, json.loads("\n".join(data)) if data else {}))
            event, data = None, []
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].strip())
    if event is not None:
        out.append((event, json.loads("\n".join(data)) if data else {}))
    return out


def reads_spanish(text: str) -> bool:
    padded = f" {text.lower()} "
    return sum(word in padded for word in SPANISH) >= 2


def receipt_png(total: str) -> bytes:
    """A clean receipt tesseract can read: merchant, items, total, date."""
    from PIL import Image, ImageDraw, ImageFont
    pastry = f"{float(total) - 5.50:.2f}"
    lines = ["BLUE BOTTLE COFFEE", "300 Webster St, Oakland CA", f"Date: {time.strftime('%Y-%m-%d')}", "",
             "Latte              5.50", f"Croissant         {pastry:>6}", "", f"TOTAL             {total:>6}",
             "VISA ****4242"]
    try:
        font = ImageFont.truetype("Arial.ttf", 40)
    except OSError:
        font = ImageFont.load_default(size=40)
    img = Image.new("RGB", (900, 120 + 60 * len(lines)), "white")
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        draw.text((60, 60 + 60 * i), line, fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


class Smoke:
    def __init__(self, url: str, persona: str, expect: str | None):
        self.url, self.persona, self.expect = url.rstrip("/"), persona, expect
        self.http = requests.Session()
        self.failures = 0
        self.user_id: str | None = None
        self.total = f"{random.randint(700, 1999) / 100:.2f}"  # a new purchase each run, never a duplicate

    # -- plumbing -------------------------------------------------------------------------------

    def check(self, name: str, fn: Callable[[], str]) -> None:
        t0 = time.perf_counter()
        try:
            detail = fn()
            status = "SKIP" if detail.startswith("skipped") else "PASS"
        except Exception as e:  # report and carry on: one broken check must not hide the others
            status, detail = "FAIL", f"{type(e).__name__}: {e}"
            self.failures += 1
        print(f"{status:4}  {name:30} {time.perf_counter() - t0:6.1f} s  {detail}", flush=True)

    def api(self, method: str, path: str, **kw) -> requests.Response:
        r = self.http.request(method, self.url + path, timeout=TIMEOUT, **kw)
        r.raise_for_status()
        return r

    def thread(self) -> str:
        return self.api("POST", "/api/v1/assistant/threads", json={}).json()["thread_id"]

    def ask(self, thread_id: str, message: str, language: str | None = None) -> dict:
        body = {"message": message, **({"language": language} if language else {})}
        r = self.http.post(f"{self.url}/api/v1/assistant/threads/{thread_id}/messages", json=body, stream=True,
                           timeout=TIMEOUT)
        r.raise_for_status()
        events = parse_sse(r.iter_lines(decode_unicode=True))
        errors = [d for e, d in events if e == "error"]
        if errors:
            raise AssertionError(f"error event: {errors[0].get('message')}")
        answers = [d for e, d in events if e == "answer"]
        if not answers:
            raise AssertionError(f"no answer event (got {[e for e, _ in events]})")
        return answers[0]

    @staticmethod
    def expect_route(answer: dict, route: str) -> None:
        if answer.get("route") != route:
            raise AssertionError(f"route {answer.get('route')!r}, expected {route!r}: {answer.get('answer', '')[:120]!r}")
        if not (answer.get("answer") or "").strip():
            raise AssertionError("empty answer")

    # -- checks ---------------------------------------------------------------------------------

    def ready(self) -> str:
        deadline = time.time() + 600
        while time.time() < deadline:
            try:
                if self.http.get(self.url + "/api/health/ready", timeout=5).status_code == 200:
                    return "app ready"
            except requests.RequestException:
                pass
            time.sleep(3)
        raise AssertionError("not ready after 10 minutes")

    def sign_in(self) -> str:
        self.http.post(self.url + "/login", data={"username": self.persona}, timeout=TIMEOUT)
        deadline = time.time() + 900
        while True:
            me = self.api("GET", "/api/v1/me").json()
            if me.get("username") != self.persona:
                raise AssertionError(f"signed in as {me.get('username')!r}, not {self.persona!r}")
            self.user_id = me["user_id"]
            if ((me.get("seeding") or {}).get("state")) != "running":
                return f"{self.persona} ({self.user_id[:8]}), history ready"
            if time.time() > deadline:
                raise AssertionError("history still being written after 15 minutes")
            time.sleep(5)

    def identity(self) -> str:
        ai = self.api("GET", "/api/v1/ai").json()
        shown = ai.get("shown") or ai
        label = f"{shown.get('name')} · {shown.get('model')}"
        if self.expect and self.expect.lower() not in json.dumps(ai).lower():
            raise AssertionError(f"serving {label}, expected {self.expect}")
        health = self.api("GET", "/api/health?refresh=1").json()  # the provider's own check, run now
        if health.get("ai_provider") != "connected":
            detail = ((health.get("components") or {}).get("ai_provider") or {}).get("detail") or ""
            raise AssertionError(f"{label} is not answering: {health.get('ai_provider')} {detail[:160]}")
        return f"{label}, answering"

    def totals(self) -> str:
        a = self.ask(self.thread(), "How much did I spend on groceries last month?")
        self.expect_route(a, "aggregate")
        if "$" not in a["answer"]:
            raise AssertionError(f"no dollar figure: {a['answer'][:120]!r}")
        return a["answer"].splitlines()[0][:100]

    def cache(self) -> str:
        a = self.ask(self.thread(), "How much did I spend on groceries last month?")
        trace = (a.get("meta") or {}).get("trace") or []
        if not (a.get("cached") or any(s.get("status") == "hit" for s in trace)):
            raise AssertionError("the repeat question hit no cache")
        return "repeat answered from the cache"

    def visits(self) -> str:
        items = self.api("GET", "/api/v1/transactions?limit=200").json()["items"]
        merchant = Counter(i["merchant"] for i in items if i.get("merchant")).most_common(1)[0][0]
        a = self.ask(self.thread(), f"did I go to {merchant}?")
        self.expect_route(a, "visits")
        if merchant.lower() not in a["answer"].lower():
            raise AssertionError(f"{merchant} not in {a['answer'][:120]!r}")
        return a["answer"].splitlines()[0][:100]

    def search_and_more(self) -> str:
        tid = self.thread()
        a = self.ask(tid, "show me my coffee purchases")
        self.expect_route(a, "plan")
        nxt = ((a.get("meta") or {}).get("search") or {}).get("next")
        if not nxt:
            return "answered; no second page to show"
        more = self.ask(tid, "show more")
        self.expect_route(more, "more")
        items = ((more.get("meta") or {}).get("search") or {}).get("items") or []
        if not items:
            raise AssertionError("show more returned no purchases")
        return f"answered, then {len(items)} more from the database"

    def spanish(self) -> str:
        a = self.ask(self.thread(), "How much did I spend on restaurants last month?", language="es-ES")
        if not reads_spanish(a.get("answer") or ""):
            raise AssertionError(f"not Spanish: {a.get('answer', '')[:120]!r}")
        return (a["answer"].splitlines()[0])[:100]

    def receipt(self) -> str:
        r = self.http.post(self.url + "/api/v1/receipts",
                           files={"receipt": ("smoke-receipt.png", receipt_png(self.total), "image/png")},
                           stream=True, timeout=TIMEOUT)
        r.raise_for_status()
        events = parse_sse(r.iter_lines(decode_unicode=True))
        errors = [d for e, d in events if e == "error"]
        if errors:
            raise AssertionError(f"error event: {errors[0].get('message')}")
        found = next((d for e, d in events if e == "receipt"), None)
        if not found:
            raise AssertionError(f"no receipt event (got {[e for e, _ in events]})")
        if found.get("status") not in ("posted", "duplicate") or not found.get("merchant"):
            raise AssertionError(f"receipt ended {found.get('status')!r} with merchant {found.get('merchant')!r}")
        tid = found["thread_id"]
        for question in ("what was the total on this receipt?", "is this receipt a duplicate?",
                         "which merchant was this receipt from?"):
            self.expect_route(self.ask(tid, question), "receipt")
        return f"{found['merchant']} ${found.get('amount')} {found['status']}, three follow-ups answered"

    def coach(self) -> str:
        if not (os.getenv("CDC_WEBHOOK_HMAC_SECRET") and os.getenv("DATABASE_URL")):
            return "skipped: needs CDC_WEBHOOK_HMAC_SECRET and DATABASE_URL in this shell"
        script = Path(__file__).resolve().parent / "coach" / "assert_nudges.py"
        out = subprocess.run([sys.executable, str(script), "--url", self.url + "/api/cdc/signals",
                              "--user-id", self.user_id or "",
                              "--timeout", "180"], capture_output=True, text=True)
        if out.returncode != 0:
            raise AssertionError((out.stdout + out.stderr).strip().splitlines()[-1][:200])
        if "[fallback]" in out.stdout:  # the coach's model call failed and a template stood in
            raise AssertionError("the coach fell back to a template nudge: its model call failed")
        return "a nudge for every signal, written by the model"

    def resume(self, thread_id: str) -> str:
        history = self.api("GET", f"/api/v1/assistant/threads/{thread_id}").json().get("messages") or []
        if len(history) < 2:
            raise AssertionError(f"the thread came back with {len(history)} messages")
        a = self.ask(thread_id, "and the month before?")
        if not (a.get("answer") or "").strip():
            raise AssertionError("empty answer")
        return f"{len(history)} messages kept, follow-up answered"

    def run(self, resume: str | None) -> int:
        self.check("ready", self.ready)
        self.check("sign in", self.sign_in)
        self.check("provider", self.identity)
        if resume:
            self.check("conversation after a restart", lambda: self.resume(resume))
            return 1 if self.failures else 0
        tid = None

        def dining() -> str:
            nonlocal tid
            tid = self.thread()
            a = self.ask(tid, "How much did I spend on dining last month?")
            self.expect_route(a, "aggregate")
            return a["answer"].splitlines()[0][:100]

        # The receipt goes first: its new purchase retires every cached answer.
        self.check("receipt and follow-ups", self.receipt)
        self.check("dining from SQL", dining)
        self.check("totals from SQL", self.totals)
        self.check("cache on a repeat", self.cache)
        self.check("visits from SQL", self.visits)
        self.check("search and show more", self.search_and_more)
        self.check("answer in Spanish", self.spanish)
        self.check("coach nudge", self.coach)
        print(f"\n{'FAILED' if self.failures else 'All checks passed'}."
              + (f" After restarting the app: --resume {tid}" if tid else ""))
        return 1 if self.failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://localhost:5000")
    ap.add_argument("--persona", default="maya")
    ap.add_argument("--expect", help="provider the app should be serving, e.g. watsonx")
    ap.add_argument("--resume", metavar="THREAD_ID", help="check a conversation from an earlier run after a restart")
    args = ap.parse_args()
    return Smoke(args.url, args.persona, args.expect).run(args.resume)


if __name__ == "__main__":
    sys.exit(main())
