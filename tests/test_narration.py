"""The demo console narrates the assistant's work, in the wording it has
always used: aggregations, vector searches, the answer cache and receipts."""

import os
import re
import uuid
from datetime import date

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from PIL import Image
from sqlalchemy import text

from banko_ai.assistant import graph as G
from banko_ai.assistant.tools import ToolContext, run_tool
from banko_ai.config.settings import reset_config
from banko_ai.db.engine import get_engine
from banko_ai.services import receipts
from banko_ai.services.seeding import seed_user
from tests._chat import ScriptedChat
from tests._money import drop_user, make_user

pytestmark = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")

AGG_QUESTION = "how much did I spend on restaurants in the last 30 days"


@pytest.fixture(scope="module")
def member():
    uid = make_user("diner")
    seed_user(uid, months=1, today=date.today())
    yield uid
    with get_engine().begin() as c:
        for t in ("chat_threads", "receipts", "memories", "assistant_actions", "answer_cache", "search_cache"):
            c.execute(text(f"DELETE FROM {t} WHERE user_id = CAST(:u AS UUID)"), {"u": uid})
    drop_user(uid)


@pytest.fixture
def graph(monkeypatch):
    monkeypatch.setenv("AI_SERVICE", "ollama")
    reset_config()
    return G.build_graph(InMemorySaver())


def _turn(member, graph, message, chat, language="en-US"):
    return G.run_turn(member, str(uuid.uuid4()), message, language_code=language, chat=chat, graph=graph)


def test_an_aggregation_turn_narrates(member, graph, capsys):
    _turn(member, graph, AGG_QUESTION, ScriptedChat("- steady"), language="es-ES")
    out = capsys.readouterr().out
    assert "\n🧮 AGGREGATION (agentic routing):\n" in out
    assert re.search(r"1\. Intent: sum on 'restaurants' \| window \d{4}-\d\d-\d\d to \d{4}-\d\d-\d\d\n", out)
    assert re.search(r"2\. SQL answered in \d+ms: \$[\d,]+\.\d\d across \d+ transactions \(category: Restaurant\)\n",
                     out)
    assert "   Figures computed by CockroachDB; the model never does the arithmetic\n" in out
    assert "3. ❌ Narrative cache MISS, generating\n" in out
    assert "   Insights narrated by ollama around the pinned figures\n" in out
    assert "   Rendered in Spanish, figures preserved\n" in out
    assert "Region-pruned" not in out  # single-region database: no region applies

    _turn(member, graph, AGG_QUESTION, ScriptedChat("- steady"), language="es-ES")
    again = capsys.readouterr().out
    assert re.search(r"3\. ✅ Narrative cache HIT \(est\. \d+ tokens saved\)\n", again)
    assert "Insights narrated" not in again and "Rendered in" not in again


def test_a_plan_turn_with_a_search_narrates(member, graph, capsys):
    plan = '{"steps": [{"tool": "search_transactions", "args": {"query": "restaurant dinners", "limit": 5}}]}'
    _turn(member, graph, "show me my restaurant dinners", ScriptedChat(plan, '{"steps": []}', "You had a few dinners."))
    out = capsys.readouterr().out
    assert "❌ Answer cache MISS\n" in out
    assert "\n🔍 VECTOR SEARCH (with caching):\n1. Query: 'restaurant dinners' | Limit: 5\n" in out
    assert "2. ❌ Vector search cache MISS, querying database\n" in out
    assert re.search(r"3\. Database query returned \d+ expense records in [\d.]+ms\n", out)
    assert "Using ollama for response generation in English\n" in out
    assert out.index("Answer cache MISS") < out.index("VECTOR SEARCH") < out.index("Using ollama")

    run_tool("search_transactions", {"query": "restaurant dinners", "limit": 5}, ToolContext(user_id=member))
    assert re.search(r"2\. ✅ Vector search cache HIT! Found \d+ cached results\n", capsys.readouterr().out)

    _turn(member, graph, "show me my restaurant dinners", ScriptedChat("not called"))
    assert re.search(r"✅ Answer cache HIT \(est\. \d+ tokens saved\)\n", capsys.readouterr().out)


def test_a_receipt_run_narrates(member, tmp_path, capsys):
    fields = {"merchant": "Maple Hardware", "amount": 23.10, "date": str(date.today()), "category": "Shopping",
              "items": ["Hinges"], "payment_method": "debit card"}
    hooks = receipts.ReceiptHooks(ocr=lambda p: "MAPLE HARDWARE 23.10", extract=lambda t: dict(fields),
                                  fraud=lambda u, e: "No issues detected", budget=lambda u: "Within budget")
    path = tmp_path / "banko_receipt_upload.png"
    Image.new("RGB", (300, 400), "white").save(path)
    final = list(receipts.process_receipt(member, str(path), "maple.png", hooks=hooks))[-1]
    out = capsys.readouterr().out
    assert final["type"] == "receipt"
    assert "📄 Receipt uploaded: maple.png\n" in out
    assert "📊 Extracted fields: {" in out and "'Maple Hardware'" in out
    assert f"💰 Expense posted through the ledger: {final['expense_id']}\n" in out
    assert "🕵️  Running fraud check...\n   No issues detected\n" in out
    assert "📊 Running budget check...\n   Within budget\n" in out
    assert str(tmp_path) not in out

    junk = {"merchant": "store name", "amount": 0.0, "date": "YYYY-MM-DD", "category": "food|transport"}
    bad = receipts.ReceiptHooks(ocr=lambda p: "x y z", extract=lambda t: dict(junk),
                                fraud=lambda u, e: "", budget=lambda u: "")
    list(receipts.process_receipt(member, str(path), "junk.png", hooks=bad))
    assert f"❌ {receipts.TEMPLATE_ECHO}\n" in capsys.readouterr().out
