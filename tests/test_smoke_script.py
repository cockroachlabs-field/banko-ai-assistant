"""The release smoke's own logic: reading the assistant's event stream and
telling a Spanish answer from an English one."""

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "smoke.py"
_spec = importlib.util.spec_from_file_location("smoke", _SCRIPT)
smoke = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(smoke)


def test_the_event_stream_parses_into_events_with_their_data():
    lines = ["event: step", 'data: {"label": "Intent routing"}', "", "event: answer",
             'data: {"answer": "You spent $42.10", "route": "aggregate"}', "", "event: done", "data: {}"]
    assert smoke.parse_sse(lines) == [("step", {"label": "Intent routing"}),
                                      ("answer", {"answer": "You spent $42.10", "route": "aggregate"}),
                                      ("done", {})]


def test_spanish_is_told_apart_from_english():
    assert smoke.reads_spanish("Gastaste **$312.40** en restaurantes el mes pasado, en 9 compras.")
    assert not smoke.reads_spanish("You spent **$312.40** on restaurants last month, across 9 purchases.")


def test_the_generated_receipt_is_a_png():
    assert smoke.receipt_png("12.34")[:8] == b"\x89PNG\r\n\x1a\n"
