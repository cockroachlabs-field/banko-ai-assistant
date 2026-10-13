"""Deterministic routing: no model call decides where a message goes."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from ..utils.intent_classifier import AggregationIntent, classify_aggregation, is_financial_query

_REMEMBER = re.compile(r"^\s*(?:please\s+)?remember(?:\s+that)?\s+(?!that\s*$)(.{3,})$", re.I)
_FORGET = re.compile(r"^\s*(?:please\s+)?forget(?:\s+(?:that|about))?\s+(.{3,})$", re.I)
_REFERENCE = re.compile(
    r"\b(this|that|the)\s+(receipt|purchase|bill|order|charge|transaction|one|total|item|items)\b"
    r"|\b(it|its|this|that)\b|\breceipt\b", re.I)
_WORD = re.compile(r"[a-z]{4,}")
_STOP = {"with", "from", "this", "that", "what", "much", "many", "have", "your", "about"}


@dataclass(frozen=True)
class Route:
    kind: str
    detail: str = ""
    payload: str | None = None
    intent: AggregationIntent | None = None

    def as_dict(self) -> dict:
        out: dict = {"kind": self.kind, "detail": self.detail, "payload": self.payload, "intent": None}
        if self.intent is not None:
            out["intent"] = {"operation": self.intent.operation, "subject": self.intent.subject,
                             "window_start": str(self.intent.window_start),
                             "window_end": str(self.intent.window_end)}
        return out


def _receipt_words(receipt: dict) -> set[str]:
    text = " ".join([str(receipt.get("merchant") or "")] + [str(i) for i in receipt.get("items") or []])
    return {w for w in _WORD.findall(text.lower().replace("'", "")) if w not in _STOP}


def _about_receipt(message: str, receipt: dict) -> bool:
    if _REFERENCE.search(message):
        return True
    words = set(_WORD.findall(message.lower().replace("'", "")))
    return bool(words & _receipt_words(receipt))


def route(message: str, *, active_receipt: dict | None = None, context: dict | None = None,
          today: date | None = None) -> Route:
    text = (message or "").strip()
    if (m := _REMEMBER.match(text)):
        return Route("remember", "memory command", payload=m.group(1).strip().rstrip("."))
    if (m := _FORGET.match(text)):
        return Route("forget", "memory command", payload=m.group(1).strip().rstrip("."))
    intent = classify_aggregation(text, today=today)
    if active_receipt and _about_receipt(text, active_receipt):
        # An aggregation with its own explicit subject and window ("restaurants in the
        # last 60 days") is a new question, not a follow-up about the receipt.
        if intent is None or not intent.subject or _REFERENCE.search(text):
            return Route("receipt", "follow-up about the active receipt")
    if context:
        return Route("plan", f"opened with {context.get('kind', 'context')}")
    if not is_financial_query(text):
        return Route("offtopic", "not a money question")
    if intent is not None:
        return Route("aggregate", f"{intent.operation} answered by SQL", intent=intent)
    return Route("plan", "open question")
