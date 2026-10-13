"""Deterministic routing: no model call decides where a message goes."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from ..utils.intent_classifier import (
    AggregationIntent,
    classify_aggregation,
    extract_window,
    is_financial_query,
)

_REMEMBER = re.compile(r"^\s*(?:please\s+)?remember(?:\s+that)?\s+(?!to\s)(?!that\s*$)(.{3,})$", re.I)
_FORGET = re.compile(r"^\s*(?:please\s+)?forget(?:\s+(?:that|about))?\s+(.{3,})$", re.I)
_RECEIPT_NOUN = re.compile(
    r"\b(this|that|the)\s+(receipt|purchase|bill|order|charge|transaction|total|item|items)\b"
    r"|\breceipt\b", re.I)
_PRONOUN = re.compile(r"\b(it|its)\b", re.I)
_THIS_THAT_BEFORE_TIME = re.compile(
    r"\b(this|that)\s+(day|days|week|weekend|month|quarter|year|morning|afternoon|evening|time|period)\b", re.I)
_WORD = re.compile(r"[a-z]{4,}")
_STOP = {"with", "from", "this", "that", "what", "much", "many", "have", "your", "about"}
_ITEM_WITH_DETERMINER = re.compile(r"\b(the|that|this|my)\s+([a-z]{4,})", re.I)


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


def _merchant_words(receipt: dict) -> set[str]:
    merchant = str(receipt.get("merchant") or "")
    return {w for w in _WORD.findall(merchant.lower().replace("'", "")) if w not in _STOP}


def _item_words(receipt: dict) -> set[str]:
    items = receipt.get("items") or []
    text = " ".join([str(i) for i in items])
    return {w for w in _WORD.findall(text.lower().replace("'", "")) if w not in _STOP}


def _about_receipt(message: str, receipt: dict) -> bool:
    # Pronoun references always count
    if _PRONOUN.search(message):
        return True
    # "this/that" before time words don't count as references
    if _THIS_THAT_BEFORE_TIME.search(message):
        # But if there's also a receipt noun, it does count
        if not _RECEIPT_NOUN.search(message):
            return False
    # Receipt noun references
    if _RECEIPT_NOUN.search(message):
        return True
    # Merchant words count bare
    merchant_words = _merchant_words(receipt)
    message_words = set(_WORD.findall(message.lower().replace("'", "")))
    if message_words & merchant_words:
        return True
    # Item words count only after determiners, but not if part of an idiom (word + "and")
    item_words = _item_words(receipt)
    msg_lower = message.lower()
    for m in _ITEM_WITH_DETERMINER.finditer(msg_lower):
        word = m.group(2)
        if word in item_words:
            # Check if this is part of an idiom like "bread and butter"
            end_pos = m.end()
            if end_pos < len(msg_lower) and msg_lower[end_pos:end_pos+5].startswith(" and "):
                continue
            return True
    return False


def route(message: str, *, active_receipt: dict | None = None, context: dict | None = None,
          today: date | None = None) -> Route:
    text = (message or "").strip()
    if (m := _REMEMBER.match(text)):
        return Route("remember", "memory command", payload=m.group(1).strip().rstrip("."))
    if (m := _FORGET.match(text)):
        return Route("forget", "memory command", payload=m.group(1).strip().rstrip("."))
    intent = classify_aggregation(text, today=today)
    if active_receipt and _about_receipt(text, active_receipt):
        # Receipt-noun references always win
        if _RECEIPT_NOUN.search(text):
            return Route("receipt", "follow-up about the active receipt")
        # An aggregation with its own explicit subject or window is a new question
        has_window = extract_window(text, today=today) is not None
        if intent and (intent.subject or has_window):
            return Route("aggregate", f"{intent.operation} answered by SQL", intent=intent)
        # Otherwise it's about the receipt
        return Route("receipt", "follow-up about the active receipt")
    if context:
        return Route("plan", f"opened with {context.get('kind', 'context')}")
    if not is_financial_query(text):
        return Route("offtopic", "not a money question")
    if intent is not None:
        return Route("aggregate", f"{intent.operation} answered by SQL", intent=intent)
    return Route("plan", "open question")
