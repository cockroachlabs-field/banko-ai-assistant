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
_NOUNS = r"(?:receipt|purchase|bill|order|charge|transaction|total|item|items)"
# Strong: "this/that <noun>" and the word receipt always mean the receipt.
_STRONG_REF = re.compile(rf"\b(?:this|that)\s+{_NOUNS}\b|\breceipt\b", re.I)
# Weak: "the total" can mean the receipt, or a new question about a total.
_WEAK_REF = re.compile(rf"\bthe\s+{_NOUNS}\b", re.I)
_TIME_WORDS = r"(?:day|days|week|weekend|month|quarter|year|morning|afternoon|evening|time|period)"
_THIS_THAT_BEFORE_TIME = re.compile(rf"\b(?:this|that)\s+{_TIME_WORDS}\b", re.I)
# Bare pronouns, which point at the receipt only while it is the topic.
_PRONOUN = re.compile(rf"\b(?:it|its)\b|\b(?:this|that)\b(?!\s+{_TIME_WORDS}\b)", re.I)
_WORD = re.compile(r"[a-z]{4,}")
_STOP = {"with", "from", "this", "that", "what", "much", "many", "have", "your", "about"}
_ITEM_WITH_DETERMINER = re.compile(r"\b(the|that|this|my)\s+([a-z]{4,})", re.I)
# Words that lean on the turn before: such a message means nothing alone.
_FOLLOW_UP = re.compile(r"\b(what about|and|also|that|those|it|them|this one|that one|how about)\b", re.I)
_SHORT_WORDS = 8
# "show more", "next 10", "keep going": a message made only of these words
# asks for the next page of the search before it. A question that merely
# contains "more" or "next" ("how can I save more?", "next month") carries
# other words too.
_PAGING_WORDS = {"show", "me", "more", "next", "page", "load", "see", "get", "give", "keep", "going", "continue",
                 "results", "result", "matches", "match", "transactions", "purchases", "ones", "of", "those",
                 "them", "these", "the", "some", "any", "anymore", "please", "pls", "ok", "okay", "yes", "sure",
                 "and", "then", "ten", "twenty"}
_PAGING_CUES = {"more", "next", "continue", "going", "anymore"}
_TOKEN = re.compile(r"[a-z]+|\d+")
# "did I go to IKEA?", "when did I last shop at Costco?", "how often do I go
# to Whole Foods?": visits to a place, answered from SQL. The place can carry
# a window ("IKEA last month"); the merchant lookup finds the name inside it.
_VISIT_LEAD = (r"(?:(?:did|have)\s+i\s+(?:ever\s+|recently\s+|already\s+)?"
               r"|when\s+(?:did|have)\s+i\s+(?:last\s+)?"
               r"|when\s+was\s+(?:i\s+last\s+|the\s+last\s+time\s+i\s+)"
               r"|when\s+was\s+my\s+(?:last|latest|most\s+recent)\s+"
               r"|how\s+(?:often|frequently)\s+(?:do|did)\s+i\s+"
               r"|how\s+many\s+times\s+(?:did|have|do)\s+i\s+)")
# Verbs that take a preposition come first, so "visit to X" never leaves
# "to X" as the place.
_VISIT_VERB = (r"(?:(?:visit|trip|purchase|order|stop)\s+(?:to|at|from)"
               r"|(?:go|gone|went|been)\s+(?:to|at)"
               r"|(?:shop|shopped|eat|eaten|ate|stop|stopped)\s+(?:at|by)"
               r"|(?:buy|bought|order|ordered|get|got)(?:\s+(?:anything|something|stuff|food))?\s+(?:from|at)"
               r"|visit|visited|at)")
_VISIT = re.compile(rf"^\s*{_VISIT_LEAD}{_VISIT_VERB}\s+(?P<place>.+?)\s*[?.!]*\s*$", re.I)


def reads_as_follow_up(message: str) -> bool:
    """True when a message leans on the conversation before it."""
    return bool(_FOLLOW_UP.search(message or ""))


def asks_for_more(message: str) -> bool:
    """True when a message only asks for more results ("show me 10 more")."""
    words = _TOKEN.findall((message or "").lower())
    return (0 < len(words) <= _SHORT_WORDS and bool(_PAGING_CUES.intersection(words))
            and all(w in _PAGING_WORDS or w.isdigit() for w in words))


@dataclass(frozen=True)
class Route:
    kind: str
    detail: str = ""
    payload: str | None = None
    intent: AggregationIntent | None = None
    # Routed this way only because of the conversation so far: the text
    # alone means nothing, so it must never key a cached answer.
    follow_up: bool = False

    def as_dict(self) -> dict:
        out: dict = {"kind": self.kind, "detail": self.detail, "payload": self.payload, "intent": None,
                     "follow_up": self.follow_up}
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


def _about_receipt(message: str, receipt: dict, last_route: str | None) -> bool:
    if _STRONG_REF.search(message) or _WEAK_REF.search(message):
        return True
    # "is it a duplicate?" right after a receipt turn; not after an unrelated one.
    if last_route == "receipt" and _PRONOUN.search(message):
        return True
    # "this month" is a window, not a reference, and rules out the rest.
    if _THIS_THAT_BEFORE_TIME.search(message):
        return False
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
          today: date | None = None, last_route: str | None = None, has_history: bool = False,
          last_search: dict | None = None) -> Route:
    """last_route is the route of the thread's previous turn ("receipt" right
    after an upload); has_history says whether the thread has messages;
    last_search is the search the previous turn showed, if it ran one."""
    text = (message or "").strip()
    if (m := _REMEMBER.match(text)):
        return Route("remember", "memory command", payload=m.group(1).strip().rstrip("."))
    if (m := _FORGET.match(text)):
        return Route("forget", "memory command", payload=m.group(1).strip().rstrip("."))
    if last_search is not None and asks_for_more(text):
        return Route("more", "next page of the last search", follow_up=True)
    intent = classify_aggregation(text, today=today)
    if active_receipt and _about_receipt(text, active_receipt, last_route):
        # Strong references always win
        if _STRONG_REF.search(text):
            return Route("receipt", "follow-up about the active receipt")
        # An aggregation with its own explicit subject or window is a new question
        has_window = extract_window(text, today=today) is not None
        if intent and (intent.subject or has_window):
            return Route("aggregate", f"{intent.operation} answered by SQL", intent=intent)
        # Otherwise it's about the receipt
        return Route("receipt", "follow-up about the active receipt")
    if (m := _VISIT.match(text)):
        return Route("visits", "visits to a place, answered by SQL", payload=m.group("place"))
    if context:
        return Route("plan", f"opened with {context.get('kind', 'context')}")
    if not active_receipt and _STRONG_REF.search(text):
        return Route("receipt", "about your latest receipt")
    if not is_financial_query(text):
        # "which one was the biggest?", "thanks!": not off topic mid-conversation.
        if has_history and (len(text.split()) <= _SHORT_WORDS or reads_as_follow_up(text)):
            return Route("plan", "follow-up in this conversation", follow_up=True)
        return Route("offtopic", "not a money question")
    if intent is not None:
        return Route("aggregate", f"{intent.operation} answered by SQL", intent=intent)
    return Route("plan", "open question")
