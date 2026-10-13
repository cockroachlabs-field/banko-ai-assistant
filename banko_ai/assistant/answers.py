"""Pieces of answers that never need a model for their facts: the
aggregation card, the language names, citation markers and translation of
deterministic text. Figures come from SQL; a model only adds prose."""

from __future__ import annotations

import re
from datetime import timedelta

LANGUAGES = {
    "en-US": "English", "es-ES": "Spanish", "fr-FR": "French", "de-DE": "German",
    "it-IT": "Italian", "pt-PT": "Portuguese", "ja-JP": "Japanese", "ko-KR": "Korean",
    "zh-CN": "Chinese", "hi-IN": "Hindi",
}
# Any [txn:...] marker a model writes, with the spaces before it on the same
# line. Models copy the template loosely: angle brackets around the id
# ([txn:<uuid>]), the bare placeholder ([txn:<expense_id>]), stray spaces.
_MARKER = re.compile(r"[ \t]*\[\s*txn\s*:([^\]]*)\]", re.I)
_MARKER_ID = re.compile(r"\s*<?\s*([0-9a-fA-F-]{36})\s*>?\s*")


def language_name(code: str | None) -> str:
    return LANGUAGES.get(code or "", "English")


def _window(agg) -> str:
    # window_end is exclusive (the day after the last one counted), so the
    # card names the last day the sum includes.
    return f"from {agg.window_start} to {agg.window_end - timedelta(days=1)}"


def render_aggregation_markdown(agg) -> str:
    window = _window(agg)
    merchant = getattr(agg, "merchant", None)
    scope = f"at {merchant}" if merchant else f"on {agg.category or 'all categories'}"
    if not agg.count:
        return (f"No matching transactions {scope} {window}. "
                f"Try a wider window or a different category.")
    if agg.operation == "count":
        headline = f"You made **{agg.count} transactions** {scope} {window}."
    elif agg.operation == "average":
        headline = (f"Your average transaction {scope} was **${agg.average:,.2f}** {window} "
                    f"({agg.count} transactions, ${agg.total:,.2f} total).")
    else:
        headline = f"You spent **${agg.total:,.2f}** {scope} {window} ({agg.count} transactions)."
    lines = [headline, "", "| Date | Merchant | Amount |", "|---|---|---|"]
    for row in agg.rows:
        lines.append(f"| {row['date']} | {row['merchant']} | ${row['amount']:,.2f} |")
    return "\n".join(lines)


def insights_messages(agg, question: str) -> list[dict]:
    """The prompt for two or three observations over figures SQL already
    computed (top merchants are summed here, never by the model)."""
    merchants: dict[str, float] = {}
    for row in agg.rows:
        merchants[row["merchant"]] = merchants.get(row["merchant"], 0) + row["amount"]
    top = ", ".join(f"{m} (${v:,.2f})" for m, v in sorted(merchants.items(), key=lambda kv: -kv[1])[:3])
    scope = getattr(agg, "merchant", None) or agg.category or "all categories"
    prompt = ("You are a personal finance coach. Exact figures, already computed from the user's "
              f"expense database: total ${agg.total:,.2f} across {agg.count} transactions on {scope} "
              f"{_window(agg)}. Top merchants: {top}. "
              f"The user asked: \"{question}\". Write two or three short, specific observations or "
              "suggestions as markdown bullets. Use only the figures given; do not compute new totals "
              "or restate the question. No headings, bullets only.")
    return [{"role": "user", "content": prompt}]


def aggregation_insights(agg, question: str, chat) -> str | None:
    """Two or three observations over figures SQL already computed."""
    try:
        text = chat.complete(insights_messages(agg, question), temperature=0.4)
    except Exception:
        return None
    return text.strip() or None


def translation_messages(text: str, language: str) -> list[dict]:
    return [{"role": "user", "content": (
        f"Translate the following financial summary into {language}. Keep every number, "
        "currency amount, percentage, date, and merchant name exactly as written. Keep the "
        f"markdown formatting. Output only the translation.\n\n{text}")}]


def translate(text: str, language: str, chat) -> str:
    if not text or language == "English":
        return text
    try:
        out = chat.complete(translation_messages(text, language), temperature=0.1)
    except Exception:
        return text
    return out.strip() or text


def _marker_id(inner: str) -> str | None:
    m = _MARKER_ID.fullmatch(inner)
    return m.group(1).lower() if m else None


def citation_ids(text: str) -> list[str]:
    """The ids the markers in text point at, in order, once each."""
    seen: list[str] = []
    for m in _MARKER.finditer(text or ""):
        tid = _marker_id(m.group(1))
        if tid and tid not in seen:
            seen.append(tid)
    return seen


def resolve_citations(text: str, known) -> str:
    """Markers for ids in known become canonical [txn:<id>]; every other
    marker (a placeholder, an id that is not one of the turn's purchases)
    goes, so nothing raw is ever cached, streamed in the answer or shown."""
    ids = {str(k).lower() for k in known}

    def keep_or_drop(m: re.Match) -> str:
        tid = _marker_id(m.group(1))
        if tid is None or tid not in ids:
            return ""
        whole = m.group(0)
        lead = whole[:len(whole) - len(whole.lstrip(" \t"))]
        return f"{lead}[txn:{tid}]"

    return _MARKER.sub(keep_or_drop, text or "")


def strip_citations(text: str) -> str:
    """The text without any [txn:...] marker, well formed or not."""
    return _MARKER.sub("", text or "")


_SECOND_PERSON = (
    (re.compile(r"\bI['’]m\b", re.I), "you're"), (re.compile(r"\bI am\b", re.I), "you are"),
    (re.compile(r"\bI was\b", re.I), "you were"), (re.compile(r"\bI['’]ve\b", re.I), "you've"),
    (re.compile(r"\bI['’]ll\b", re.I), "you'll"), (re.compile(r"\bI['’]d\b", re.I), "you'd"),
    (re.compile(r"\bI\b", re.I), "you"), (re.compile(r"\bmyself\b", re.I), "yourself"),
    (re.compile(r"\bmine\b", re.I), "yours"), (re.compile(r"\bmy\b", re.I), "your"),
    (re.compile(r"\bme\b", re.I), "you"),
)


def second_person(text: str) -> str:
    """A memory as said back to the user: "I'm saving for Japan" becomes
    "you're saving for Japan". Covers the common first-person forms only."""
    for pattern, repl in _SECOND_PERSON:
        text = pattern.sub(repl, text)
    return text


def redirect_message() -> str:
    from ..utils.intent_classifier import REDIRECT_MESSAGE
    return REDIRECT_MESSAGE
