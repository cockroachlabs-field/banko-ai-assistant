"""Every figure a model writes has to be one it was given.

SQL computes the figures and code writes the cards. A model still writes the
sentences around an open question's facts, the observations under a total
and the translation of a card. It is told never to invent a number and never
to do arithmetic, and now and then it rounds one, adds two up or miscounts a
list all the same. So what it writes is read before it is shown: an amount,
a count, a percentage or a date that is in nothing the model was given is
not shown.

What a model was given is the facts from the tools, the question, the
conversation so far, the user's memories and the context the conversation
was opened from (given). A count may also be the length of a list among
them: "ten purchases" beside ten rows was counted, not invented, and the
count is checked against the list.

This reads figures, not meaning. A figure that was given can still stand in
a sentence that says the wrong thing about it; what it rules out is a figure
from nowhere: rounded, added up, miscounted, or made up. No model is asked,
so the same text is judged the same way every time."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

_CENT = Decimal("0.01")
_MONTH_NAMES = {"january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4,
                "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9, "sep": 9,
                "sept": 9, "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12}
# "may" is a month only with a capital: "you may 2" is no date.
_MONTH = (r"(?:(?i:january|jan|february|feb|march|mar|april|apr|june|jun|july|jul|august|aug|september|sept|sep|"
          r"october|oct|november|nov|december|dec)|May)")
_TH = r"(?:st|nd|rd|th)"
# What holds digits and is no figure: a purchase's marker, an id, a time of
# day, a list's own numbering, a footnote mark ("[1]", "[Cite: 2]"), a rank
# ("the 2nd largest").
_NO_FIGURE = re.compile(
    r"\[\s*(?:cite\s*:\s*)?txn\s*:[^\]]*\]"
    r"|\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
    r"|(?<![\d.,])\d{1,2}:\d{2}(?::\d{2})?(?![\d.,])"
    r"|(?m:^[ \t]*\d{1,3}[.)](?=[ \t]))"
    r"|\[\s*(?:[Cc]ite\s*:?\s*)?\d{1,3}\s*\]"
    rf"|\b\d+{_TH}\b")
_ISO = re.compile(r"(?<![\d-])(\d{4})-(\d{2})-(\d{2})(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?(?![\d-])")
_YEAR_MONTH = re.compile(r"(?<![\d-])(\d{4})-(\d{2})(?![\d-])")       # a month as a tool names it: "2026-09"
_SLASHED = re.compile(r"(?<![\d/.])(\d{1,2})[/.](\d{1,2})[/.](\d{4})(?![\d/]|\.\d)")
_MONTH_YEAR = re.compile(rf"\b({_MONTH})\.?,?\s+(\d{{4}})\b")
# "Nov 1–23", "November 1-23, 2026", "1–23 November": two days of one month.
_DAYS_OF_MONTH = re.compile(rf"\b({_MONTH})\.?\s+(\d{{1,2}}){_TH}?\s*[–—-]\s*(\d{{1,2}}){_TH}?\b(?!\s*[:%$€£])(?:,?\s+(\d{{4}})\b)?")
_DAYS_BEFORE_MONTH = re.compile(rf"(?<![A-Za-z0-9_.,$€£])(\d{{1,2}}){_TH}?\s*[–—-]\s*(\d{{1,2}}){_TH}?\s+(?:of\s+)?({_MONTH})\b\.?"
                                r"(?:,?\s+(\d{4})\b)?")
_MONTH_DAY = re.compile(rf"\b({_MONTH})\.?\s+(\d{{1,2}}){_TH}?\b(?!\s*[:%$€£])(?:,?\s+(\d{{4}})\b)?")
_DAY_MONTH = re.compile(rf"(?<![A-Za-z0-9_.,$€£])(\d{{1,2}}){_TH}?\s+(?:of\s+)?({_MONTH})\b\.?(?:,?\s+(\d{{4}})\b)?")
# A letter of the Latin alphabet or a digit right before a number makes it part of a word ("A380",
# "v2"); a letter of a script written without spaces does not ("は17件", "消费了947").
_IN_A_WORD = r"A-Za-z0-9_À-ÖØ-öø-ÿ"
_YEAR = re.compile(r"(?<![" + _IN_A_WORD + r".,$€£-])((?:19|20)\d{2})(?![" + _IN_A_WORD + r"%$€£]|[.,]\d|\s?(?:dollars|USD)\b)")
_SPACE = "  "
_NUMBER = re.compile(
    r"(?<![" + _IN_A_WORD + r".,])(?P<lead>[$€£]\s?)?(?P<n>\d{1,3}(?:[" + _SPACE + r" ]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)*)"
    r"(?P<k>[kK](?![A-Za-z]))?(?P<tail>\s?%|\s?[$€£]|\s(?:dollars|USD)\b|\s?percent\b)?(?!-?[A-Za-z_])")
# A sentence ends at a full stop, a question or exclamation mark before a
# space and a capital, a digit or what opens a line of markdown; not after
# an abbreviation, and not inside a figure, which has no space after its point.
_ABBREVIATIONS = frozenset("st mr mrs ms dr vs no inc ltd co jr sr approx est e.g i.e etc ave blvd".split())
_SENTENCE_END = re.compile(r"([.!?][\"')\]*_]*)(\s+)(?=[A-Z0-9\"'(\[*_$€£¿¡]|[^\x00-\x7f])|([。！？।][\"')\]*_」』）]*)(\s*)")
_WHOLE_LINE = re.compile(r"^\s*(?:\||[-*+]\s|#)")


def _date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _decimal(digits: str) -> Decimal | None:
    try:
        return Decimal(digits)
    except InvalidOperation:
        return None


def _values(raw: str) -> tuple[Decimal, ...]:
    """What a number as written may be worth, likeliest first. "1,850.00"
    and "1.850,00" are one amount; "947,33" has a decimal comma; "1,850"
    is a thousand and more, and "1.850" may be that or 1.85."""
    text = re.sub(r"[" + _SPACE + r" ]", "", raw)
    points, commas = text.count("."), text.count(",")
    if not points and not commas:
        only = _decimal(text)
        return (only,) if only is not None else ()
    last = max(text.rfind("."), text.rfind(","))
    whole, after = text[:last], text[last + 1:]
    grouped = re.fullmatch(r"\d{1,3}(?:[.,]\d{3})*", whole) is not None and len(set(re.findall(r"[.,]", whole))) <= 1
    plain = re.sub(r"[.,]", "", text)
    out: list[Decimal | None] = []
    if points and commas:                      # both: the last one is the decimal point
        out.append(_decimal(f"{re.sub(r'[.,]', '', whole)}.{after}") if grouped else None)
    elif points + commas > 1:                  # one kind, more than once: they group thousands
        out.append(_decimal(plain) if grouped and len(after) == 3 else None)
    elif len(after) == 3 and len(whole) <= 3 and whole != "0":
        # "1,850" is a thousand and more, and nothing else: read as 1.85 as well, a rounded "$8,600" stood on
        # the strength of an $8.60 coffee. "1.850" may be either: a thousand in German, three decimals in English.
        out += [_decimal(plain)] if text[last] == "," else [_decimal(f"{whole}.{after}"), _decimal(plain)]
    else:
        out.append(_decimal(f"{whole}.{after}"))
    return tuple(v for v in out if v is not None)


@dataclass(frozen=True)
class Figure:
    at: int
    raw: str
    kind: str                                   # number, percent, day, month or year
    values: tuple[Decimal, ...] = ()            # number and percent: what it may be worth, likeliest first
    money: bool = False                         # an amount: a currency sign or word stands with it
    days: tuple[tuple[int | None, int, int | None], ...] = ()   # day: (year, month, day) readings; month: (year, month, None)


def _blank(text: str, pattern: re.Pattern[str]) -> str:
    return pattern.sub(lambda m: " " * len(m.group(0)), text)


def read(text: str) -> list[Figure]:
    """Every figure in a text, in the order it stands: amounts and counts,
    percentages, days, months and years. A count written as a word is none:
    "the past four months" and "two things stand out" count what the
    sentence itself lists, and a model that counts in words was held back
    for it when it was right."""
    work = _blank(text or "", _NO_FIGURE)
    found: list[Figure] = []

    def take(pattern: re.Pattern, make) -> None:
        nonlocal work
        for m in pattern.finditer(work):
            made = make(m)
            if made:
                found.extend(made if isinstance(made, list) else [made])
                work = work[:m.start()] + " " * (m.end() - m.start()) + work[m.end():]

    def month(name: str) -> int:
        return _MONTH_NAMES[name.lower().rstrip(".")]

    take(_ISO, lambda m: Figure(m.start(), m.group(0), "day", days=((int(m[1]), int(m[2]), int(m[3])),))
         if _date(int(m[1]), int(m[2]), int(m[3])) else None)
    take(_YEAR_MONTH, lambda m: Figure(m.start(), m.group(0), "month", days=((int(m[1]), int(m[2]), None),))
         if 1 <= int(m[2]) <= 12 else None)
    take(_SLASHED, lambda m: Figure(m.start(), m.group(0), "day", days=tuple(
        d for d in ((int(m[3]), int(m[1]), int(m[2])), (int(m[3]), int(m[2]), int(m[1]))) if _date(*d))) or None)
    def days_of(m: re.Match, name: str, first: str, last: str, year: str | None) -> list[Figure] | None:
        if not (1 <= int(first) <= 31 and 1 <= int(last) <= 31):
            return None
        return [Figure(m.start(), m.group(0), "day", days=((int(year) if year else None, month(name), int(day)),))
                for day in (first, last)]

    take(_DAYS_OF_MONTH, lambda m: days_of(m, m[1], m[2], m[3], m[4]))
    take(_DAYS_BEFORE_MONTH, lambda m: days_of(m, m[3], m[1], m[2], m[4]))
    take(_MONTH_YEAR, lambda m: Figure(m.start(), m.group(0), "month", days=((int(m[2]), month(m[1]), None),)))
    take(_MONTH_DAY, lambda m: Figure(m.start(), m.group(0), "day",
                                      days=((int(m[3]) if m[3] else None, month(m[1]), int(m[2])),))
         if 1 <= int(m[2]) <= 31 else None)
    take(_DAY_MONTH, lambda m: Figure(m.start(), m.group(0), "day",
                                      days=((int(m[3]) if m[3] else None, month(m[2]), int(m[1])),))
         if 1 <= int(m[1]) <= 31 else None)
    take(_YEAR, lambda m: Figure(m.start(), m.group(0), "year", values=(Decimal(m[1]),),
                                 days=((int(m[1]), 0, None),)))

    def number(m: re.Match) -> Figure | None:
        values = _values(m["n"])
        if m["k"]:
            values = tuple(v * 1000 for v in values)
        if not values:
            return None
        tail = (m["tail"] or "").strip()
        return Figure(m.start(), m.group(0).strip(), "percent" if tail in ("%", "percent") else "number", values,
                      money=bool(m["lead"]) or tail not in ("", "%", "percent"))

    take(_NUMBER, number)
    return sorted(found, key=lambda f: f.at)


# A lookup's name for a number that is a percentage as it stands ("apr": 22.99). A share between 0 and 1
# ("pct": 0.8731, "utilization": 0.42) needs no name: times a hundred it is the percentage.
_PERCENT_NAME = re.compile(r"apr|rate|percent|pct", re.I)


def _decimals(value: Decimal) -> int:
    """How many decimals a number was written with."""
    exponent = value.as_tuple().exponent
    return -exponent if isinstance(exponent, int) and exponent < 0 else 0


@dataclass(frozen=True)
class Given:
    """The figures in what a model was given."""
    cents: frozenset[Decimal]                   # every number, without its sign, to the cent
    exact: frozenset[Decimal]                   # the same as given, for a share said as a percentage
    dates: frozenset[date]
    months: frozenset[tuple[int, int]] = frozenset()   # a month named without its days: "2026-09", "September 2026"
    percents: frozenset[Decimal] = frozenset()  # given as a percentage: under a name that says so, or with its sign

    def _worked_out(self, value: Decimal) -> bool:
        """Whether an amount or a count is the sum or the difference of two
        that were given, to the cent, or (with decimals) one of them divided
        by a count that was given: an average."""
        cent = value.quantize(_CENT)
        if any((cent - a) in self.cents for a in self.cents if _CENT <= a < cent):       # a + b
            return True
        if any((cent + a) in self.cents for a in self.cents if a >= _CENT):              # b - a
            return True
        if not _decimals(value):
            return False
        want, counts = float(value), [float(n) for n in self.exact if n >= 2 and n == n.to_integral_value()]
        return any(abs(float(a) / n - want) <= 0.005 + 1e-9 for a in self.exact if a > 0 for n in counts)

    def holds(self, figure: Figure) -> bool:
        """Whether a figure was given. An amount (a currency sign, or
        decimals) may also be worked out from two that were: to the cent,
        that leaves no room for a made-up one. A bare count has to be given
        as it is: small numbers add up to anything. A share that was not
        given is the reader's to check, against the figures written beside
        it (shared)."""
        if figure.kind == "number":
            return any(abs(v).quantize(_CENT) in self.cents
                       or ((figure.money or _decimals(v)) and self._worked_out(abs(v))) for v in figure.values)
        if figure.kind == "percent":
            # As a percentage, or as a share between 0 and 1 rounded to the decimals it is written with.
            # Never an amount of the same digits: "57%" beside a $57.00 purchase was 54%.
            return any(abs(v).quantize(_CENT) in self.percents
                       or any(abs(e * 100 - abs(v)) <= Decimal(5) / Decimal(10) ** (_decimals(v) + 1) for e in self.exact)
                       for v in figure.values)
        if figure.kind == "year":
            year = figure.days[0][0]
            return (any(d.year == year for d in self.dates) or any(y == year for y, _ in self.months)
                    or figure.values[0] in self.cents)
        if figure.kind == "month":
            return figure.days[0][:2] in self.months
        return any((year is None or d.year == year) and d.month == month and d.day == day
                   for year, month, day in figure.days for d in self.dates)


def given(*material: Any) -> Given:
    """The figures in everything a model was given: tool results, the
    question, earlier messages, memories. Any nesting of dicts, lists,
    strings and numbers. A list also gives its length, so a count of its
    rows can be checked. Digits inside a date or an id give no number: the
    18 of "2026-10-18" is a day, and must not vouch for "18 purchases"."""
    numbers: set[Decimal] = set()
    dates: set[date] = set()
    months: set[tuple[int, int]] = set()
    percents: set[Decimal] = set()

    def walk(value: Any, as_percent: bool = False) -> None:
        if value is None or isinstance(value, bool):
            return
        if isinstance(value, (int, float, Decimal)):
            number = _decimal(str(value))
            if number is not None and number.is_finite():
                numbers.add(abs(number))
                if as_percent:
                    percents.add(abs(number))
        elif isinstance(value, date):
            dates.add(value if type(value) is date else value.date())  # type: ignore[attr-defined]
        elif isinstance(value, str):
            for figure in read(value):
                if figure.kind in ("number", "percent"):
                    numbers.update(abs(v) for v in figure.values)
                    if figure.kind == "percent":
                        percents.update(abs(v) for v in figure.values)
                elif figure.kind == "day":
                    dates.update(d for d in (_date(y, m, day) for y, m, day in figure.days
                                             if y is not None and day is not None) if d)
                elif figure.kind == "month":
                    months.add((figure.days[0][0], figure.days[0][1]))  # type: ignore[arg-type]
                elif figure.kind == "year":
                    numbers.add(figure.values[0])
        elif isinstance(value, dict):
            for key, inner in value.items():
                walk(inner, bool(_PERCENT_NAME.search(str(key))))
        elif isinstance(value, Iterable):
            items = list(value)
            numbers.add(Decimal(len(items)))
            for inner in items:
                walk(inner, as_percent)

    for part in material:
        walk(part)
    months.update((d.year, d.month) for d in dates)
    return Given(frozenset(n.quantize(_CENT) for n in numbers), frozenset(numbers), frozenset(dates), frozenset(months),
                 frozenset(p.quantize(_CENT) for p in percents))


def shared(figure: Figure, written: Iterable[Decimal]) -> bool:
    """Whether a percentage is one figure as a share of another, or how much
    more or less one is than the other, to the decimals it is written with.
    written: the amounts and counts in the same answer. Of all the figures a
    model was given, some two give nearly any percentage; of the two or
    three it wrote down beside it, only the right one."""
    numbers = [float(n) for n in written if n > 0]
    for value in figure.values:
        want, slack = float(abs(value)), 0.5 * 10 ** -_decimals(value) + 1e-9
        if any(abs(a / b * 100 - want) <= slack or abs(abs(a - b) / b * 100 - want) <= slack
               for a in numbers for b in numbers if a != b):
            return True
    return False


def unfounded(text: str, known: Given) -> list[str]:
    """The figures in a text that are in nothing the model was given, as
    written: those of every piece a Reader would hold back."""
    reader = Reader(known)
    reader.feed(text or "")
    reader.finish()
    return [raw for _, missing in reader.held for raw in missing]


def units(text: str) -> list[str]:
    """A text cut where a sentence or a line ends, every character kept: the
    pieces a reader lets through or holds back whole. A table row, a list
    item and a heading are one piece each."""
    out: list[str] = []
    for line in re.findall(r"[^\n]*\n|[^\n]+", text or ""):
        if _WHOLE_LINE.match(line):
            out.append(line)
            continue
        start = 0
        for m in _SENTENCE_END.finditer(line):
            before = re.findall(r"[\w.]+$", line[:m.start()])
            if before and before[0].lower().rstrip(".") in _ABBREVIATIONS:
                continue
            out.append(line[start:m.end()])
            start = m.end()
        if start < len(line):
            out.append(line[start:])
    return out


@dataclass
class Reader:
    """Reads a model's answer as it arrives and lets through what stands.
    feed() takes the next piece and returns what may be shown now: every
    whole sentence or line whose figures were all given. finish() does the
    same for what is left. A piece with a figure from nowhere is held back
    (held: the piece and those figures) and is never shown.

    A share that was not given ("22.99% of your spending") stands when it
    is the share of two figures written in the same answer. Those may come
    in the sentence after it ("that is $946.15 of $4,114.88"), so a piece
    with such a share waits, and what follows waits behind it to keep its
    place, until the figures are there or the answer ends."""
    known: Given
    shown: str = ""
    held: list[tuple[str, list[str]]] = field(default_factory=list)
    checked: int = 0
    figures: int = 0
    _waiting: str = ""
    _behind: list[tuple[str, list[Figure]]] = field(default_factory=list)   # whole pieces, with the shares to check
    _written: list[Decimal] = field(default_factory=list)                   # the amounts and counts written so far

    def _judge(self, pieces: list[str], last: bool = False) -> str:
        for piece in pieces:
            read_here = read(piece)
            if piece.strip():
                self.checked += 1
            self.figures += len(read_here)
            missing = [f for f in read_here if not self.known.holds(f)]
            if any(f.kind != "percent" for f in missing):
                self.held.append((piece, [f.raw for f in missing]))
                continue
            self._written += [abs(v) for f in read_here if f.kind == "number" for v in f.values]
            self._behind.append((piece, missing))
        out = []
        while self._behind:
            piece, shares = self._behind[0]
            unchecked = [f for f in shares if not shared(f, self._written)]
            if unchecked and not last:
                break       # what it is a share of may still follow
            self._behind.pop(0)
            if unchecked:
                self.held.append((piece, [f.raw for f in unchecked]))
            else:
                out.append(piece)
        self.shown += "".join(out)
        return "".join(out)

    def feed(self, piece: str) -> str:
        self._waiting += piece
        whole = units(self._waiting)
        if len(whole) < 2:
            return ""
        self._waiting = whole[-1]       # the last one may not have ended yet
        return self._judge(whole[:-1])

    def finish(self) -> str:
        rest, self._waiting = self._waiting, ""
        return self._judge(units(rest), last=True)


def _as_translated(original: str) -> Given:
    """What a translation of a text may hold: the text's own figures, and
    the numbers its days are made of, for a day written out in another
    language ("1 de octubre de 2026", "01.10.2026")."""
    known = given(original)
    parts = {Decimal(n).quantize(_CENT) for d in known.dates for n in (d.year, d.month, d.day)}
    return Given(known.cents | parts, known.exact, known.dates, known.months, known.percents)


def translation_reader(original: str) -> Reader:
    """A Reader for a translation as it streams: a sentence with a figure
    the original does not have is held back."""
    return Reader(_as_translated(original))


def released(stream: Iterable[str], reader: Reader) -> Iterable[str]:
    """A model's stream as a reader lets it through: whole sentences and
    lines whose figures were all given, as they complete. A stream that
    breaks still gives what it had begun, when that stands, before the
    error goes on to whoever reads it."""
    try:
        for piece in stream:
            out = reader.feed(piece)
            if out:
                yield out
    except Exception:
        out = reader.finish()
        if out:
            yield out
        raise
    out = reader.finish()
    if out:
        yield out


def translation_keeps(original: str, other: str) -> bool:
    """Whether another wording of a text (its translation) keeps the
    figures: none changed, none added, and no amount of money lost.

    Every figure in the translation has to be one of the original's. A day
    may be written the language's own way ("1 de octubre de 2026",
    "01.10.2026"): its numbers are the day's. A decimal comma or a point
    for thousands is the same amount. And every amount of money in the
    original has to be in the translation: a count spelled out in words is
    no figure lost, a purchase's amount that is gone is."""
    if not (other or "").strip():
        return False
    known = _as_translated(original)
    theirs = read(other)
    if any(not known.holds(f) for f in theirs):
        return False
    have = {abs(v).quantize(_CENT) for f in theirs if f.kind in ("number", "percent") for v in f.values}
    return all(any(abs(v).quantize(_CENT) in have for v in f.values) for f in read(original) if f.money)
