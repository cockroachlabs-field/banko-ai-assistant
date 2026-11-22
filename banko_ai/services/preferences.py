"""A person's own settings for the assistant.

The deployment's settings (config/settings.py) say how Banko behaves for
everyone. A few of them shape the answers one person gets and are read when
a question is asked: how the answer cache matches questions, how many
purchases a search page holds, what the assistant remembers. Those a person
may set for themselves, on the Settings page.

Each one stands in for a deployment setting of the same name, which stays
the default: only what a person chose differently is stored (users.
preferences), so a default that changes reaches everyone who never chose.
Nothing secret is here, nothing read once at boot, and nothing that belongs
to whoever runs Banko (connections, pools, the model, what a question may
cost). Nor how long a cached answer or search is kept: a persona's cache is
shared, and one person's lifetime would decide how old an answer the next
person gets."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text

from ..config.settings import get_config
from ..db.engine import get_engine, routed, run_transaction
from ._sql import valid_uuid


@dataclass(frozen=True)
class Setting:
    key: str    # the deployment setting it stands in for
    group: str
    label: str
    help: str
    kind: str   # choice, number, whole or flag
    low: float | None = None
    high: float | None = None
    step: float | None = None
    choices: tuple[tuple[str, str, str], ...] = ()  # value, label, what choosing it means
    only_with: tuple[str, str] | None = None        # matters only while another setting has this value
    percent: bool = False                           # a share of one, which the page shows as a percentage
    shapes: bool = False                            # changes what an answer contains (see Preferences.shape)
    default: Callable[[], Any] | None = None        # when the deployment's value needs more than reading


def _page_size() -> int:
    from .transactions import search_page_size
    return search_page_size()


SETTINGS: tuple[Setting, ...] = (
    Setting("cache_match", "Answer cache", "Reuse an answer for",
            "Which earlier question's answer may be replayed instead of asking the model again. A replayed answer "
            "says which question it was written for.", "choice",
            # A category or one of the user's merchants in other words was a third choice. Those questions
            # are read from SQL and never reach a model, so there is no answer of theirs to reuse.
            choices=(("exact", "The same question",
                      "The same words only, give or take capitals and punctuation."),
                     ("similar", "Also a similar question",
                      "Also, between the first questions of two conversations, any question as similar as set "
                      "below, unless it names other dates or numbers or differs in a word such as not or under. It "
                      "cannot tell what a question is about, so some answers will be to another question."))),
    Setting("cache_similarity_threshold", "Answer cache", "How similar",
            "For a similar question: how close it must be to one already answered. Questions about different "
            "things are often close too (\"coffee expenses\" and \"tea expenses\" are 63% similar), so lower means "
            "more answers from the cache and more of them to another question.", "number",
            low=0.5, high=0.99, step=0.01, only_with=("cache_match", "similar"), percent=True),
    Setting("search_page_size", "Search", "Purchases per page",
            "How many purchases a search in the assistant shows before Show more.", "whole", low=1, high=25,
            default=_page_size, shapes=True),
    Setting("assistant_implicit_memory", "Memory", "Remember what I mention",
            "Save lasting facts you mention (\"I prefer\", \"my goal is\", \"I'm saving for\") without being asked. "
            "Each one costs a model call after the answer.", "flag"),
    Setting("assistant_memory_recall_k", "Memory", "Memories per question",
            "How many of your saved memories the assistant may bring to one question.", "whole", low=1, high=20,
            shapes=True),
    Setting("assistant_memory_max_distance", "Memory", "How loosely a memory may fit",
            "The largest distance between a question and a memory that still counts as relevant, from 0.1 (almost "
            "the same words) to 1.", "number", low=0.1, high=1.0, step=0.05, shapes=True),
)
_BY_KEY = {s.key: s for s in SETTINGS}


class Invalid(ValueError):
    """A value that does not fit its setting. str() is a sentence the page shows."""

    def __init__(self, key: str | None, message: str):
        super().__init__(message)
        self.key = key


def _default(setting: Setting) -> Any:
    return setting.default() if setting.default else getattr(get_config(), setting.key)


def _plain(number: float) -> str:
    return f"{number:g}"


def _wanted(setting: Setting) -> str:
    kind = "whole number" if setting.kind == "whole" else "number"
    return f"{setting.label} takes a {kind} between {_plain(setting.low or 0)} and {_plain(setting.high or 0)}."


def _typed(setting: Setting, value: Any) -> Any:
    """The value in its setting's type, or Invalid. A bool is never a number."""
    if setting.kind == "flag":
        if not isinstance(value, bool):
            raise Invalid(setting.key, f"{setting.label} is either on or off.")
        return value
    if setting.kind == "choice":
        offered = [c[0] for c in setting.choices]
        if value not in offered:
            raise Invalid(setting.key, f"{setting.label} takes one of: {', '.join(offered)}.")
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Invalid(setting.key, _wanted(setting))
    if setting.kind == "whole":
        if isinstance(value, float):
            if not math.isfinite(value) or value != int(value):
                raise Invalid(setting.key, _wanted(setting))
            return int(value)
        return value  # an int of any size: the range refuses what is too large
    try:
        number = float(value)
    except OverflowError:  # an integer of hundreds of digits
        raise Invalid(setting.key, _wanted(setting)) from None
    if not math.isfinite(number):
        raise Invalid(setting.key, _wanted(setting))
    return round(number, 4)


def _in_range(setting: Setting, value: Any) -> Any:
    if setting.kind in ("number", "whole") and not (setting.low or 0) <= value <= (setting.high or 0):
        raise Invalid(setting.key, _wanted(setting))
    return value


class Preferences:
    """The settings in effect for one person: their own where they chose,
    the deployment's otherwise. Read like the settings object, by name."""

    __slots__ = ("own",)

    def __init__(self, own: Mapping[str, Any] | None = None):
        self.own: dict[str, Any] = {k: v for k, v in (own or {}).items() if k in _BY_KEY}

    def __getattr__(self, key: str) -> Any:
        setting = _BY_KEY.get(key)
        if setting is None:
            raise AttributeError(key)
        return self.own[key] if key in self.own else _default(setting)

    @property
    def shape(self) -> str:
        """The settings that change what an answer contains, as one string.
        A cached answer is kept and looked up by it: three purchases a page
        make another answer than ten, and whoever has ten must not be served
        the three. How questions are matched shapes no answer and is not in
        it."""
        return ";".join(f"{s.key}={getattr(self, s.key)}" for s in SETTINGS if s.shapes)


def defaults() -> Preferences:
    """Nobody's choices: the deployment's settings."""
    return Preferences()


def _usable(stored: Any) -> dict[str, Any]:
    """What was stored, minus anything that no longer fits: a setting that
    is gone, a value outside a range that has since narrowed."""
    out: dict[str, Any] = {}
    for key, value in (stored if isinstance(stored, Mapping) else {}).items():
        setting = _BY_KEY.get(key)
        if setting is None:
            continue
        try:
            out[key] = _in_range(setting, _typed(setting, value))
        except Invalid:
            continue
    return out


def of(user_id: str, *, url: str | None = None) -> Preferences:
    """This person's settings. An account that is not there has none."""
    if not valid_uuid(user_id):
        return Preferences()
    with get_engine(routed(url, user_id)).connect() as conn:
        stored = conn.execute(text("SELECT preferences FROM users WHERE user_id = CAST(:u AS UUID)"),
                              {"u": user_id}).scalar()
    return Preferences(_usable(stored))


def save(user_id: str, values: Any, *, url: str | None = None) -> Preferences:
    """Make these the person's settings, all of them: a setting left out, or
    equal to the deployment's, goes back to following the deployment. One
    value that does not fit raises Invalid and nothing is saved."""
    if not isinstance(values, Mapping):
        raise Invalid(None, "Send the settings as values by name.")
    own: dict[str, Any] = {}
    for key, value in values.items():
        setting = _BY_KEY.get(str(key))
        if setting is None:
            raise Invalid(str(key)[:60], f"There is no setting called {str(key)[:60]}.")
        typed = _typed(setting, value)
        if typed != _default(setting):  # the deployment's own value may lie outside the range offered here
            own[setting.key] = _in_range(setting, typed)

    def fn(conn) -> int:
        return int(conn.execute(text(
            "UPDATE users SET preferences = CAST(:p AS JSONB) WHERE user_id = CAST(:u AS UUID)"),
            {"p": json.dumps(own) if own else None, "u": user_id}).rowcount)

    if not valid_uuid(user_id) or not run_transaction(fn, url=url, user_id=user_id):
        raise LookupError(f"no account {user_id}")
    return Preferences(own)


def describe(mine: Preferences) -> dict[str, Any]:
    """Every setting as the Settings page shows it, with the value in effect."""
    listed = []
    for s in SETTINGS:
        default = _default(s)
        listed.append({
            "key": s.key, "group": s.group, "label": s.label, "help": s.help, "kind": s.kind,
            "min": s.low, "max": s.high, "step": s.step if s.kind == "number" else (1 if s.kind == "whole" else None),
            "choices": [{"value": value, "label": label, "help": means} for value, label, means in s.choices],
            "only_with": {"key": s.only_with[0], "value": s.only_with[1]} if s.only_with else None,
            "percent": s.percent,
            "value": getattr(mine, s.key), "default": default, "changed": s.key in mine.own,
        })
    return {"groups": list(dict.fromkeys(s.group for s in SETTINGS)), "settings": listed}
