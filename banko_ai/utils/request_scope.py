"""Whose data the current request is about.

Caches use this to keep one user's results away from another's without
every caller threading a user id through five providers. Flask sets it per
request from the session; background threads have no scope and must pass
one explicitly."""

from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager

_scope: contextvars.ContextVar[str] = contextvars.ContextVar("banko_user_scope", default="")


def current_user_scope() -> str:
    return _scope.get()


def set_user_scope(user_id: str | None) -> contextvars.Token:
    return _scope.set(str(user_id or ""))


def reset_user_scope(token: contextvars.Token) -> None:
    _scope.reset(token)


@contextmanager
def user_scope(user_id: str | None) -> Iterator[None]:
    token = set_user_scope(user_id)
    try:
        yield
    finally:
        _scope.reset(token)
