"""What every page's shell needs: brand, theme, languages, the persona the
UI shows, the signed-in user and the nav. The nav lists only pages whose
routes are registered, so a page appears in the sidebar when it ships."""

from __future__ import annotations

import logging

from flask import current_app, g, request, session, url_for

from ..assistant.answers import LANGUAGES
from ..config.settings import get_config
from ..services.profile import get_profile

log = logging.getLogger("banko.shell")
_UNREAD = object()

NAV = (
    {"key": "dashboard", "label": "Dashboard", "icon": "layout-dashboard", "endpoint": "pages.dashboard", "group": "money"},
    {"key": "accounts", "label": "Accounts", "icon": "landmark", "endpoint": "pages.accounts", "group": "money"},
    {"key": "transactions", "label": "Transactions", "icon": "arrow-left-right", "endpoint": "pages.transactions", "group": "money"},
    {"key": "cashflow", "label": "Cash flow", "icon": "chart-column", "endpoint": "pages.cashflow", "group": "money"},
    {"key": "budget", "label": "Budget", "icon": "piggy-bank", "endpoint": "pages.budget", "group": "money"},
    {"key": "recurring", "label": "Recurring", "icon": "repeat", "endpoint": "pages.recurring", "group": "money"},
    {"key": "goals", "label": "Goals", "icon": "target", "endpoint": "pages.goals", "group": "money"},
    {"key": "assistant", "label": "Assistant", "icon": "sparkles", "endpoint": "pages.assistant", "group": "ai"},
    {"key": "coach", "label": "Coach", "icon": "lightbulb", "endpoint": "pages.coach", "group": "ai"},
    {"key": "agents", "label": "Agents", "icon": "bot", "endpoint": "pages.agents", "group": "ai"},
    {"key": "platform", "label": "Platform", "icon": "server", "endpoint": "pages.platform", "group": "system"},
    {"key": "settings", "label": "Settings", "icon": "settings", "endpoint": "pages.settings", "group": "system"},
)
_GROUPS = (("money", None), ("ai", "AI"), ("system", None))


def nav_groups(app) -> list[dict]:
    have = set(app.view_functions)
    out = []
    for group, title in _GROUPS:
        items = [{"key": n["key"], "label": n["label"], "icon": n["icon"], "href": url_for(n["endpoint"])}
                 for n in NAV if n["group"] == group and n["endpoint"] in have]
        if items:
            out.append({"title": title, "items": items})
    return out


def _region(profile: dict) -> str | None:
    from ..db.regions import regional_tables_ready
    try:
        return profile.get("home_region") if regional_tables_ready(get_config().database_url) else None
    except Exception:
        return None


def signed_in_profile() -> dict | None:
    """The signed-in user's profile, read once per request: None without a
    session or when that user no longer exists. A failed lookup raises, and
    raises again for the rest of the request instead of retrying."""
    user_id = session.get("user_id")
    if not user_id:
        return None
    found = g.get("_banko_profile", _UNREAD)
    if found is _UNREAD:
        try:
            found = get_profile(user_id)
        except Exception as exc:
            found = exc
        g._banko_profile = found
    if isinstance(found, Exception):
        raise found
    profile: dict | None = found
    return profile


def plain_page() -> None:
    """Mark this request's page as one that renders without the shell (the
    error and warming pages), so it never waits on the database."""
    g._banko_plain = True


def shell_context() -> dict:
    if g.get("_banko_plain"):
        return {}
    try:
        profile = signed_in_profile()
        return {"shell": _shell(profile)} if profile is not None else {}
    except Exception as exc:
        # Class name only: the message could carry a connection string.
        log.warning("page shell unavailable (%s); rendering without it", exc.__class__.__name__)
        return {}


def _is_persona(user_id: str) -> bool:
    from ..config.catalog import get_catalog
    try:
        return user_id in {p.user_id for p in get_catalog().personas}
    except Exception:  # a broken catalog file is reported at boot
        return False


def _shell(profile: dict) -> dict:
    cfg = get_config()
    shown = current_app.extensions["banko_ai"]["shown"]()
    page_raw = (request.endpoint or "").rsplit(".", 1)[-1]
    return {
        "brand": {"name": cfg.brand_name, "accent": cfg.brand_accent},
        "theme": cfg.ui_theme,
        "languages": [{"code": c, "name": LANGUAGES[c]} for c in cfg.ui_languages.split(",")],
        "provider": {"name": shown.name, "model": shown.model,
                     "icon_url": url_for("static", filename=shown.icon_file)},
        "user": {"user_id": profile["user_id"], "username": profile["username"],
                 "display_name": profile["display_name"], "avatar_color": profile["avatar_color"],
                 "initial": (profile["display_name"] or "?")[0].upper(), "region": _region(profile),
                 "persona": _is_persona(profile["user_id"])},
        "nav": nav_groups(current_app),
        "page": {"assistant_thread": "assistant"}.get(page_raw, page_raw),
        "section": {"assistant_thread": "assistant", "account": "accounts"}.get(page_raw, page_raw),
    }
