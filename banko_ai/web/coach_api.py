"""The coach inbox and presenter-only simulated signals, for the signed-in
user (or the default persona without a session, like the rest of /api/v1)."""

from __future__ import annotations

import logging

from flask import Blueprint, current_app, jsonify, request, session

from ..coach.simulate import SimulateUnavailable, build_signal
from ..config.settings import get_config
from ..services import coach_feed

coach_api = Blueprint("coach_api", __name__, url_prefix="/api/v1/coach")
log = logging.getLogger("banko.coach")


def _uid() -> str:
    return session.get("user_id") or get_config().coach_default_user_id


@coach_api.after_request
def _name_the_user(response):
    response.headers["X-Banko-User"] = _uid()
    return response


@coach_api.get("/nudges")
def nudges():
    return jsonify({"nudges": coach_feed.list_nudges(_uid())})


@coach_api.get("/nudges/<nudge_id>")
def nudge(nudge_id: str):
    detail = coach_feed.nudge_detail(_uid(), nudge_id)
    return (jsonify(detail), 200) if detail else (jsonify({"error": "not_found"}), 404)


@coach_api.post("/simulate")
def simulate():
    body = request.get_json(silent=True) or {}
    try:
        signal = build_signal(_uid(), str(body.get("signal_type", "")))
    except SimulateUnavailable as exc:
        return jsonify({"error": str(exc)}), 409
    except ValueError:
        return jsonify({"error": "Pick budget_threshold, anomaly or recurring_drift"}), 400
    queued = current_app.extensions["banko_coach"]["submit"](signal)
    return jsonify({"status": "queued" if queued else "replayed", "signal_id": signal.signal_id}), 202
