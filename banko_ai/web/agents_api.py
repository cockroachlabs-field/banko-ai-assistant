"""What the agents did for the signed-in user."""

from flask import Blueprint, jsonify, session

from ..config.settings import get_config
from ..services import agent_feed

agents_api = Blueprint("agents_api", __name__, url_prefix="/api/v1/agents")


def _uid() -> str:
    return session.get("user_id") or get_config().coach_default_user_id


@agents_api.after_request
def _name_the_user(response):
    response.headers["X-Banko-User"] = _uid()
    return response


@agents_api.get("/activity")
def activity():
    return jsonify({"items": agent_feed.activity(_uid())})


@agents_api.get("/stats")
def stats():
    return jsonify(agent_feed.stats(_uid()))
