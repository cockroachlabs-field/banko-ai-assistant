"""Page routes. Each renders one template from templates/pages/; the page's
Alpine component fetches its data from /api/v1. Sign-in is enforced by the
app's before_request hook."""

from flask import Blueprint

pages = Blueprint("pages", __name__)
