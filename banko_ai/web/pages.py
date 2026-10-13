"""Page routes. Each renders one template from templates/pages/; the page's
Alpine component fetches its data from /api/v1. Sign-in is enforced by the
app's before_request hook."""

from flask import Blueprint, abort, render_template, session

from ..config.catalog import get_catalog
from ..services import accounts as accounts_service
from ..services import threads as threads_service

pages = Blueprint("pages", __name__)


@pages.get("/")
def dashboard():
    return render_template("pages/dashboard.html")


@pages.get("/accounts")
def accounts():
    return render_template("pages/accounts.html")


@pages.get("/accounts/<account_id>")
def account(account_id: str):
    if accounts_service.get_account(session["user_id"], account_id) is None:
        abort(404)
    return render_template("pages/account.html", account_id=account_id)


@pages.get("/transactions")
def transactions():
    return render_template("pages/transactions.html")


@pages.get("/cashflow")
def cashflow():
    return render_template("pages/cashflow.html")


@pages.get("/budget")
def budget():
    return render_template("pages/budget.html")


@pages.get("/recurring")
def recurring():
    return render_template("pages/recurring.html")


@pages.get("/goals")
def goals():
    return render_template("pages/goals.html")


@pages.get("/assistant")
def assistant():
    return render_template("pages/assistant.html", thread_id=None)


@pages.get("/assistant/<thread_id>")
def assistant_thread(thread_id: str):
    if threads_service.get_thread(session["user_id"], thread_id) is None:
        abort(404)
    return render_template("pages/assistant.html", thread_id=thread_id)


@pages.get("/coach")
def coach():
    return render_template("pages/coach.html")


@pages.get("/agents")
def agents():
    return render_template("pages/agents.html")


@pages.get("/settings")
def settings():
    personas = [{"user_id": p.user_id, "username": p.username, "display_name": p.display_name}
                for p in get_catalog().personas]
    return render_template("pages/settings.html", personas=personas)
