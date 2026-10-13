"""Page routes. Each renders one template from templates/pages/; the page's
Alpine component fetches its data from /api/v1. Sign-in is enforced by the
app's before_request hook."""

from flask import Blueprint, abort, render_template, session

from ..services import accounts as accounts_service

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
