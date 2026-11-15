"""The fraud agent's checks answer on a customer's real history. A check
that fails raises no signal, so a broken one reads exactly like a clean
purchase: the outlier check failed on every call for as long as amounts
were bound as floats, and nothing showed it."""

import json
import os
from datetime import date

import pytest

from banko_ai.services.seeding import seed_user
from tests._money import card_purchase, drop_user, make_user, splurge

pytestmark = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")


@pytest.fixture(scope="module")
def member():
    uid = make_user("diner")
    seed_user(uid, months=1, today=date.today())
    yield uid
    drop_user(uid)


@pytest.fixture(scope="module")
def tools():
    from banko_ai.agents.tools.analysis_tools import create_analysis_tools
    from banko_ai.agents.tools.search_tools import create_search_tools
    from banko_ai.config.settings import get_config
    from banko_ai.utils.embeddings import load_embedding_model
    url = get_config().database_url
    return {t.name: t.func for t in create_search_tools(url, load_embedding_model()) + create_analysis_tools(url)}


def test_the_outlier_check_finds_a_purchase_far_above_the_customers_usual(member, tools):
    expense_id = card_purchase(member, "Rare Splurge Outfitters", splurge(member), "Shopping")
    found = json.loads(tools["detect_anomalies"](user_id=member, threshold=2.0))
    assert found["success"] is True, found.get("error")
    mine = next(a for a in found["anomalies"] if a["expense_id"] == expense_id)
    assert mine["type"] == "unusually_high" and mine["z_score"] > 2
