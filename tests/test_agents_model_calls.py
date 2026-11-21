"""The agents behind a receipt ask the model only when there is something to
explain. A purchase with no fraud signal and a routine budget check cost no
tokens: what they report is computed, so it is written in code, from the
checks that ran. A check that could not run is named, never counted as clean."""

import json
import os
from datetime import date

import pytest
from langchain_core.messages import AIMessage
from sqlalchemy import text

from banko_ai.db.engine import get_engine
from banko_ai.services.seeding import seed_user
from tests._money import card_purchase, drop_user, make_user, splurge

pytestmark = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")

AGENTS: list[str] = []  # every agent a test registered, removed with the module's user


class CountingModel:
    """A LangChain-shaped model that counts what it is asked."""

    def __init__(self, reply: str = "The model's own explanation."):
        self.reply, self.asked = reply, []

    def invoke(self, messages):
        self.asked.append(messages)
        return AIMessage(content=self.reply)


class PartsModel:
    """Gemini and Bedrock answer in content parts, not in one string."""

    def invoke(self, messages):
        return AIMessage(content=[{"type": "text", "text": "Two identical charges "},
                                  {"type": "text", "text": "point to a duplicate."}])


class DownModel:
    """A provider that cannot answer, and says why in words no page should show."""

    def invoke(self, messages):
        raise RuntimeError("403 token_quota_reached for https://provider.example/v1 (request 8f2c)")


def _forget_agents(*user_ids: str) -> None:
    with get_engine().begin() as c:
        c.execute(text("DELETE FROM agent_tasks WHERE source_agent_id = ANY(CAST(:a AS UUID[])) "
                       "OR target_agent_id = ANY(CAST(:a AS UUID[]))"), {"a": AGENTS})
        c.execute(text("DELETE FROM agent_decisions WHERE agent_id = ANY(CAST(:a AS UUID[]))"), {"a": AGENTS})
        c.execute(text("DELETE FROM agent_memory WHERE agent_id = ANY(CAST(:a AS UUID[])) OR user_id = ANY(:u)"),
                  {"a": AGENTS, "u": list(user_ids)})
        c.execute(text("DELETE FROM agent_state WHERE agent_id = ANY(CAST(:a AS UUID[]))"), {"a": AGENTS})
    AGENTS.clear()


@pytest.fixture(scope="module")
def member():
    uid = make_user("diner")
    seed_user(uid, months=1, today=date.today())
    yield uid
    _forget_agents(uid)
    drop_user(uid)


def _fraud_agent(model):
    from banko_ai.agents.fraud_agent import FraudAgent
    from banko_ai.config.settings import get_config
    from banko_ai.utils.embeddings import load_embedding_model
    agent = FraudAgent(region="test", llm=model, database_url=get_config().database_url,
                       embedding_model=load_embedding_model(), fraud_threshold=0.7, duplicate_window_days=3)
    AGENTS.append(str(agent.agent_id))
    return agent


def _tool_answers(agent, **answers):
    """Make named tools return a fixed answer; the rest run for real."""
    run = agent.execute_tool
    agent.execute_tool = lambda name, **kwargs: json.dumps(answers[name]) if name in answers else run(name, **kwargs)


FAILED = {"success": False, "error": "the database did not answer"}


def _last_decision(agent) -> str:
    with get_engine().connect() as c:
        return str(c.execute(text("SELECT reasoning FROM agent_decisions WHERE agent_id = CAST(:a AS UUID) "
                                  "ORDER BY created_at DESC LIMIT 1"), {"a": agent.agent_id}).scalar())


CLEAN = ("No fraud signal for the customer's $6.40 purchase at Quiet Corner Cafe: no other charge of that amount "
         "there on the same day, not one of the customer's outlier amounts, no close match in the similarity search, "
         "and not above the $1,000 high-value mark.")


def test_a_purchase_with_no_fraud_signal_is_explained_without_the_model(member):
    model = CountingModel()
    agent = _fraud_agent(model)
    check = agent.analyze_expense(card_purchase(member, "Quiet Corner Cafe", "6.40"))
    assert check["signals"] == [] and check["recommendation"] == "approve"
    assert model.asked == []
    assert check["reasoning"] == CLEAN  # every check ran and found nothing, and the line says which
    assert check["unchecked"] == []
    assert _last_decision(agent) == check["reasoning"]  # the Agents page shows this


def test_the_line_for_a_clean_purchase_names_only_the_checks_that_ran():
    from banko_ai.agents.fraud_agent import no_signal_reasoning
    expense = {"merchant": "Quiet Corner Cafe", "amount": 6.4}
    assert no_signal_reasoning(expense, {"duplicates": True, "outliers": True, "similar": True}) == CLEAN
    # What could not be checked comes first, where a line cut short still has it.
    assert no_signal_reasoning(expense, {"duplicates": True, "outliers": False, "similar": False}) == (
        "Not checked this time: outlier amounts and similar purchases. The other checks found no fraud signal for "
        "the customer's $6.40 purchase at Quiet Corner Cafe: no other charge of that amount there on the same day "
        "and not above the $1,000 high-value mark.")
    assert no_signal_reasoning({"merchant": None, "amount": 6.4}, {"duplicates": True, "outliers": True, "similar": True}) == (
        CLEAN.replace("Quiet Corner Cafe", "an unnamed merchant"))
    assert no_signal_reasoning(expense, {}) == (
        "Not checked this time: duplicate charges, outlier amounts, and similar purchases. The other checks found "
        "no fraud signal for the customer's $6.40 purchase at Quiet Corner Cafe: not above the $1,000 high-value mark.")


@pytest.mark.parametrize("tool, missed, cleared, amount", [
    ("vector_search_expenses", "similar purchases", "no close match in the similarity search", "7.15"),
    ("detect_anomalies", "outlier amounts", "not one of the customer's outlier amounts", "7.16"),
])
def test_a_check_that_could_not_run_is_not_reported_as_clean(member, tool, missed, cleared, amount):
    model = CountingModel()
    agent = _fraud_agent(model)
    _tool_answers(agent, **{tool: FAILED})
    check = agent.analyze_expense(card_purchase(member, "Quiet Corner Cafe", amount))
    assert check["signals"] == [] and model.asked == []
    assert check["reasoning"].startswith(f"Not checked this time: {missed}. The other checks found no fraud signal")
    assert cleared not in check["reasoning"]
    assert check["unchecked"] == [missed]


def test_too_little_history_to_compare_is_not_reported_as_no_outlier(member):
    """One purchase has no spread to be an outlier of; the tool says so and compares nothing."""
    agent = _fraud_agent(CountingModel())
    _tool_answers(agent, detect_anomalies={"success": True, "anomalies": [],
                                           "message": "Not enough data variation to detect anomalies"})
    check = agent.analyze_expense(card_purchase(member, "Quiet Corner Cafe", "7.25"))
    assert check["unchecked"] == ["outlier amounts"]
    assert "not one of the customer's outlier amounts" not in check["reasoning"]


def test_the_duplicate_check_counts_when_either_of_its_two_lookups_ran(member):
    def unreachable(*args, **kwargs):
        raise RuntimeError("the database did not answer")

    # The agent's own same-day lookup fails; the wider scan still ran.
    agent = _fraud_agent(CountingModel())
    agent._same_day_duplicates = unreachable
    check = agent.analyze_expense(card_purchase(member, "Quiet Corner Cafe", "7.35"))
    assert check["unchecked"] == [] and "no other charge of that amount there on the same day" in check["reasoning"]

    # Both fail: nothing looked for a duplicate, and the line says so.
    agent = _fraud_agent(CountingModel())
    agent._same_day_duplicates = unreachable
    _tool_answers(agent, find_duplicates=FAILED)
    check = agent.analyze_expense(card_purchase(member, "Quiet Corner Cafe", "7.45"))
    assert check["unchecked"] == ["duplicate charges"]
    assert check["reasoning"].startswith("Not checked this time: duplicate charges.")
    assert "no other charge of that amount" not in check["reasoning"]


def test_a_purchase_with_a_fraud_signal_is_explained_by_the_model(member):
    model = CountingModel("Two identical charges on one day point to a duplicate.")
    agent = _fraud_agent(model)
    card_purchase(member, "Twice Charged Diner", "41.07")
    check = agent.analyze_expense(card_purchase(member, "Twice Charged Diner", "41.07"))
    assert [s["type"] for s in check["signals"]] == ["duplicate"]
    assert len(model.asked) == 1
    assert check["reasoning"] == "Two identical charges on one day point to a duplicate."


def test_a_flagged_purchase_still_names_a_check_that_could_not_run(member):
    agent = _fraud_agent(CountingModel("Two identical charges on one day point to a duplicate."))
    _tool_answers(agent, detect_anomalies=FAILED)
    card_purchase(member, "Charged Twice Again", "38.11")
    check = agent.analyze_expense(card_purchase(member, "Charged Twice Again", "38.11"))
    assert check["reasoning"] == ("Not checked this time: outlier amounts. "
                                  "Two identical charges on one day point to a duplicate.")


def test_a_model_that_cannot_answer_leaves_the_signals_not_its_error(member):
    """With the provider down, the Agents page gets the signals the checks
    found. It used to get the provider's error text."""
    agent = _fraud_agent(DownModel())
    card_purchase(member, "Charged Twice Offline", "52.19")
    check = agent.analyze_expense(card_purchase(member, "Charged Twice Offline", "52.19"))
    assert check["reasoning"] == ("Flagged on the customer's $52.19 purchase at Charged Twice Offline: Duplicate "
                                  "transaction: 2 occurrences of $52.19 at Charged Twice Offline. Fraud confidence 40%.")
    assert _last_decision(agent) == check["reasoning"]


def test_a_purchase_far_above_the_customers_usual_is_explained_by_the_model(member):
    model = CountingModel("Far above what this customer usually spends.")
    agent = _fraud_agent(model)
    check = agent.analyze_expense(card_purchase(member, "Rare Splurge Outfitters", splurge(member), "Shopping"))
    assert "anomaly" in [s["type"] for s in check["signals"]]
    assert len(model.asked) == 1 and check["reasoning"] == "Far above what this customer usually spends."


def test_an_outlier_is_flagged_even_when_twenty_larger_ones_exist():
    """The outlier tool lists a customer's twenty most unusual purchases. A
    customer with more than that still has outliers below them, and the agent
    used to call those clean because they were not on the list."""
    uid = make_user("diner")
    try:
        with get_engine().begin() as c:
            c.execute(text("""
                INSERT INTO expenses (user_id, expense_date, expense_amount, shopping_type, description, merchant,
                                      payment_method, recurring, tags)
                SELECT CAST(:u AS UUID), current_date - (g % 300), CASE WHEN g <= 21 THEN 1000 ELSE 50 END, 'Other',
                       'history', 'Everyday Shop', 'Credit Card', false, ARRAY['other']
                FROM generate_series(1, 221) AS g"""), {"u": uid})
            mean, spread = c.execute(text("SELECT avg(expense_amount), stddev(expense_amount) FROM expenses "
                                          "WHERE user_id = CAST(:u AS UUID)"), {"u": uid}).one()
            amount = round(mean + spread * 23 / 10, 2)  # past two spreads, under the twenty-one at $1,000
            assert amount < 1000
            expense_id = str(c.execute(text("""
                INSERT INTO expenses (user_id, expense_date, expense_amount, shopping_type, description, merchant,
                                      payment_method, recurring, tags)
                VALUES (CAST(:u AS UUID), current_date, :amt, 'Other', 'the one checked', 'Quiet Outlier Shop',
                        'Credit Card', false, ARRAY['other']) RETURNING expense_id"""), {"u": uid, "amt": amount}).scalar())
        agent = _fraud_agent(CountingModel("Above this customer's usual amounts."))
        check = agent.analyze_expense(expense_id)
        assert "anomaly" in [s["type"] for s in check["signals"]], check["reasoning"]
    finally:
        _forget_agents(uid)
        drop_user(uid)


def test_a_model_that_answers_with_nothing_leaves_the_signals(member):
    agent = _fraud_agent(CountingModel("  "))
    card_purchase(member, "Charged Twice Silently", "27.63")
    check = agent.analyze_expense(card_purchase(member, "Charged Twice Silently", "27.63"))
    assert check["reasoning"].startswith("Flagged on the customer's $27.63 purchase at Charged Twice Silently: Duplicate")


def test_an_explanation_sent_in_parts_is_read_as_its_text(member):
    agent = _fraud_agent(PartsModel())
    card_purchase(member, "Charged Twice In Parts", "19.84")
    check = agent.analyze_expense(card_purchase(member, "Charged Twice In Parts", "19.84"))
    assert check["reasoning"] == "Two identical charges point to a duplicate."


def test_a_budget_check_asks_the_model_nothing(member):
    """Its status and advice are computed. The text the model was asked for
    on top of them went, cut to 200 characters, into a memory row that
    nothing reads; the computed advice goes there now."""
    from banko_ai.agents.budget_agent import BudgetAgent
    from banko_ai.config.settings import get_config
    model = CountingModel()
    agent = BudgetAgent(region="test", llm=model, database_url=get_config().database_url, alert_threshold=0.8)
    AGENTS.append(str(agent.agent_id))
    status = agent.check_budget_status(user_id=member, monthly_budget=3000.0)
    assert status["status"] in {"on_track", "on_pace_to_exceed", "approaching_limit", "over_budget"}
    assert status["recommendation"]  # the computed advice is still there
    assert model.asked == [] and "ai_insights" not in status
    with get_engine().connect() as c:
        remembered = str(c.execute(text("SELECT content FROM agent_memory WHERE agent_id = CAST(:a AS UUID) "
                                        "ORDER BY created_at DESC LIMIT 1"), {"a": agent.agent_id}).scalar())
    assert status["recommendation"][0] in remembered
