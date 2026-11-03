"""The fraud agent writes about a bank customer's purchase, briefly: its
reasoning shows on the Agents page."""

from banko_ai.agents.fraud_agent import reasoning_prompt

EXPENSE = {"merchant": "BLUE BOTTLE COFFEE", "amount": 15.5, "date": "2026-11-02", "category": "Coffee"}


def test_the_prompt_is_about_a_customer_and_asks_for_a_short_answer():
    prompt = reasoning_prompt(EXPENSE, {"signals": [], "confidence": 0.0})
    assert "customer" in prompt.lower()
    assert "expense" not in prompt.lower() and "employee" not in prompt.lower()
    assert "two or three sentences" in prompt
    assert "BLUE BOTTLE COFFEE" in prompt and "$15.5" in prompt and "0.00%" in prompt
