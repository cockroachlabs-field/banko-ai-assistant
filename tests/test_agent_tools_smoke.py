"""Smoke tests for agent tool modules to catch import and basic execution errors.

These tests verify that tool factories can be instantiated and their tools
can be invoked without import errors or basic execution failures. They use
read-only operations against the test database."""

import json
import os

import pytest

DATABASE_URL = os.getenv("DATABASE_URL")


@pytest.mark.skipif(not DATABASE_URL, reason="DATABASE_URL not set")
def test_search_tools_can_import_and_execute():
    """Search tools factory works and get_expense_by_id runs without import error."""
    from banko_ai.agents.tools.search_tools import create_search_tools

    # Create tools (embedding_model can be None for SQL-only tools like get_expense_by_id)
    tools = create_search_tools(DATABASE_URL, None)
    assert len(tools) > 0, "Should create at least one search tool"

    # Find get_expense_by_id tool
    get_expense_tool = next((t for t in tools if "expense_by_id" in t.name.lower()), None)
    assert get_expense_tool is not None, "Should have get_expense_by_id tool"

    # Invoke with a non-existent ID (won't find anything but should not error on import)
    result_str = get_expense_tool.func(
        expense_id="00000000-0000-0000-0000-000000000000"
    )
    result = json.loads(result_str)

    # Should not have an import error
    if not result.get("success", True):
        error = result.get("error", "")
        assert "No module named" not in error, f"Import error: {error}"
        assert "banko_ai.agents.db" not in error, f"Wrong import path: {error}"


@pytest.mark.skipif(not DATABASE_URL, reason="DATABASE_URL not set")
def test_analysis_tools_can_import_and_execute():
    """Analysis tools factory works and calculate_statistics runs without import error."""
    from banko_ai.agents.tools.analysis_tools import create_analysis_tools

    tools = create_analysis_tools(DATABASE_URL)
    assert len(tools) > 0, "Should create at least one analysis tool"

    # Find calculate_statistics tool
    stats_tool = next((t for t in tools if "statistic" in t.name.lower()), None)
    assert stats_tool is not None, "Should have calculate_statistics tool"

    # Invoke with a non-existent user (won't find anything but should not error on import)
    result_str = stats_tool.func(user_id="00000000-0000-0000-0000-000000000000")
    result = json.loads(result_str)

    # Should not have an import error
    if not result.get("success", True):
        error = result.get("error", "")
        assert "No module named" not in error, f"Import error: {error}"
        assert "banko_ai.agents.db" not in error, f"Wrong import path: {error}"


@pytest.mark.skipif(not DATABASE_URL, reason="DATABASE_URL not set")
def test_document_tools_can_import():
    """Document tools factory can be imported and instantiated."""
    from banko_ai.agents.tools.document_tools import create_document_tools

    # Document tools need OCR dependencies which may not be available,
    # so just verify the import works and factory returns tools
    tools = create_document_tools(DATABASE_URL, None)
    assert isinstance(tools, list), "Should return a list of tools"
    # May be empty if OCR deps not available, but import should work


def test_the_receipt_prompt_offers_the_catalog_labels():
    from banko_ai.agents.tools import document_tools

    class Stub:
        prompt = ""

        def invoke(self, prompt):
            Stub.prompt = prompt

            class Message:
                content = '{"merchant": "Blue Bottle", "amount": 13.58, "category": "Coffee"}'
            return Message()

    out = document_tools.parse_receipt_fields("BLUE BOTTLE COFFEE 13.58", Stub())
    assert out["success"] and out["fields"]["category"] == "Coffee"
    assert "one of: Housing, Groceries" in Stub.prompt and "food or transportation" not in Stub.prompt
    custom = document_tools.parse_receipt_fields("X 1.00", Stub(), categories=["Snacks", "Rent"])
    assert custom["success"] and "one of: Snacks, Rent" in Stub.prompt


def test_create_document_tools_has_no_second_receipt_parser():
    import inspect

    from banko_ai.agents.tools import document_tools
    assert "def parse_receipt_fields" not in inspect.getsource(document_tools.create_document_tools)
