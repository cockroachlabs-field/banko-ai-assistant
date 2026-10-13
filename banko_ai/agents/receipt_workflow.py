"""LangGraph receipt processing workflow with CockroachDB checkpointer.

Defines a multi-agent graph:
    receipt_node -> fraud_node -> budget_node -> done

State is persisted via CockroachDBSaver so the workflow survives
restarts and can be inspected / replayed.
"""

import uuid
from datetime import datetime
from typing import Any, TypedDict

from langchain_cockroachdb import CockroachDBSaver
from langgraph.graph import END, StateGraph

# ---------------------------------------------------------------------------
# Workflow state schema
# ---------------------------------------------------------------------------

class ReceiptWorkflowState(TypedDict, total=False):
    """State that flows through every node of the receipt workflow."""
    # Input
    file_path: str
    user_id: str
    database_url: str
    # Receipt agent output
    extracted: dict[str, Any]
    expense_id: str
    receipt_status: str
    receipt_errors: list[str]
    # Fraud agent output
    fraud_result: str
    fraud_detected: bool
    fraud_confidence: float
    # Budget agent output
    budget_result: str
    budget_status: str
    # Metadata
    workflow_id: str
    started_at: str


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

def receipt_node(state: ReceiptWorkflowState) -> dict:
    """Extract data from the uploaded receipt using the Receipt Agent."""
    from banko_ai.agents.llm_factory import get_embedding_model, get_llm_for_agent
    from banko_ai.agents.receipt_agent import ReceiptAgent

    llm = get_llm_for_agent(temperature=0.7)
    embedding_model = get_embedding_model()

    from banko_ai.db.regions import agent_home_region

    agent = ReceiptAgent(
        region=agent_home_region(state.get("database_url", "")),
        llm=llm,
        database_url=state.get("database_url", ""),
        embedding_model=embedding_model,
    )

    result = agent.process_document(
        file_path=state["file_path"],
        user_id=state.get("user_id", "demo_user"),
        document_type="receipt",
    )

    if not result.get("success", False):
        return {
            "receipt_status": "failed",
            "receipt_errors": result.get("errors", ["Unknown error"]),
            "extracted": {},
        }

    return {
        "receipt_status": "success",
        "receipt_errors": [],
        "extracted": result.get("extracted_fields", {}),
    }


def fraud_node(state: ReceiptWorkflowState) -> dict:
    """Run fraud analysis on the newly created expense."""
    if state.get("receipt_status") != "success":
        return {"fraud_result": "Skipped (receipt failed)", "fraud_detected": False}

    expense_id = state.get("expense_id")
    if not expense_id:
        return {"fraud_result": "Skipped (no expense_id)", "fraud_detected": False}

    from banko_ai.agents.fraud_agent import FraudAgent
    from banko_ai.agents.llm_factory import get_embedding_model, get_llm_for_agent
    from banko_ai.config.settings import get_config

    cfg = get_config()
    llm = get_llm_for_agent(temperature=0.7)
    embedding_model = get_embedding_model()

    from banko_ai.db.regions import agent_home_region

    agent = FraudAgent(
        region=agent_home_region(state.get("database_url", "")),
        llm=llm,
        database_url=state.get("database_url", ""),
        embedding_model=embedding_model,
        fraud_threshold=0.7,
        duplicate_window_days=cfg.fraud_duplicate_window_days,
    )

    check = agent.analyze_expense(expense_id)

    detected = check.get("fraud_detected", False)
    confidence = check.get("confidence", 0.0)
    msg = (
        f"⚠️ Suspicious ({confidence:.0%})" if detected
        else "✅ No issues"
    )
    return {
        "fraud_result": msg,
        "fraud_detected": detected,
        "fraud_confidence": confidence,
    }


def budget_node(state: ReceiptWorkflowState) -> dict:
    """Check budget impact of the new expense."""
    if state.get("receipt_status") != "success":
        return {"budget_result": "Skipped", "budget_status": "unknown"}

    from banko_ai.agents.budget_agent import BudgetAgent
    from banko_ai.agents.llm_factory import get_llm_for_agent
    from banko_ai.config.settings import get_config

    cfg = get_config()
    llm = get_llm_for_agent(temperature=0.7)

    agent = BudgetAgent(
        region="us-central-1",
        llm=llm,
        database_url=state.get("database_url", ""),
        alert_threshold=0.8,
    )

    check = agent.check_budget_status(
        user_id=state.get("user_id", "demo_user"),
        monthly_budget=cfg.monthly_budget_default,
    )

    status = check.get("status", "unknown")
    if status == "over_budget":
        msg = "⚠️ Over budget!"
    elif status == "on_pace_to_exceed":
        msg = "⚠️ On pace to exceed"
    else:
        msg = "✅ Within budget"

    return {"budget_result": msg, "budget_status": status}


# ---------------------------------------------------------------------------
# Graph definition
# ---------------------------------------------------------------------------

def build_receipt_graph() -> StateGraph:
    """Build the receipt-processing LangGraph."""
    graph = StateGraph(ReceiptWorkflowState)

    graph.add_node("receipt", receipt_node)
    graph.add_node("fraud", fraud_node)
    graph.add_node("budget", budget_node)

    graph.set_entry_point("receipt")
    graph.add_edge("receipt", "fraud")
    graph.add_edge("fraud", "budget")
    graph.add_edge("budget", END)

    return graph


def run_receipt_workflow(
    file_path: str,
    user_id: str,
    database_url: str,
    expense_id: str | None = None,
) -> dict[str, Any]:
    """Execute the full receipt workflow synchronously.

    Uses CockroachDBSaver as a context-managed checkpointer so the
    workflow can be inspected, replayed, or resumed after a crash.

    Returns the final state dict.
    """
    from ..utils.db_retry import get_database_url
    url = get_database_url(database_url)

    with CockroachDBSaver.from_conn_string(url) as checkpointer:
        checkpointer.setup()

        from ..config.settings import get_config
        cfg = get_config()
        if cfg.checkpoint_ttl_days > 0:
            checkpointer.enable_ttl(
                ttl_interval=f"{cfg.checkpoint_ttl_days} days", cron="@daily"
            )

        graph = build_receipt_graph()
        workflow = graph.compile(checkpointer=checkpointer)

        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        initial_state: ReceiptWorkflowState = {
            "file_path": file_path,
            "user_id": user_id,
            "database_url": database_url,
            "expense_id": expense_id or "",
            "workflow_id": thread_id,
            "started_at": datetime.utcnow().isoformat(),
        }

        result = workflow.invoke(initial_state, config)
        return dict(result)
