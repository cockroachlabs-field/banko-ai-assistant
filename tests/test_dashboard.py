"""
Test Agent Dashboard with Live Agents.

This script:
1. Starts the Flask app with SocketIO
2. Creates test agents
3. Simulates agent activity
4. Shows real-time updates in dashboard

Run this, then open: http://localhost:5001/agents
"""

import os

import pytest

# Calls a real model, so it runs only when asked for: BANKO_LIVE_MODEL_TESTS=1.
pytestmark = pytest.mark.skipif(os.getenv("BANKO_LIVE_MODEL_TESTS") != "1",
                                reason="calls a real model; set BANKO_LIVE_MODEL_TESTS=1 to run it")
import threading
import time

# Disable tokenizers parallelism warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer

from banko_ai.agents.budget_agent import BudgetAgent
from banko_ai.agents.fraud_agent import FraudAgent
from banko_ai.agents.orchestrator_agent import OrchestratorAgent
from banko_ai.web.app import create_app


def simulate_agent_activity():
    """Simulate agent activity in background"""
    time.sleep(3)  # Wait for app to start
    
    print("\n🤖 Starting agent simulation...")
    
    # Configuration
    database_url = os.getenv(
        'DATABASE_URL',
        'cockroachdb://root@localhost:26257/defaultdb?sslmode=disable'
    )
    openai_api_key = os.getenv('OPENAI_API_KEY')
    
    if not openai_api_key:
        print("❌ OPENAI_API_KEY not set")
        return
    
    # Initialize models
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=openai_api_key,
        temperature=0.7
    )
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create agents
    print("   Creating agents...")
    fraud_agent = FraudAgent(
        region="us-west-2",
        llm=llm,
        database_url=database_url,
        embedding_model=embedding_model,
        fraud_threshold=0.7
    )
    
    budget_agent = BudgetAgent(
        region="us-central-1",
        llm=llm,
        database_url=database_url,
        alert_threshold=0.8
    )
    
    orchestrator = OrchestratorAgent(
        region="us-east-1",
        llm=llm,
        database_url=database_url
    )
    orchestrator.register_agent('fraud', fraud_agent)
    orchestrator.register_agent('budget', budget_agent)
    
    print("   ✅ Agents created and registered")
    print("\n📊 Watch the dashboard at: http://localhost:5001/agents")
    print("   (Refresh to see agent status updates)")
    print()
    
    # Get a sample user
    from sqlalchemy import create_engine, text
    from sqlalchemy.pool import NullPool
    
    engine = create_engine(database_url, poolclass=NullPool)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT DISTINCT user_id FROM expenses LIMIT 1"))
        row = result.fetchone()
        user_id = str(row[0]) if row else "user_01"
    engine.dispose()
    
    # Simulate workflows with delays
    try:
        # Test 1: Fraud scan
        print("\n1️⃣  Running fraud scan...")
        fraud_result = fraud_agent.scan_recent_expenses(hours=24, limit=5)
        print(f"   ✅ Fraud scan complete: {fraud_result.get('total_analyzed', 0)} expenses analyzed")
        time.sleep(3)
        
        # Test 2: Budget check
        print("\n2️⃣  Checking budget...")
        budget_result = budget_agent.check_budget_status(
            user_id=user_id,
            monthly_budget=1000.00
        )
        print(f"   ✅ Budget check complete: Status = {budget_result.get('status', 'unknown')}")
        time.sleep(3)
        
        # Test 3: Complex workflow
        print("\n3️⃣  Running complex workflow...")
        workflow_result = orchestrator.execute_workflow(
            user_request="Check for fraud and verify my budget",
            context={'user_id': user_id, 'monthly_budget': 1000.00}
        )
        print(f"   ✅ Workflow complete: {len(workflow_result['steps_executed'])} steps executed")
        
        print("\n🎉 Simulation complete! Check the dashboard for real-time updates.")
        print("   Press Ctrl+C to stop the server.")
        
    except Exception as e:
        print(f"\n⚠️  Error during simulation: {e}")


if __name__ == "__main__":
    print("🚀 Starting Banko AI with Agent Dashboard")
    print("="*70)
    
    # Start agent simulation in background thread
    simulator_thread = threading.Thread(target=simulate_agent_activity, daemon=True)
    simulator_thread.start()
    
    # Create and run app
    app = create_app()
    socketio = app.socketio
    
    print("\n✅ Server starting...")
    print("   Main App: http://localhost:5001/")
    print("   Dashboard: http://localhost:5001/agents")
    print("   API Status: http://localhost:5001/api/agents/status")
    print("\n   Press Ctrl+C to stop\n")
    
    # Run with SocketIO
    try:
        socketio.run(app, host='0.0.0.0', port=5001, debug=False, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
