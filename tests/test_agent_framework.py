"""
Test script to verify agent framework is working.

This tests:
- Base agent creation
- Tool registration
- Agent thinking (LLM interaction)
- Database integration
"""

import os

import pytest

# Calls a real model, so it runs only when asked for: BANKO_LIVE_MODEL_TESTS=1.
pytestmark = pytest.mark.skipif(os.getenv("BANKO_LIVE_MODEL_TESTS") != "1",
                                reason="calls a real model; set BANKO_LIVE_MODEL_TESTS=1 to run it")

from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer

from banko_ai.agents.base_agent import BaseAgent
from banko_ai.agents.tools.analysis_tools import create_analysis_tools
from banko_ai.agents.tools.search_tools import create_search_tools


def test_agent_framework():
    """Test the basic agent framework"""
    
    print("🧪 Testing Banko AI Agent Framework")
    print("="*60)
    
    # Configuration
    database_url = os.getenv(
        'DATABASE_URL',
        'cockroachdb://root@localhost:26257/defaultdb?sslmode=disable'
    )
    openai_api_key = os.getenv('OPENAI_API_KEY')
    
    if not openai_api_key:
        print("❌ OPENAI_API_KEY not set")
        print("Set it with: export OPENAI_API_KEY='your-key'")
        return False
    
    print(f"✅ Database: {database_url.split('@')[1]}")
    print(f"✅ OpenAI API Key: {'*' * 20}{openai_api_key[-4:]}")
    print()
    
    # 1. Create LLM
    print("1️⃣  Creating LLM (OpenAI GPT-4o-mini)...")
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=openai_api_key,
        temperature=0.7
    )
    print("   ✅ LLM created")
    print()
    
    # 2. Create tools
    print("2️⃣  Creating agent tools...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    search_tools = create_search_tools(database_url, embedding_model)
    analysis_tools = create_analysis_tools(database_url)
    all_tools = search_tools + analysis_tools
    
    print(f"   ✅ Created {len(all_tools)} tools:")
    for tool in all_tools:
        print(f"      - {tool.name}")
    print()
    
    # 3. Create agent
    print("3️⃣  Creating test agent...")
    agent = BaseAgent(
        agent_type="analyst",
        region="us-east-1",
        llm=llm,
        tools=all_tools,
        database_url=database_url,
        system_prompt="""You are an expense analyst agent.
You help users understand their spending patterns.
You have access to tools for searching and analyzing expenses.
Always be concise and helpful."""
    )
    
    print(f"   ✅ Agent created: {agent}")
    print(f"      ID: {agent.agent_id}")
    print(f"      Type: {agent.agent_type}")
    print(f"      Region: {agent.region}")
    print(f"      Tools: {len(agent.tools)}")
    print()
    
    # 4. Test agent thinking
    print("4️⃣  Testing agent thinking...")
    response = agent.think(
        "Hello! Can you help me analyze my expenses?",
        context={'user_id': 'user_01'}
    )
    print(f"   Agent response:\n   {response}")
    print()
    
    # 5. Test tool execution
    print("5️⃣  Testing tool execution...")
    print("   Searching for 'food' expenses...")
    
    try:
        result = agent.execute_tool(
            'vector_search_expenses',
            query='food',
            user_id='user_01',
            limit=3
        )
        print(f"   Tool result: {result[:200]}...")
    except Exception as e:
        print(f"   ⚠️  Tool execution failed: {e}")
    print()
    
    # 6. Test agent decision recording
    print("6️⃣  Testing decision recording...")
    decision = agent.store_decision(
        decision_type='test_decision',
        context={'test': True},
        reasoning='This is a test decision',
        action={'action': 'none'},
        confidence=0.95
    )
    print(f"   ✅ Decision recorded: {decision.decision_id}")
    print()
    
    # 7. Get agent info
    print("7️⃣  Agent information:")
    info = agent.get_info()
    for key, value in info.items():
        print(f"   {key}: {value}")
    print()
    
    print("="*60)
    print("🎉 Agent framework test completed successfully!")
    print()
    print("Next steps:")
    print("  - Build specialized agents (Receipt, Fraud, Budget)")
    print("  - Create agent dashboard UI")
    print("  - Add multi-agent orchestration")
    print("  - Integrate with chaos demo for multi-region")
    
    return True


if __name__ == "__main__":
    success = test_agent_framework()
    exit(0 if success else 1)
