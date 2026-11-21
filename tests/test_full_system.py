"""
Comprehensive System Test - All Components

Tests the complete agentic AI system:
1. Database connectivity
2. All 4 agents (Receipt, Fraud, Budget, Orchestrator)
3. Real-time dashboard
4. Multi-agent workflows
5. Decision tracking
6. WebSocket events

Run this to verify everything works before the demo.
"""

import os

import pytest

# Calls a real model, so it runs only when asked for: BANKO_LIVE_MODEL_TESTS=1.
pytestmark = pytest.mark.skipif(os.getenv("BANKO_LIVE_MODEL_TESTS") != "1",
                                reason="calls a real model; set BANKO_LIVE_MODEL_TESTS=1 to run it")
import sys
import time
from datetime import datetime

# Disable tokenizers parallelism warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

from banko_ai.agents.budget_agent import BudgetAgent
from banko_ai.agents.fraud_agent import FraudAgent
from banko_ai.agents.orchestrator_agent import OrchestratorAgent
from banko_ai.agents.receipt_agent import ReceiptAgent


class SystemTester:
    """Comprehensive system tester"""
    
    def __init__(self):
        self.database_url = os.getenv(
            'DATABASE_URL',
            'cockroachdb://root@localhost:26257/defaultdb?sslmode=disable'
        )
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.tests_passed = 0
        self.tests_failed = 0
        self.start_time = datetime.now()
    
    def log_test(self, name, passed, message=""):
        """Log test result"""
        if passed:
            print(f"   ✅ {name}")
            self.tests_passed += 1
        else:
            print(f"   ❌ {name}: {message}")
            self.tests_failed += 1
    
    def test_database(self):
        """Test 1: Database connectivity"""
        print("\n1️⃣  Testing Database Connectivity...")
        
        try:
            engine = create_engine(self.database_url, poolclass=NullPool)
            
            with engine.connect() as conn:
                # Test basic query
                result = conn.execute(text("SELECT 1"))
                self.log_test("Database connection", True)
                
                # Test agent tables
                result = conn.execute(text("""
                    SELECT table_name FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name LIKE 'agent%'
                """))
                tables = [row[0] for row in result.fetchall()]
                self.log_test(f"Agent tables ({len(tables)} found)", len(tables) >= 5)
                
                # Test expenses table
                result = conn.execute(text("SELECT COUNT(*) FROM expenses"))
                count = result.fetchone()[0]
                self.log_test(f"Expenses table ({count} records)", count > 0)
            
            engine.dispose()
            return True
            
        except Exception as e:
            self.log_test("Database connectivity", False, str(e))
            return False
    
    def test_models(self):
        """Test 2: AI models initialization"""
        print("\n2️⃣  Testing AI Models...")
        
        try:
            if not self.openai_api_key:
                self.log_test("OpenAI API key", False, "OPENAI_API_KEY not set")
                return False
            
            # Test LLM
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                api_key=self.openai_api_key,
                temperature=0.7
            )
            self.log_test("LLM initialization", True)
            
            # Test embedding model
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.log_test("Embedding model initialization", True)
            
            # Test embedding generation
            test_embedding = embedding_model.encode("test")
            self.log_test(f"Embedding generation ({len(test_embedding)} dims)", len(test_embedding) == 384)
            
            self.llm = llm
            self.embedding_model = embedding_model
            return True
            
        except Exception as e:
            self.log_test("AI models", False, str(e))
            return False
    
    def test_agents(self):
        """Test 3: Individual agents"""
        print("\n3️⃣  Testing Individual Agents...")
        
        try:
            # Create Receipt Agent
            receipt_agent = ReceiptAgent(
                region="us-east-1",
                llm=self.llm,
                database_url=self.database_url,
                embedding_model=self.embedding_model
            )
            self.log_test(f"Receipt Agent ({receipt_agent.agent_id[:8]}...)", True)
            
            # Create Fraud Agent
            fraud_agent = FraudAgent(
                region="us-west-2",
                llm=self.llm,
                database_url=self.database_url,
                embedding_model=self.embedding_model,
                fraud_threshold=0.7
            )
            self.log_test(f"Fraud Agent ({fraud_agent.agent_id[:8]}...)", True)
            
            # Create Budget Agent
            budget_agent = BudgetAgent(
                region="us-central-1",
                llm=self.llm,
                database_url=self.database_url,
                alert_threshold=0.8
            )
            self.log_test(f"Budget Agent ({budget_agent.agent_id[:8]}...)", True)
            
            self.receipt_agent = receipt_agent
            self.fraud_agent = fraud_agent
            self.budget_agent = budget_agent
            return True
            
        except Exception as e:
            self.log_test("Agent creation", False, str(e))
            return False
    
    def test_fraud_detection(self):
        """Test 4: Fraud detection workflow"""
        print("\n4️⃣  Testing Fraud Detection...")
        
        try:
            # Scan recent expenses
            result = self.fraud_agent.scan_recent_expenses(hours=24, limit=5)
            
            self.log_test("Fraud scan executed", result.get('success', False))
            self.log_test(f"Analyzed {result.get('total_analyzed', 0)} expenses", 
                         result.get('total_analyzed', 0) > 0)
            
            flagged = result.get('total_flagged', 0)
            self.log_test(f"Found {flagged} suspicious transactions", True)
            
            return True
            
        except Exception as e:
            self.log_test("Fraud detection", False, str(e))
            return False
    
    def test_budget_monitoring(self):
        """Test 5: Budget monitoring"""
        print("\n5️⃣  Testing Budget Monitoring...")
        
        try:
            # Get a sample user
            engine = create_engine(self.database_url, poolclass=NullPool)
            with engine.connect() as conn:
                result = conn.execute(text("SELECT DISTINCT user_id FROM expenses LIMIT 1"))
                row = result.fetchone()
                user_id = str(row[0]) if row else "user_01"
            engine.dispose()
            
            # Check budget status
            result = self.budget_agent.check_budget_status(
                user_id=user_id,
                monthly_budget=1000.00
            )
            
            self.log_test("Budget check executed", result.get('success', False))
            self.log_test(f"Status: {result.get('status', 'unknown')}", True)
            self.log_test(f"Spent: ${result.get('spent', 0):.2f}", True)
            
            return True
            
        except Exception as e:
            self.log_test("Budget monitoring", False, str(e))
            return False
    
    def test_orchestrator(self):
        """Test 6: Multi-agent orchestration"""
        print("\n6️⃣  Testing Orchestrator...")
        
        try:
            # Create Orchestrator
            orchestrator = OrchestratorAgent(
                region="us-east-1",
                llm=self.llm,
                database_url=self.database_url
            )
            self.log_test(f"Orchestrator created ({orchestrator.agent_id[:8]}...)", True)
            
            # Register agents
            orchestrator.register_agent('fraud', self.fraud_agent)
            orchestrator.register_agent('budget', self.budget_agent)
            orchestrator.register_agent('receipt', self.receipt_agent)
            self.log_test("Agents registered", True)
            
            # Get sample user
            engine = create_engine(self.database_url, poolclass=NullPool)
            with engine.connect() as conn:
                result = conn.execute(text("SELECT DISTINCT user_id FROM expenses LIMIT 1"))
                row = result.fetchone()
                user_id = str(row[0]) if row else "user_01"
            engine.dispose()
            
            # Execute workflow
            workflow_result = orchestrator.execute_workflow(
                user_request="Check my budget status",
                context={'user_id': user_id, 'monthly_budget': 1000.00}
            )
            
            self.log_test("Workflow planning", workflow_result.get('plan') is not None)
            self.log_test(f"Steps executed ({len(workflow_result.get('steps_executed', []))})", 
                         len(workflow_result.get('steps_executed', [])) > 0)
            self.log_test("Workflow success", workflow_result.get('success', False))
            
            return True
            
        except Exception as e:
            self.log_test("Orchestrator", False, str(e))
            return False
    
    def test_decision_tracking(self):
        """Test 7: Decision tracking and audit trail"""
        print("\n7️⃣  Testing Decision Tracking...")
        
        try:
            engine = create_engine(self.database_url, poolclass=NullPool)
            
            with engine.connect() as conn:
                # Count total decisions
                result = conn.execute(text("SELECT COUNT(*) FROM agent_decisions"))
                total_decisions = result.fetchone()[0]
                self.log_test(f"Total decisions recorded ({total_decisions})", total_decisions > 0)
                
                # Get recent decisions
                result = conn.execute(text("""
                    SELECT d.decision_type, d.confidence, a.agent_type, a.region
                    FROM agent_decisions d
                    JOIN agent_state a ON d.agent_id = a.agent_id
                    ORDER BY d.created_at DESC
                    LIMIT 5
                """))
                
                recent = result.fetchall()
                self.log_test(f"Recent decisions ({len(recent)} found)", len(recent) > 0)
                
                # Print recent decisions
                for dec in recent[:3]:
                    decision_type, confidence, agent_type, region = dec
                    print(f"      • {agent_type} ({region}): {decision_type} [{int(confidence*100)}%]")
            
            engine.dispose()
            return True
            
        except Exception as e:
            self.log_test("Decision tracking", False, str(e))
            return False
    
    def test_dashboard_api(self):
        """Test 8: Dashboard API endpoints"""
        print("\n8️⃣  Testing Dashboard API...")
        
        try:
            import requests
            
            # Test agent status endpoint
            try:
                response = requests.get('http://localhost:5001/api/agents/status', timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    agent_count = data.get('count', 0)
                    self.log_test(f"Status API ({agent_count} agents)", True)
                else:
                    self.log_test("Status API", False, f"HTTP {response.status_code}")
            except requests.exceptions.ConnectionError:
                self.log_test("Status API", False, "Server not running on :5001")
            
            # Test activity endpoint
            try:
                response = requests.get('http://localhost:5001/api/agents/activity', timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    activity_count = data.get('count', 0)
                    self.log_test(f"Activity API ({activity_count} activities)", True)
                else:
                    self.log_test("Activity API", False, f"HTTP {response.status_code}")
            except requests.exceptions.ConnectionError:
                self.log_test("Activity API", False, "Server not running on :5001")
            
            return True
            
        except ImportError:
            self.log_test("Dashboard API", False, "requests library not installed")
            return False
        except Exception as e:
            self.log_test("Dashboard API", False, str(e))
            return False
    
    def run_all_tests(self):
        """Run all tests"""
        print("╔═══════════════════════════════════════════════════════════════╗")
        print("║                                                               ║")
        print("║              COMPREHENSIVE SYSTEM TEST                        ║")
        print("║                                                               ║")
        print("╚═══════════════════════════════════════════════════════════════╝")
        
        # Run tests
        self.test_database()
        self.test_models()
        self.test_agents()
        self.test_fraud_detection()
        self.test_budget_monitoring()
        self.test_orchestrator()
        self.test_decision_tracking()
        self.test_dashboard_api()
        
        # Summary
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        print("\n" + "="*70)
        print("\n📊 TEST SUMMARY")
        print("="*70)
        print(f"  ✅ Passed: {self.tests_passed}")
        print(f"  ❌ Failed: {self.tests_failed}")
        print(f"  ⏱️  Time: {elapsed:.1f}s")
        print()
        
        if self.tests_failed == 0:
            print("🎉 ALL TESTS PASSED! System is ready for demo.")
            print()
            print("Next steps:")
            print("  1. Start dashboard: python test_dashboard.py")
            print("  2. Open browser: http://localhost:5001/agents")
            print("  3. Run demo workflows and watch the magic!")
            return True
        else:
            print(f"⚠️  {self.tests_failed} test(s) failed. Please review errors above.")
            return False


if __name__ == "__main__":
    tester = SystemTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)
