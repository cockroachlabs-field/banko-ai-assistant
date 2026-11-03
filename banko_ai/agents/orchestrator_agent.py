"""
Orchestrator Agent - Coordinates multiple agents for complex workflows.

This agent:
- Plans multi-step workflows
- Delegates tasks to specialized agents
- Synthesizes results from multiple agents
- Manages agent coordination and communication
- Handles complex user queries requiring multiple capabilities
"""

import json
from typing import Any

from .base_agent import BaseAgent


class OrchestratorAgent(BaseAgent):
    """
    Specialized agent for coordinating other agents.
    
    The orchestrator:
    1. Receives complex user requests
    2. Plans a workflow (which agents to use, in what order)
    3. Delegates sub-tasks to specialized agents
    4. Collects and synthesizes results
    5. Provides unified response to user
    
    Example workflow for "Audit my expenses":
    - Step 1: Use Fraud Agent to check for suspicious transactions
    - Step 2: Use Budget Agent to check spending vs budget
    - Step 3: Use Analyst tools to generate statistics
    - Step 4: Synthesize findings into comprehensive report
    """
    
    def __init__(
        self,
        region: str,
        llm: Any,
        database_url: str,
        available_agents: dict[str, BaseAgent] | None = None
    ):
        """
        Initialize Orchestrator Agent.
        
        Args:
            region: AWS region where orchestrator runs
            llm: LangChain LLM instance
            database_url: CockroachDB connection string
            available_agents: Dictionary of {agent_type: agent_instance}
        """
        # System prompt for Orchestrator
        system_prompt = f"""You are an Orchestrator Agent in region {region}.

Your job is to coordinate a team of specialized agents to handle complex requests:

Available Agents and their methods:
- fraud: FraudAgent
  • analyze_expense(expense_id) - Analyze single expense for fraud
  • scan_recent_expenses(hours, limit) - Scan recent expenses
  • learn_from_feedback(decision_id, feedback) - Learn from corrections
  
- budget: BudgetAgent
  • check_budget_status(user_id, monthly_budget, category) - Check current status
  • monitor_users(budget_configs) - Monitor multiple users
  • forecast_spending(user_id, days_ahead) - Forecast future spending
  
- receipt: ReceiptAgent (if available)
  • process_document(file_path, user_id, document_type) - Process receipt/document
  • process_batch(file_paths, user_id) - Process multiple documents

When you receive a request:
1. Break it down into sub-tasks
2. Determine which agents/tools are needed
3. Plan the execution order (what depends on what)
4. Execute the workflow step by step
5. Synthesize all results into a coherent response

Be efficient - don't use agents you don't need.
Be thorough - make sure to address all parts of the request.

Example workflows:
- "Am I over budget?" → budget.check_budget_status(user_id, monthly_budget)
- "Audit my expenses" → fraud.scan_recent_expenses(hours=24) + budget.check_budget_status()
- "Check for fraud" → fraud.scan_recent_expenses(hours=24)
- "Forecast my spending" → budget.forecast_spending(user_id, days_ahead=30)

IMPORTANT: Only use methods that are explicitly listed above. Don't make up method names."""

        # Initialize base agent with no tools initially
        # Orchestrator doesn't need tools - it delegates to other agents
        super().__init__(
            agent_type="orchestrator",
            region=region,
            llm=llm,
            tools=[],
            database_url=database_url,
            system_prompt=system_prompt
        )
        
        self.available_agents = available_agents or {}
    
    def register_agent(self, agent_type: str, agent: BaseAgent):
        """
        Register an agent with the orchestrator.
        
        Args:
            agent_type: Type of agent (receipt, fraud, budget, etc.)
            agent: Agent instance
        """
        self.available_agents[agent_type] = agent
        print(f"   ✅ Registered {agent_type} agent with orchestrator")
    
    def plan_workflow(self, user_request: str, context: dict | None = None) -> dict[str, Any]:
        """
        Plan a workflow to handle the user's request.
        
        Args:
            user_request: User's question or task
            context: Optional context (user_id, filters, etc.)
        
        Returns:
            Dictionary with workflow plan
        """
        self.update_status("thinking", {"action": "plan_workflow", "request": user_request})
        
        # Ask LLM to create a plan
        planning_prompt = f"""Given this user request, create a step-by-step plan.

User Request: "{user_request}"
Context: {json.dumps(context) if context else 'None'}

Available Agents:
{json.dumps(list(self.available_agents.keys()))}

Create a plan with these fields:
1. steps: Array of steps, each with:
   - step_number: Integer
   - agent: Which agent to use (or "synthesize" for final step)
   - action: What to do
   - depends_on: Array of step numbers this depends on (empty if none)
   - params: Parameters needed

Return ONLY valid JSON, no explanation.

Example:
{{
  "steps": [
    {{"step_number": 1, "agent": "fraud", "action": "scan_recent_expenses", "depends_on": [], "params": {{"hours": 24}}}},
    {{"step_number": 2, "agent": "budget", "action": "check_status", "depends_on": [], "params": {{"user_id": "user_01", "budget": 1000}}}},
    {{"step_number": 3, "agent": "synthesize", "action": "combine_results", "depends_on": [1, 2], "params": {{}}}}
  ]
}}"""

        try:
            response = self.llm.invoke(planning_prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Clean response
            cleaned = response_text.strip()
            if cleaned.startswith('```json'):
                cleaned = cleaned.replace('```json', '').replace('```', '').strip()
            elif cleaned.startswith('```'):
                cleaned = cleaned.replace('```', '').strip()
            
            plan = json.loads(cleaned)
            
            self.update_status("idle")
            return {
                'success': True,
                'plan': plan,
                'raw_response': response_text
            }
        
        except Exception as e:
            self.update_status("idle")
            return {
                'success': False,
                'error': str(e),
                'raw_response': response_text if 'response_text' in locals() else None
            }
    
    def execute_workflow(
        self,
        user_request: str,
        context: dict | None = None
    ) -> dict[str, Any]:
        """
        Plan and execute a complete workflow.
        
        Args:
            user_request: User's question or task
            context: Optional context (user_id, filters, etc.)
        
        Returns:
            Dictionary with workflow results
        """
        self.update_status("acting", {"action": "execute_workflow", "request": user_request})
        
        result = {
            'request': user_request,
            'context': context,
            'plan': None,
            'steps_executed': [],
            'final_result': None,
            'success': False
        }
        
        try:
            # Step 1: Plan the workflow
            print(f"\n   🧠 Planning workflow for: '{user_request}'")
            plan_result = self.plan_workflow(user_request, context)
            
            if not plan_result.get('success'):
                result['error'] = f"Planning failed: {plan_result.get('error')}"
                return result
            
            plan = plan_result['plan']
            result['plan'] = plan
            
            print(f"   📋 Plan created with {len(plan.get('steps', []))} steps")
            
            # Step 2: Execute the plan
            step_results = {}
            
            for step in plan.get('steps', []):
                step_num = step['step_number']
                agent_type = step['agent']
                action = step['action']
                params = step.get('params', {})
                depends_on = step.get('depends_on', [])
                
                print(f"\n   ▶️  Step {step_num}: {agent_type}.{action}")
                
                # Check dependencies
                for dep in depends_on:
                    if dep not in step_results:
                        result['error'] = f"Step {step_num} depends on step {dep} which hasn't been executed"
                        return result
                
                # Execute step
                if agent_type == 'synthesize':
                    # Final synthesis step
                    step_result = self._synthesize_results(
                        user_request,
                        {dep: step_results[dep] for dep in depends_on},
                        context
                    )
                else:
                    # Delegate to agent
                    step_result = self._execute_agent_action(
                        agent_type,
                        action,
                        params,
                        {dep: step_results[dep] for dep in depends_on}
                    )
                
                step_results[step_num] = step_result
                result['steps_executed'].append({
                    'step': step_num,
                    'agent': agent_type,
                    'action': action,
                    'success': step_result.get('success', False),
                    'result': step_result
                })
                
                print(f"      {'✅' if step_result.get('success') else '❌'} Step {step_num} completed")
            
            # Get final result (last step)
            if step_results:
                last_step = max(step_results.keys())
                result['final_result'] = step_results[last_step]
                result['success'] = True
            
            # Record decision
            self.store_decision(
                decision_type='workflow_execution',
                context={
                    'request': user_request,
                    'plan': plan,
                    'steps_executed': len(result['steps_executed'])
                },
                reasoning=f"Executed {len(result['steps_executed'])} steps to handle: {user_request}",
                action={
                    'workflow': 'completed',
                    'steps': result['steps_executed']
                },
                confidence=0.9 if result['success'] else 0.3
            )
        
        except Exception as e:
            result['error'] = str(e)
        
        finally:
            self.update_status("idle")
        
        return result
    
    def _execute_agent_action(
        self,
        agent_type: str,
        action: str,
        params: dict[str, Any],
        dependencies: dict[int, Any]
    ) -> dict[str, Any]:
        """
        Execute an action on a specific agent.
        
        Args:
            agent_type: Type of agent
            action: Method name to call
            params: Parameters for the method
            dependencies: Results from previous steps
        
        Returns:
            Result from agent
        """
        if agent_type not in self.available_agents:
            return {
                'success': False,
                'error': f"Agent '{agent_type}' not available"
            }
        
        agent = self.available_agents[agent_type]
        
        try:
            # Check if agent has the method
            if not hasattr(agent, action):
                return {
                    'success': False,
                    'error': f"Agent '{agent_type}' doesn't have method '{action}'"
                }
            
            # Call the method
            method = getattr(agent, action)
            result = method(**params)
            
            return {
                'success': True,
                'agent': agent_type,
                'action': action,
                'result': result
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'agent': agent_type,
                'action': action
            }
    
    def _synthesize_results(
        self,
        user_request: str,
        step_results: dict[int, Any],
        context: dict | None
    ) -> dict[str, Any]:
        """
        Synthesize results from multiple steps into final response.
        
        Args:
            user_request: Original user request
            step_results: Results from all previous steps
            context: Original context
        
        Returns:
            Synthesized result
        """
        # Ask LLM to synthesize
        synthesis_prompt = f"""Synthesize these results into a comprehensive response.

User Request: "{user_request}"

Results from agents:
{json.dumps(step_results, indent=2, default=str)}

Provide a clear, actionable summary that:
1. Directly answers the user's question
2. Highlights key findings
3. Provides specific recommendations
4. Is concise but complete

Be conversational and helpful."""

        try:
            response = self.llm.invoke(synthesis_prompt)
            synthesis = response.content if hasattr(response, 'content') else str(response)
            
            return {
                'success': True,
                'synthesis': synthesis,
                'based_on_steps': list(step_results.keys())
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_agent_status(self) -> dict[str, Any]:
        """
        Get status of all registered agents.
        
        Returns:
            Dictionary with agent statuses
        """
        from sqlalchemy import text
                
        statuses = {}
        
        try:
            from ..db.engine import get_engine

            engine = get_engine(self.database_url)

            with engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT agent_id, agent_type, region, status, last_heartbeat
                    FROM agent_state
                    ORDER BY agent_type, region
                """))
                
                for row in result.fetchall():
                    agent_id = str(row[0])
                    statuses[agent_id] = {
                        'agent_id': agent_id,
                        'type': row[1],
                        'region': row[2],
                        'status': row[3],
                        'last_heartbeat': row[4].isoformat() if row[4] else None
                    }
        except Exception as e:
            statuses['error'] = str(e)
        
        return statuses
