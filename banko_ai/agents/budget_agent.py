"""
Budget Agent - Proactive budget monitoring and forecasting.

This agent:
- Monitors user spending against budgets
- Forecasts spending trends
- Sends proactive alerts before overspending
- Generates budget recommendations
- Tracks spending velocity
"""

import json
from datetime import datetime, timedelta
from typing import Any

from langchain_core.tools import Tool

from .base_agent import BaseAgent
from .tools.analysis_tools import create_analysis_tools


class BudgetAgent(BaseAgent):
    """
    Specialized agent for budget monitoring and forecasting.
    
    Operates proactively:
    1. Monitors spending against budget limits
    2. Calculates spending velocity (rate of spend)
    3. Forecasts month-end totals
    4. Sends alerts before overspending
    5. Generates actionable recommendations
    """
    
    def __init__(
        self,
        region: str,
        llm: Any,
        database_url: str,
        alert_threshold: float = 0.80  # Alert at 80% of budget
    ):
        """
        Initialize Budget Agent.
        
        Args:
            region: AWS region where agent runs
            llm: LangChain LLM instance
            database_url: CockroachDB connection string
            alert_threshold: Percentage of budget to trigger alert (0.0-1.0)
        """
        # Create tools
        analysis_tools = create_analysis_tools(database_url)
        
        # System prompt for Budget Agent
        system_prompt = f"""You are a Budget Monitoring Agent in region {region}.

Your job is to help users stay within their budgets:
1. Monitor spending against budget limits
2. Calculate spending velocity (daily/weekly rate)
3. Forecast month-end totals based on current trends
4. Alert users BEFORE they overspend (at {alert_threshold:.0%} of budget)
5. Provide actionable recommendations to stay on track

You are proactive - don't wait for users to ask. When you detect:
- Spending approaching budget limit (>{alert_threshold:.0%})
- Unusual spending velocity
- Projected overspend

Take action:
- Generate clear, actionable alerts
- Explain the situation with data
- Recommend specific actions

Be helpful and non-judgmental. Focus on solutions, not problems."""
        
        # Initialize base agent
        super().__init__(
            agent_type="budget",
            region=region,
            llm=llm,
            tools=analysis_tools,
            database_url=database_url,
            system_prompt=system_prompt
        )
        
        self.alert_threshold = alert_threshold
    
    def check_budget_status(
        self,
        user_id: str,
        monthly_budget: float,
        category: str | None = None
    ) -> dict[str, Any]:
        """
        Check current budget status for a user.
        
        Args:
            user_id: User ID to check
            monthly_budget: Monthly budget limit
            category: Optional category to check (or overall if None)
        
        Returns:
            Dictionary with budget status and alerts
        """
        self.update_status("thinking", {"action": "check_budget", "user_id": user_id})
        
        from sqlalchemy import create_engine, text
        from sqlalchemy.pool import NullPool
        
        result = {
            'user_id': user_id,
            'budget': monthly_budget,
            'category': category,
            'alert': False,
            'alert_level': 'none',  # none, warning, critical
            'recommendation': [],
            'status': 'ok'
        }
        
        try:
            # Get current month dates
            now = datetime.now()
            month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            
            # Calculate days into month and days remaining
            days_in_month = (month_start.replace(month=month_start.month % 12 + 1, day=1) - timedelta(days=1)).day
            days_elapsed = now.day
            days_remaining = days_in_month - days_elapsed
            
            # Get current month spending
            stats_result = self.execute_tool(
                'calculate_statistics',
                user_id=user_id,
                start_date=month_start.isoformat(),
                end_date=now.isoformat(),
                category=category
            )
            stats_data = json.loads(stats_result)
            
            if not stats_data.get('success'):
                result['error'] = 'Could not retrieve spending data'
                return result
            
            overall = stats_data['overall']
            current_spend = overall['total']
            
            # Calculate metrics
            percent_of_budget = (current_spend / monthly_budget) * 100 if monthly_budget > 0 else 0
            percent_of_month = (days_elapsed / days_in_month) * 100
            
            # Calculate spending velocity
            daily_average = current_spend / days_elapsed if days_elapsed > 0 else 0
            projected_month_end = daily_average * days_in_month
            projected_overspend = projected_month_end - monthly_budget
            
            result.update({
                'current_spend': current_spend,
                'percent_of_budget': percent_of_budget,
                'percent_of_month': percent_of_month,
                'days_elapsed': days_elapsed,
                'days_remaining': days_remaining,
                'daily_average': daily_average,
                'projected_month_end': projected_month_end,
                'projected_overspend': projected_overspend,
                'is_on_track': projected_month_end <= monthly_budget
            })
            
            # Determine alert level
            if percent_of_budget >= 100:
                result['alert'] = True
                result['alert_level'] = 'critical'
                result['status'] = 'over_budget'
                result['recommendation'].append(f"❌ OVER BUDGET: You've spent ${current_spend:.2f} of ${monthly_budget:.2f}")
                result['recommendation'].append("Stop non-essential spending immediately")
            
            elif percent_of_budget >= (self.alert_threshold * 100):
                result['alert'] = True
                result['alert_level'] = 'warning'
                result['status'] = 'approaching_limit'
                result['recommendation'].append(f"⚠️  WARNING: {percent_of_budget:.1f}% of budget used with {days_remaining} days left")
                
                if projected_overspend > 0:
                    result['recommendation'].append(f"Current pace will exceed budget by ${projected_overspend:.2f}")
                    daily_target = (monthly_budget - current_spend) / days_remaining if days_remaining > 0 else 0
                    result['recommendation'].append(f"Reduce daily spending to ${daily_target:.2f} to stay on track")
            
            elif projected_month_end > monthly_budget:
                result['alert'] = True
                result['alert_level'] = 'info'
                result['status'] = 'on_pace_to_exceed'
                result['recommendation'].append(f"📊 INFO: Current pace will exceed budget by ${projected_overspend:.2f}")
                result['recommendation'].append(f"Your daily average: ${daily_average:.2f}")
                daily_target = monthly_budget / days_in_month
                result['recommendation'].append(f"Target daily average: ${daily_target:.2f}")
            
            else:
                result['status'] = 'on_track'
                remaining = monthly_budget - current_spend
                result['recommendation'].append(f"✅ On track: ${remaining:.2f} remaining")
                daily_budget = remaining / days_remaining if days_remaining > 0 else remaining
                result['recommendation'].append(f"Available per day: ${daily_budget:.2f}")
            
            # Get spending breakdown
            result['breakdown'] = stats_data.get('by_category', [])
            
            # Generate AI-powered insights
            insight_prompt = f"""Analyze this budget situation and provide specific recommendations:

Budget: ${monthly_budget:.2f}/month
Current Spend: ${current_spend:.2f} ({percent_of_budget:.1f}%)
Days Elapsed: {days_elapsed}/{days_in_month}
Daily Average: ${daily_average:.2f}
Projected Month-End: ${projected_month_end:.2f}

Status: {result['status']}
Alert Level: {result['alert_level']}

Top Categories:
{json.dumps(result['breakdown'][:3], indent=2)}

Provide 2-3 specific, actionable recommendations. Be concise and practical."""
            
            insights = self.think(insight_prompt)
            result['ai_insights'] = insights
            
            # Store decision
            self.store_decision(
                decision_type='budget_check',
                context={
                    'user_id': user_id,
                    'budget': monthly_budget,
                    'current_spend': current_spend,
                    'status': result['status']
                },
                reasoning=f"Budget status: {result['status']}. {percent_of_budget:.1f}% spent with {percent_of_month:.1f}% of month elapsed.",
                action={
                    'alert_sent': result['alert'],
                    'alert_level': result['alert_level'],
                    'recommendations': result['recommendation']
                },
                confidence=0.95 if result['alert'] else 0.85
            )
            
            # Store in agent memory
            memory_content = f"Budget check for {user_id}: ${current_spend:.2f}/${monthly_budget:.2f} ({percent_of_budget:.1f}%). Status: {result['status']}. {insights[:200]}"
            self.store_memory(
                user_id=user_id,
                memory_type='decision',
                content=memory_content,
                metadata={
                    'budget': monthly_budget,
                    'current_spend': current_spend,
                    'status': result['status'],
                    'alert_level': result['alert_level']
                }
            )
            
            # If approaching limit, check if we need to create tasks for other agents
            if result['alert'] and result['alert_level'] in ['warning', 'critical']:
                # Get fraud agent ID to create escalation task
                from sqlalchemy import create_engine, text
                from sqlalchemy.pool import NullPool
                
                engine = get_engine(self.database_url if hasattr(self, "database_url") else database_url)
                with engine.connect() as conn:
                    # Freshest fraud agent wherever it runs; region labels
                    # are real localities now, not fixed AWS names.
                    fraud_result = conn.execute(text("""
                        SELECT agent_id FROM agent_state
                        WHERE agent_type = 'fraud'
                        ORDER BY last_heartbeat DESC LIMIT 1
                    """))
                    fraud_row = fraud_result.fetchone()
                    
                    if fraud_row:
                        # Create task for fraud agent to review high spenders
                        task_id = self.create_task(
                            target_agent_id=str(fraud_row[0]),
                            task_type='analyze',
                            payload={
                                'user_id': user_id,
                                'reason': 'high_spending',
                                'current_spend': current_spend,
                                'budget': monthly_budget
                            },
                            priority=8 if result['alert_level'] == 'critical' else 5
                        )
                        result['task_created'] = task_id
except Exception as e:
            result['error'] = str(e)
            self.update_status("idle")
        
        return result
    
    def monitor_users(
        self,
        budget_configs: dict[str, float]
    ) -> dict[str, Any]:
        """
        Monitor multiple users' budgets (autonomous worker mode).
        
        Args:
            budget_configs: Dictionary of {user_id: monthly_budget}
        
        Returns:
            Dictionary with monitoring results
        """
        self.update_status("acting", {"action": "monitor_users", "count": len(budget_configs)})
        
        results = {
            'monitored': 0,
            'alerts_sent': 0,
            'users': []
        }
        
        try:
            for user_id, monthly_budget in budget_configs.items():
                results['monitored'] += 1
                
                status = self.check_budget_status(user_id, monthly_budget)
                
                if status.get('alert'):
                    results['alerts_sent'] += 1
                
                results['users'].append({
                    'user_id': user_id,
                    'status': status['status'],
                    'alert_level': status['alert_level'],
                    'current_spend': status.get('current_spend', 0),
                    'budget': monthly_budget
                })
        
        except Exception as e:
            results['error'] = str(e)
            self.update_status("idle")
        
        return results
    
    def forecast_spending(
        self,
        user_id: str,
        days_ahead: int = 30
    ) -> dict[str, Any]:
        """
        Forecast future spending based on historical trends.
        
        Args:
            user_id: User ID to forecast
            days_ahead: Number of days to forecast
        
        Returns:
            Dictionary with forecast
        """
        self.update_status("thinking", {"action": "forecast", "user_id": user_id})
        
        from datetime import timedelta
        
        result = {
            'user_id': user_id,
            'forecast_days': days_ahead,
            'forecast': []
        }
        
        try:
            # Get last 90 days of spending to establish baseline
            now = datetime.now()
            start_date = (now - timedelta(days=90)).isoformat()
            
            stats_result = self.execute_tool(
                'calculate_statistics',
                user_id=user_id,
                start_date=start_date,
                end_date=now.isoformat()
            )
            stats_data = json.loads(stats_result)
            
            if not stats_data.get('success'):
                result['error'] = 'Could not retrieve historical data'
                return result
            
            overall = stats_data['overall']
            daily_average = overall['total'] / 90  # Average over 90 days
            
            # Simple linear forecast (in production, use more sophisticated models)
            forecast_total = daily_average * days_ahead
            
            result.update({
                'historical_daily_average': daily_average,
                'forecast_total': forecast_total,
                'confidence': 0.7  # Medium confidence for simple model
            })
            
            # Generate forecast insights
            forecast_prompt = f"""Based on this spending data, provide a forecast analysis:

User: {user_id}
Historical Daily Average: ${daily_average:.2f}
Forecast Period: {days_ahead} days
Projected Total: ${forecast_total:.2f}

Recent Category Breakdown:
{json.dumps(stats_data.get('by_category', [])[:5], indent=2)}

Provide:
1. Confidence in this forecast (and why)
2. Factors that could change the forecast
3. Recommendations for the user

Be concise and practical."""
            
            insights = self.think(forecast_prompt)
            result['insights'] = insights
        
        except Exception as e:
            result['error'] = str(e)
            self.update_status("idle")
        
        return result
