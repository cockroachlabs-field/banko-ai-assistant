"""
Fraud Agent - Autonomous fraud detection and monitoring.

This agent:
- Runs continuously in the background
- Monitors new expenses for suspicious patterns
- Uses vector search to find similar past fraud cases
- Calculates fraud scores using multiple signals
- Escalates high-confidence cases to humans
- Learns from user feedback
"""

import json
from datetime import datetime, timedelta
from typing import Any

from langchain_core.tools import Tool

from .base_agent import BaseAgent
from .tools.analysis_tools import create_analysis_tools
from .tools.search_tools import create_search_tools


class FraudAgent(BaseAgent):
    """
    Specialized agent for fraud detection and prevention.
    
    Operates autonomously:
    1. Monitors new/recent expenses
    2. Checks for suspicious patterns (duplicates, anomalies, blacklist)
    3. Uses vector search to find similar past fraud cases
    4. Calculates fraud confidence score
    5. Escalates to humans if confidence > threshold
    6. Learns from feedback to improve detection
    """
    
    def __init__(
        self,
        region: str,
        llm: Any,
        database_url: str,
        embedding_model: Any,
        fraud_threshold: float = 0.75,
        duplicate_window_days: int = 60
    ):
        """
        Initialize Fraud Agent.
        
        Args:
            region: AWS region where agent runs
            llm: LangChain LLM instance
            database_url: CockroachDB connection string
            embedding_model: Sentence transformer model
            duplicate_window_days: Days to look back for duplicates (60 for demo, 14 for production)
            fraud_threshold: Confidence threshold for escalation (0.0-1.0)
        """
        # Create tools
        search_tools = create_search_tools(database_url, embedding_model)
        analysis_tools = create_analysis_tools(database_url)
        
        # System prompt for Fraud Agent
        system_prompt = f"""You are a Fraud Detection Agent in region {region}.

Your job is to identify potentially fraudulent expenses:
1. Look for duplicate transactions (same merchant, amount, date)
2. Check for statistical anomalies (unusually high/low amounts)
3. Search for similar past fraud cases using vector search
4. Analyze merchant patterns and blacklists
5. Calculate overall fraud confidence score

You operate autonomously 24/7. When you detect suspicious activity:
- Investigate thoroughly using your tools
- Calculate a confidence score (0.0 to 1.0)
- Explain your reasoning clearly
- If confidence > {fraud_threshold}, escalate to human review

Be thorough but not overly cautious. False positives are costly."""
        
        all_tools = search_tools + analysis_tools
        
        # Initialize base agent
        super().__init__(
            agent_type="fraud",
            region=region,
            llm=llm,
            tools=all_tools,
            database_url=database_url,
            system_prompt=system_prompt
        )
        
        self.embedding_model = embedding_model
        self.fraud_threshold = fraud_threshold
        self.duplicate_window_days = duplicate_window_days
    
    def analyze_expense(self, expense_id: str) -> dict[str, Any]:
        """
        Analyze a single expense for fraud indicators.
        
        Args:
            expense_id: ID of expense to analyze
        
        Returns:
            Dictionary with fraud analysis results
        """
        self.update_status("thinking", {"action": "analyze_expense", "expense_id": expense_id})
        
        result = {
            'expense_id': expense_id,
            'fraud_detected': False,
            'confidence': 0.0,
            'signals': [],
            'reasoning': '',
            'recommendation': 'approve'
        }
        
        try:
            # Get expense details
            expense_result = self.execute_tool('get_expense_by_id', expense_id=expense_id)
            expense_data = json.loads(expense_result)
            
            if not expense_data.get('success'):
                result['error'] = 'Expense not found'
                return result
            
            expense = expense_data['expense']
            user_id = expense['user_id']
            
            # Normalize field names (tool returns expense_amount/expense_date)
            expense['amount'] = expense.get('amount') or expense.get('expense_amount', 0)
            expense['date'] = expense.get('date') or expense.get('expense_date', '')
            
            # Signal 1: Check for duplicates within the configured window
            try:
                from sqlalchemy import text
                from ..db.engine import get_engine
                dup_engine = get_engine(self.database_url)
                with dup_engine.connect() as conn:
                    dup_result = conn.execute(text("""
                        SELECT expense_id::TEXT, merchant, expense_amount, expense_date
                        FROM expenses
                        WHERE user_id = :user_id
                        AND merchant ILIKE :merchant
                        AND expense_amount = :amount
                        AND expense_date = :expense_date
                        AND expense_id != :expense_id
                    """), {
                        'user_id': user_id,
                        'merchant': expense['merchant'],
                        'amount': expense['amount'],
                        'expense_date': expense['date'],
                        'expense_id': expense_id,
                    })
                    dup_rows = dup_result.fetchall()

                if dup_rows:
                    result['signals'].append({
                        'type': 'duplicate',
                        'severity': 'high',
                        'details': f"Duplicate transaction: {len(dup_rows) + 1} occurrences of ${expense['amount']} at {expense['merchant']}"
                    })
                    result['confidence'] += 0.4
                    result['fraud_detected'] = True
            except Exception as dup_err:
                print(f"⚠️  Duplicate check error: {dup_err}")
            
            # Also run the general duplicate scan via tool
            if not result['fraud_detected']:
                duplicate_result = self.execute_tool(
                    'find_duplicates',
                    user_id=user_id,
                    days=self.duplicate_window_days
                )
                duplicate_data = json.loads(duplicate_result)
                
                if duplicate_data.get('success') and duplicate_data.get('count', 0) > 0:
                    for dup_group in duplicate_data.get('duplicates', []):
                        if expense_id in dup_group.get('expense_ids', []):
                            result['signals'].append({
                                'type': 'duplicate',
                                'severity': 'high',
                                'details': f"Duplicate transaction: {dup_group['count']} occurrences of ${dup_group['amount']} at {dup_group['merchant']}"
                            })
                            result['confidence'] += 0.4
                            result['fraud_detected'] = True
                            break
            
            # Signal 2: Statistical anomaly detection
            anomaly_result = self.execute_tool(
                'detect_anomalies',
                user_id=user_id,
                threshold=2.0
            )
            anomaly_data = json.loads(anomaly_result)
            
            if anomaly_data.get('success'):
                anomalies = anomaly_data.get('anomalies', [])
                for anomaly in anomalies:
                    if anomaly['expense_id'] == expense_id:
                        z_score = abs(anomaly['z_score'])
                        result['signals'].append({
                            'type': 'anomaly',
                            'severity': 'high' if z_score > 3.0 else 'medium',
                            'details': f"Statistical outlier: z-score {z_score:.2f} ({anomaly['type']})"
                        })
                        result['confidence'] += min(0.3, z_score / 10.0)
                        break
            
            # Signal 3: Search for similar past fraud cases
            search_query = f"{expense['merchant']} {expense['category']} suspicious fraud"
            similar_result = self.execute_tool(
                'vector_search_expenses',
                query=search_query,
                user_id=user_id,
                limit=5
            )
            similar_data = json.loads(similar_result)
            
            if similar_data.get('success') and similar_data.get('count', 0) > 0:
                # Check if any similar expenses were flagged as fraud
                # (In production, you'd check against a fraud database)
                high_similarity = [e for e in similar_data['results'] if e['similarity_score'] > 0.85]
                if high_similarity:
                    result['signals'].append({
                        'type': 'similar_to_past_fraud',
                        'severity': 'medium',
                        'details': f"Found {len(high_similarity)} similar transactions (highest similarity: {high_similarity[0]['similarity_score']:.2f})"
                    })
                    result['confidence'] += 0.2
            
            # Signal 4: Amount-based rules
            amount = expense['amount']
            if amount > 1000:
                result['signals'].append({
                    'type': 'high_value',
                    'severity': 'low',
                    'details': f"High value transaction: ${amount}"
                })
                result['confidence'] += 0.1
            
            # Calculate final verdict
            result['confidence'] = min(1.0, result['confidence'])
            
            if result['confidence'] >= self.fraud_threshold:
                result['fraud_detected'] = True
                result['recommendation'] = 'escalate'
            elif result['confidence'] >= 0.5:
                result['recommendation'] = 'review'
            else:
                result['recommendation'] = 'approve'
            
            # Generate reasoning with LLM
            reasoning_prompt = f"""Analyze this expense for fraud:

Expense Details:
- Merchant: {expense['merchant']}
- Amount: ${expense['amount']}
- Date: {expense['date']}
- Category: {expense['category']}

Fraud Signals Detected:
{json.dumps(result['signals'], indent=2)}

Fraud Confidence: {result['confidence']:.2%}

Provide a concise explanation of why this is or isn't fraud. Be specific about the signals."""

            reasoning = self.think(reasoning_prompt)
            result['reasoning'] = reasoning
            
            # Store decision for learning
            self.store_decision(
                decision_type='fraud_analysis',
                context={
                    'expense_id': expense_id,
                    'expense': expense,
                    'signals': result['signals']
                },
                reasoning=reasoning,
                action={
                    'recommendation': result['recommendation'],
                    'fraud_detected': result['fraud_detected']
                },
                confidence=result['confidence']
            )
            
            # Store in agent memory for future reference
            if expense.get('user_id'):
                memory_content = f"Fraud analysis for {expense['merchant']} ${expense['amount']}: {result['recommendation']} (confidence: {result['confidence']:.2%}). {reasoning[:200]}"
                self.store_memory(
                    user_id=expense['user_id'],
                    memory_type='decision',
                    content=memory_content,
                    metadata={
                        'expense_id': expense_id,
                        'fraud_detected': result['fraud_detected'],
                        'confidence': result['confidence'],
                        'signals': result['signals']
                    }
                )
            
        except Exception as e:
            result['error'] = str(e)
        
        finally:
            self.update_status("idle")
        
        return result
    
    def scan_recent_expenses(
        self,
        hours: int = 1,
        limit: int = 50
    ) -> dict[str, Any]:
        """
        Scan recent expenses for fraud (autonomous worker mode).
        
        Args:
            hours: How many hours back to scan
            limit: Max number of expenses to check
        
        Returns:
            Dictionary with scan results
        """
        self.update_status("acting", {"action": "scan_recent", "hours": hours})
        
        from sqlalchemy import text
                
        results = {
            'scanned': 0,
            'flagged': 0,
            'escalated': 0,
            'expenses': []
        }
        
        try:
            # Get recent expenses
            from ..db.engine import get_engine

            engine = get_engine(self.database_url if hasattr(self, "database_url") else database_url)
            
            with engine.connect() as conn:
                from datetime import datetime, timedelta
                cutoff_date = (datetime.now() - timedelta(hours=hours)).date()
                result = conn.execute(text("""
                    SELECT expense_id
                    FROM expenses
                    WHERE expense_date >= :cutoff
                    ORDER BY expense_date DESC
                    LIMIT :limit
                """), {'cutoff': cutoff_date, 'limit': limit})
                
                expense_ids = [row[0] for row in result.fetchall()]
            # Analyze each expense
            for expense_id in expense_ids:
                results['scanned'] += 1
                
                analysis = self.analyze_expense(str(expense_id))
                
                if analysis.get('fraud_detected'):
                    results['flagged'] += 1
                    
                    if analysis['recommendation'] == 'escalate':
                        results['escalated'] += 1
                    
                    results['expenses'].append(analysis)
            
        except Exception as e:
            results['error'] = str(e)
        
        finally:
            self.update_status("idle")
        
        return results
    
    def learn_from_feedback(
        self,
        decision_id: str,
        feedback: str,
        notes: str | None = None
    ) -> dict[str, Any]:
        """
        Learn from human feedback on fraud decisions.
        
        Args:
            decision_id: ID of the fraud decision
            feedback: 'correct' or 'incorrect'
            notes: Optional explanation
        
        Returns:
            Dictionary with learning result
        """
        self.update_status("thinking", {"action": "learn_from_feedback", "decision_id": decision_id})
        
        from sqlalchemy import text
                
        try:
            from ..db.engine import get_engine

            engine = get_engine(self.database_url if hasattr(self, "database_url") else database_url)
            
            with engine.connect() as conn:
                # Update decision with feedback
                conn.execute(text("""
                    UPDATE agent_decisions
                    SET user_feedback = :feedback,
                        feedback_at = :feedback_at,
                        context = context || JSONB_BUILD_OBJECT('feedback_notes', :notes)
                    WHERE decision_id = :decision_id
                """), {
                    'decision_id': decision_id,
                    'feedback': feedback,
                    'feedback_at': datetime.utcnow(),
                    'notes': notes or ''
                })
                conn.commit()
                
                # Get decision details for learning
                result = conn.execute(text("""
                    SELECT decision_type, context, reasoning, confidence
                    FROM agent_decisions
                    WHERE decision_id = :decision_id
                """), {'decision_id': decision_id})
                
                row = result.fetchone()
            if not row:
                return {'success': False, 'error': 'Decision not found'}
            
            # In a production system, you would:
            # 1. Update fraud detection models
            # 2. Adjust confidence thresholds
            # 3. Update pattern databases
            # For now, just record the feedback
            
            learning_result = {
                'success': True,
                'decision_id': decision_id,
                'feedback': feedback,
                'was_correct': feedback == 'correct',
                'confidence': float(row[3]) if row[3] else 0.0
            }
            
            # Store learning decision
            self.store_decision(
                decision_type='fraud_learning',
                context={
                    'original_decision_id': decision_id,
                    'feedback': feedback,
                    'notes': notes
                },
                reasoning=f"Received {feedback} feedback on fraud decision {decision_id}",
                action={
                    'action': 'update_patterns',
                    'feedback': feedback
                },
                confidence=1.0
            )
            
            return learning_result
        
        except Exception as e:
            return {'success': False, 'error': str(e)}
        
        finally:
            self.update_status("idle")
