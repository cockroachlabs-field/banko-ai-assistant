"""
Analysis tools for agents to calculate statistics and detect patterns.

Provides:
- Statistical analysis (totals, averages, trends)
- Anomaly detection
- Pattern recognition
"""

import json
from datetime import datetime, timedelta
from typing import Any

from langchain_core.tools import Tool
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool


def create_analysis_tools(database_url: str) -> list[Tool]:
    """
    Create analysis tools for agents.
    
    Args:
        database_url: CockroachDB connection string
    
    Returns:
        List of LangChain Tool objects
    """
    
    def calculate_statistics(
        user_id: str,
        start_date: str = None,
        end_date: str = None,
        category: str = None
    ) -> str:
        """
        Calculate expense statistics for a user.
        
        Args:
            user_id: User ID to analyze
            start_date: Optional start date (ISO format)
            end_date: Optional end date (ISO format)
            category: Optional category filter
        
        Returns:
            JSON string with statistics
        """
        try:
            engine = get_engine(self.database_url if hasattr(self, "database_url") else database_url)
            
            with engine.connect() as conn:
                # Build WHERE clause
                where_clauses = ["user_id = :user_id"]
                params = {'user_id': user_id}
                
                if start_date:
                    where_clauses.append("expense_date >= :start_date")
                    params['start_date'] = start_date
                
                if end_date:
                    where_clauses.append("expense_date <= :end_date")
                    params['end_date'] = end_date
                
                if category:
                    where_clauses.append("category = :category")
                    params['category'] = category
                
                where_sql = " AND ".join(where_clauses)
                
                # Get overall statistics
                result = conn.execute(text(f"""
                    SELECT 
                        COUNT(*) as count,
                        SUM(expense_amount) as total,
                        AVG(expense_amount) as average,
                        MIN(expense_amount) as min_amount,
                        MAX(expense_amount) as max_amount,
                        STDDEV(expense_amount) as std_dev
                    FROM expenses
                    WHERE {where_sql}
                """), params)
                
                row = result.fetchone()
                overall = {
                    'count': int(row[0]) if row[0] else 0,
                    'total': float(row[1]) if row[1] else 0.0,
                    'average': float(row[2]) if row[2] else 0.0,
                    'min': float(row[3]) if row[3] else 0.0,
                    'max': float(row[4]) if row[4] else 0.0,
                    'std_dev': float(row[5]) if row[5] else 0.0
                }
                
                # Get by category breakdown
                result = conn.execute(text(f"""
                    SELECT 
                        shopping_type,
                        COUNT(*) as count,
                        SUM(expense_amount) as total
                    FROM expenses
                    WHERE {where_sql}
                    GROUP BY shopping_type
                    ORDER BY total DESC
                    LIMIT 10
                """), params)
                
                by_category = []
                for row in result.fetchall():
                    by_category.append({
                        'category': row[0],
                        'count': int(row[1]),
                        'total': float(row[2]),
                        'percentage': (float(row[2]) / overall['total'] * 100) if overall['total'] > 0 else 0
                    })
                
                # Get by merchant
                result = conn.execute(text(f"""
                    SELECT 
                        merchant,
                        COUNT(*) as count,
                        SUM(expense_amount) as total
                    FROM expenses
                    WHERE {where_sql}
                    GROUP BY merchant
                    ORDER BY total DESC
                    LIMIT 10
                """), params)
                
                by_merchant = []
                for row in result.fetchall():
                    by_merchant.append({
                        'merchant': row[0],
                        'count': int(row[1]),
                        'total': float(row[2])
                    })
return json.dumps({
                'success': True,
                'user_id': user_id,
                'period': {
                    'start': start_date,
                    'end': end_date
                },
                'overall': overall,
                'by_category': by_category,
                'by_merchant': by_merchant
            }, indent=2)
        
        except Exception as e:
            return json.dumps({
                'success': False,
                'error': str(e)
            })
    
    def detect_anomalies(
        user_id: str,
        threshold: float = 2.0
    ) -> str:
        """
        Detect unusual expenses (statistical anomalies).
        
        Args:
            user_id: User ID to analyze
            threshold: Number of standard deviations for anomaly (default 2.0)
        
        Returns:
            JSON string with detected anomalies
        """
        try:
            engine = get_engine(self.database_url if hasattr(self, "database_url") else database_url)
            
            with engine.connect() as conn:
                # Get user's mean and std dev
                result = conn.execute(text("""
                    SELECT 
                        AVG(expense_amount) as mean,
                        STDDEV(expense_amount) as std_dev
                    FROM expenses
                    WHERE user_id = :user_id
                """), {'user_id': user_id})
                
                row = result.fetchone()
                mean = float(row[0]) if row[0] else 0.0
                std_dev = float(row[1]) if row[1] else 0.0
                
                if std_dev == 0:
                    return json.dumps({
                        'success': True,
                        'anomalies': [],
                        'message': 'Not enough data variation to detect anomalies'
                    })
                
                # Find outliers
                upper_bound = mean + (threshold * std_dev)
                lower_bound = mean - (threshold * std_dev)
                
                result = conn.execute(text("""
                    SELECT 
                        expense_id, description, expense_amount, merchant,
                        shopping_type, expense_date,
                        (expense_amount - :mean) / :std_dev as z_score
                    FROM expenses
                    WHERE user_id = :user_id
                    AND (expense_amount > :upper OR expense_amount < :lower)
                    ORDER BY ABS((expense_amount - :mean) / :std_dev) DESC
                    LIMIT 20
                """), {
                    'user_id': user_id,
                    'mean': mean,
                    'std_dev': std_dev,
                    'upper': upper_bound,
                    'lower': lower_bound
                })
                
                anomalies = []
                for row in result.fetchall():
                    anomalies.append({
                        'expense_id': str(row[0]),
                        'description': row[1],
                        'amount': float(row[2]),
                        'merchant': row[3],
                        'category': row[4],
                        'date': row[5].isoformat() if row[5] else None,
                        'z_score': float(row[6]),
                        'type': 'unusually_high' if row[2] > mean else 'unusually_low'
                    })
return json.dumps({
                'success': True,
                'user_id': user_id,
                'statistics': {
                    'mean': mean,
                    'std_dev': std_dev,
                    'threshold': threshold
                },
                'anomalies': anomalies,
                'count': len(anomalies)
            }, indent=2)
        
        except Exception as e:
            return json.dumps({
                'success': False,
                'error': str(e)
            })
    
    def find_duplicates(user_id: str, days: int = 30) -> str:
        """
        Find potential duplicate expenses (same merchant + amount within window).
        
        Args:
            user_id: User ID to check
            days: Number of days to look back
        
        Returns:
            JSON string with potential duplicates
        """
        try:
            engine = get_engine(self.database_url if hasattr(self, "database_url") else database_url)
            
            with engine.connect() as conn:
                # Find duplicates: same merchant + amount + date
                result = conn.execute(text("""
                    WITH grouped AS (
                        SELECT 
                            merchant,
                            expense_amount,
                            expense_date,
                            COUNT(*) as count,
                            ARRAY_AGG(expense_id) as expense_ids,
                            ARRAY_AGG(description) as descriptions
                        FROM expenses
                        WHERE user_id = :user_id
                        GROUP BY merchant, expense_amount, expense_date
                        HAVING COUNT(*) > 1
                    )
                    SELECT * FROM grouped
                    ORDER BY count DESC, expense_amount DESC
                    LIMIT 20
                """), {'user_id': user_id})
                
                duplicates = []
                for row in result.fetchall():
                    duplicates.append({
                        'merchant': row[0],
                        'amount': float(row[1]),
                        'date': str(row[2]),
                        'count': int(row[3]),
                        'expense_ids': [str(eid) for eid in row[4]],
                        'descriptions': row[5],
                    })
return json.dumps({
                'success': True,
                'user_id': user_id,
                'duplicates': duplicates,
                'count': len(duplicates)
            }, indent=2)
        
        except Exception as e:
            return json.dumps({
                'success': False,
                'error': str(e)
            })
    
    def compare_periods(
        user_id: str,
        period1_start: str,
        period1_end: str,
        period2_start: str,
        period2_end: str
    ) -> str:
        """
        Compare spending between two time periods.
        
        Args:
            user_id: User ID
            period1_start, period1_end: First period dates (ISO format)
            period2_start, period2_end: Second period dates (ISO format)
        
        Returns:
            JSON string with comparison
        """
        try:
            engine = get_engine(self.database_url if hasattr(self, "database_url") else database_url)
            
            with engine.connect() as conn:
                # Period 1 stats
                result1 = conn.execute(text("""
                    SELECT 
                        COUNT(*) as count,
                        SUM(expense_amount) as total,
                        AVG(expense_amount) as average
                    FROM expenses
                    WHERE user_id = :user_id
                    AND expense_date BETWEEN :start AND :end
                """), {'user_id': user_id, 'start': period1_start, 'end': period1_end})
                
                row1 = result1.fetchone()
                period1 = {
                    'count': int(row1[0]) if row1[0] else 0,
                    'total': float(row1[1]) if row1[1] else 0.0,
                    'average': float(row1[2]) if row1[2] else 0.0
                }
                
                # Period 2 stats
                result2 = conn.execute(text("""
                    SELECT 
                        COUNT(*) as count,
                        SUM(expense_amount) as total,
                        AVG(expense_amount) as average
                    FROM expenses
                    WHERE user_id = :user_id
                    AND expense_date BETWEEN :start AND :end
                """), {'user_id': user_id, 'start': period2_start, 'end': period2_end})
                
                row2 = result2.fetchone()
                period2 = {
                    'count': int(row2[0]) if row2[0] else 0,
                    'total': float(row2[1]) if row2[1] else 0.0,
                    'average': float(row2[2]) if row2[2] else 0.0
                }
                
                # Calculate changes
                changes = {
                    'count_change': period2['count'] - period1['count'],
                    'count_change_pct': ((period2['count'] - period1['count']) / period1['count'] * 100) if period1['count'] > 0 else 0,
                    'total_change': period2['total'] - period1['total'],
                    'total_change_pct': ((period2['total'] - period1['total']) / period1['total'] * 100) if period1['total'] > 0 else 0,
                    'average_change': period2['average'] - period1['average'],
                    'average_change_pct': ((period2['average'] - period1['average']) / period1['average'] * 100) if period1['average'] > 0 else 0
                }
return json.dumps({
                'success': True,
                'user_id': user_id,
                'period1': {
                    'range': f"{period1_start} to {period1_end}",
                    'stats': period1
                },
                'period2': {
                    'range': f"{period2_start} to {period2_end}",
                    'stats': period2
                },
                'changes': changes
            }, indent=2)
        
        except Exception as e:
            return json.dumps({
                'success': False,
                'error': str(e)
            })
    
    # Create LangChain tools
    tools = [
        Tool(
            name="calculate_statistics",
            description="""Calculate comprehensive expense statistics for a user.
            Args: user_id (str), start_date (optional ISO date), end_date (optional ISO date), category (optional str)
            Returns: JSON with count, total, average, min, max, std dev, breakdown by category and merchant""",
            func=lambda user_id, start_date=None, end_date=None, category=None: 
                calculate_statistics(user_id, start_date, end_date, category)
        ),
        Tool(
            name="detect_anomalies",
            description="""Detect unusual expenses using statistical analysis (z-score).
            Finds expenses that are significantly higher or lower than the user's typical spending.
            Args: user_id (str), threshold (optional float, default 2.0 standard deviations)
            Returns: JSON with detected anomalies and their z-scores""",
            func=lambda user_id, threshold=2.0: detect_anomalies(user_id, threshold)
        ),
        Tool(
            name="find_duplicates",
            description="""Find potential duplicate expenses (same merchant, amount, and date).
            Args: user_id (str), days (optional int, default 30)
            Returns: JSON with groups of potential duplicates""",
            func=lambda user_id, days=30: find_duplicates(user_id, days)
        ),
        Tool(
            name="compare_periods",
            description="""Compare spending between two time periods.
            Args: user_id (str), period1_start (ISO date), period1_end (ISO date), 
                  period2_start (ISO date), period2_end (ISO date)
            Returns: JSON with statistics for both periods and changes (absolute and percentage)""",
            func=lambda user_id, period1_start, period1_end, period2_start, period2_end:
                compare_periods(user_id, period1_start, period1_end, period2_start, period2_end)
        )
    ]
    
    return tools
