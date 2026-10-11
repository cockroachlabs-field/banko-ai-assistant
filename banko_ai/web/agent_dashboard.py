"""
Agent Dashboard - Real-time visualization of agent activity.

Provides WebSocket-based updates for:
- Agent status (thinking, acting, idle)
- Current tasks
- Workflow execution
- Decision making
"""

from datetime import datetime, timezone

from flask import Blueprint, jsonify, render_template
from sqlalchemy import text

agent_dashboard = Blueprint('agent_dashboard', __name__)


@agent_dashboard.route('/agents')
def dashboard():
    """Render agent dashboard page"""
    return render_template('agent_dashboard.html')


@agent_dashboard.route('/api/agents/status')
def get_all_agent_status():
    """Get status of all registered agents"""
    from ..config.settings import get_config
    from ..db.engine import get_engine
    
    config = get_config()
    engine = get_engine(config.database_url)
    
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT 
                    agent_id,
                    agent_type,
                    region,
                    status,
                    current_task,
                    last_heartbeat,
                    created_at
                FROM agent_state
                ORDER BY agent_type, region, created_at DESC
            """))
            
            agents = []
            for row in result.fetchall():
                agents.append({
                    'agent_id': str(row[0]),
                    'type': row[1],
                    'region': row[2],
                    'status': row[3],
                    'current_task': row[4],
                    'last_heartbeat': row[5].isoformat() if row[5] else None,
                    'created_at': row[6].isoformat() if row[6] else None
                })
        return jsonify({
            'success': True,
            'agents': agents,
            'count': len(agents)
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@agent_dashboard.route('/api/agents/activity')
def get_recent_activity():
    """Get recent agent decisions and activity (last 10 minutes only)"""

    from ..config.settings import get_config
    
    config = get_config()
    from ..db.engine import get_engine
    engine = get_engine(config.database_url)
    
    try:
        with engine.connect() as conn:
            # Get recent decisions from last 10 minutes only
            result = conn.execute(text("""
                SELECT 
                    d.decision_id,
                    d.agent_id,
                    a.agent_type,
                    a.region,
                    d.decision_type,
                    d.confidence,
                    d.created_at,
                    d.reasoning
                FROM agent_decisions d
                JOIN agent_state a ON d.agent_id = a.agent_id
                WHERE d.created_at > NOW() - INTERVAL '10 minutes'
                ORDER BY d.created_at DESC
                LIMIT 20
            """))
            
            activities = []
            for row in result.fetchall():
                activities.append({
                    'decision_id': str(row[0]),
                    'agent_id': str(row[1]),
                    'agent_type': row[2],
                    'region': row[3],
                    'decision_type': row[4],
                    'confidence': float(row[5]) if row[5] else 0,
                    'created_at': row[6].isoformat() if row[6] else None,
                    'reasoning': row[7][:200] if row[7] else None  # First 200 chars
                })
        return jsonify({
            'success': True,
            'activities': activities,
            'count': len(activities)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@agent_dashboard.route('/api/agents/stats')
def get_agent_stats():
    """Get agent memory and task statistics"""

    from ..config.settings import get_config
    
    config = get_config()
    from ..db.engine import get_engine
    engine = get_engine(config.database_url)
    
    try:
        with engine.connect() as conn:
            # Get memory counts per agent
            memory_result = conn.execute(text("""
                SELECT 
                    a.agent_id,
                    a.agent_type,
                    a.region,
                    COUNT(m.memory_id) as memory_count
                FROM agent_state a
                LEFT JOIN agent_memory m ON a.agent_id = m.agent_id
                GROUP BY a.agent_id, a.agent_type, a.region
            """))
            
            # Get task counts per agent
            task_result = conn.execute(text("""
                SELECT 
                    a.agent_id,
                    COUNT(CASE WHEN t.status = 'pending' THEN 1 END) as pending_tasks,
                    COUNT(CASE WHEN t.status = 'completed' THEN 1 END) as completed_tasks
                FROM agent_state a
                LEFT JOIN agent_tasks t ON a.agent_id = t.target_agent_id
                GROUP BY a.agent_id
            """))
            
            memory_stats = {}
            for row in memory_result.fetchall():
                agent_id = str(row[0])
                memory_stats[agent_id] = {
                    'agent_type': row[1],
                    'region': row[2],
                    'memory_count': row[3] or 0
                }
            
            task_stats = {}
            for row in task_result.fetchall():
                agent_id = str(row[0])
                task_stats[agent_id] = {
                    'pending_tasks': row[1] or 0,
                    'completed_tasks': row[2] or 0
                }
            
            # Combine stats
            agent_stats = []
            for agent_id, mem_data in memory_stats.items():
                tasks = task_stats.get(agent_id, {'pending_tasks': 0, 'completed_tasks': 0})
                agent_stats.append({
                    'agent_id': agent_id,
                    'agent_type': mem_data['agent_type'],
                    'region': mem_data['region'],
                    'memory_count': mem_data['memory_count'],
                    'pending_tasks': tasks['pending_tasks'],
                    'completed_tasks': tasks['completed_tasks']
                })
# Overall stats
        total_memory = sum(a['memory_count'] for a in agent_stats)
        total_pending = sum(a['pending_tasks'] for a in agent_stats)
        total_completed = sum(a['completed_tasks'] for a in agent_stats)
        
        return jsonify({
            'success': True,
            'agent_stats': agent_stats,
            'totals': {
                'memory_count': total_memory,
                'pending_tasks': total_pending,
                'completed_tasks': total_completed
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# WebSocket event handlers
def emit_agent_status_update(agent_id, status, task=None):
    """Emit agent status update via WebSocket"""
    try:
        from flask import current_app
        if hasattr(current_app, 'socketio'):
            current_app.socketio.emit('agent_status_update', {
                'agent_id': str(agent_id),
                'status': status,
                'task': task,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
    except Exception as e:
        print(f"⚠️  Could not emit agent status: {e}")


def emit_agent_decision(agent_id, decision_type, confidence, reasoning):
    """Emit agent decision via WebSocket"""
    try:
        # Get agent info from database

        from ..config.settings import get_config
        
        config = get_config()
        from ..db.engine import get_engine
        engine = get_engine(config.database_url)
        
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT agent_type, region
                FROM agent_state
                WHERE agent_id = :agent_id
            """), {'agent_id': agent_id})
            row = result.fetchone()
            
            if row:
                agent_type = row[0]
                region = row[1]
            else:
                agent_type = 'unknown'
                region = 'unknown'
# Emit via WebSocket
        from flask import current_app
        if hasattr(current_app, 'socketio'):
            current_app.socketio.emit('agent_decision', {
                'agent_id': str(agent_id),
                'agent_type': agent_type,
                'region': region,
                'decision_type': decision_type,
                'confidence': confidence,
                'reasoning': reasoning[:200] if reasoning else None,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
    except Exception as e:
        print(f"⚠️  Could not emit agent decision: {e}")


def emit_workflow_update(workflow_id, step, status, result=None):
    """Emit workflow execution update via WebSocket"""
    try:
        from flask import current_app
        if hasattr(current_app, 'socketio'):
            current_app.socketio.emit('workflow_update', {
                'workflow_id': workflow_id,
                'step': step,
                'status': status,
                'result': result,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
    except Exception as e:
        print(f"⚠️  Could not emit workflow update: {e}")
