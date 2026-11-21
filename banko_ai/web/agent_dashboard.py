"""The ping the Agents page refreshes on when an agent records a decision.

The old dashboard's own API (/api/agents/status, /activity and /stats) is
gone: it needed no sign-in and answered for every user at once, reasoning and
all. The Agents page reads /api/v1/agents, which answers for the signed-in
user."""

from datetime import datetime, timezone

from sqlalchemy import text


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
        # A broadcast ping the Agents page refreshes on. No reasoning and no
        # score: the fraud agent's reasoning names the merchant and amount,
        # and every connected browser gets this.
        from flask import current_app
        if hasattr(current_app, 'socketio'):
            current_app.socketio.emit('agent_decision', {
                'agent_type': agent_type,
                'region': region,
                'decision_type': decision_type,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
    except Exception as e:
        print(f"⚠️  Could not emit agent decision: {e}")
