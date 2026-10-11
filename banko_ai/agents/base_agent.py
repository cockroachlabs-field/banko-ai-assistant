"""
Base Agent class for all specialized agents.

Provides common functionality:
- Memory management (vector + transactional)
- Tool execution
- State persistence
- Cross-region coordination
"""

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import Tool
from sqlalchemy import text


def json_serializer(obj):
    """JSON serializer for objects not serializable by default"""
    if isinstance(obj, uuid.UUID):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

@dataclass
class AgentDecision:
    """Represents an agent's decision with reasoning"""
    decision_id: str
    decision_type: str
    context: dict[str, Any]
    reasoning: str
    action: dict[str, Any]
    confidence: float
    created_at: datetime
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage"""
        data = asdict(self)
        data['created_at'] = data['created_at'].isoformat()
        return data


class BaseAgent:
    """
    Base class for all agents in the system.
    
    Each agent has:
    - Unique ID and type
    - Region assignment
    - Memory (short-term and long-term)
    - Tools for specific tasks
    - Decision tracking
    """
    
    def __init__(
        self,
        agent_type: str,
        region: str,
        llm: Any,  # LangChain LLM instance
        tools: list[Tool] | None = None,
        database_url: str | None = None,
        system_prompt: str | None = None
    ):
        """
        Initialize the base agent.
        
        Args:
            agent_type: Type of agent (e.g., 'receipt', 'fraud', 'budget')
            region: Region label recorded in agent_state (the deployment's
                gateway region on multi-region clusters, else 'local')
            llm: LangChain LLM instance (provider-agnostic)
            tools: List of tools this agent can use
            database_url: CockroachDB connection string
            system_prompt: Custom system prompt for this agent
        """
        self.agent_id = str(uuid.uuid4())
        self.agent_type = agent_type
        self.region = region
        self.llm = llm
        self.tools = tools or []
        self.database_url = database_url
        self.system_prompt = system_prompt or self._default_system_prompt()
        
        # State
        self.status = "idle"  # idle, thinking, acting, blocked
        self.current_task = None
        self.conversation_history = []
        
        # Register agent in database
        if self.database_url:
            self._register_agent()
    
    def _default_system_prompt(self) -> str:
        """Default system prompt for the agent"""
        return f"""You are a {self.agent_type} agent in the Banko AI system.
You are running in region {self.region}.
Your role is to help users with their financial data.
Always explain your reasoning before taking actions.
Be concise but thorough in your responses."""
    
    def _register_agent(self):
        """Register this agent in the database"""
        from ..db.engine import get_engine

        try:
            engine = get_engine(self.database_url)

            with engine.connect() as conn:
                # Check if agent_state table exists, if not, skip registration
                result = conn.execute(text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'agent_state'
                    )
                """))
                table_exists = result.scalar()
                
                if table_exists:
                    conn.execute(text("""
                        INSERT INTO agent_state 
                        (agent_id, agent_type, region, status, last_heartbeat, metadata)
                        VALUES (:agent_id, :agent_type, :region, :status, :heartbeat, :metadata)
                        ON CONFLICT (agent_id) 
                        DO UPDATE SET 
                            status = EXCLUDED.status,
                            last_heartbeat = EXCLUDED.last_heartbeat
                    """), {
                        'agent_id': self.agent_id,
                        'agent_type': self.agent_type,
                        'region': self.region,
                        'status': self.status,
                        'heartbeat': datetime.utcnow(),
                        'metadata': json.dumps({
                            'tools': [t.name for t in self.tools],
                            'system_prompt': self.system_prompt
                        })
                    })
                    conn.commit()
                    print(f"✅ Agent {self.agent_type} ({self.agent_id}) registered in {self.region}")

        except Exception as e:
            print(f"⚠️  Could not register agent: {e}")
            # Non-fatal - agent can still work without registration
    
    def update_status(self, status: str, task: dict | None = None):
        """Update agent status in database"""
        self.status = status
        self.current_task = task
        
        if not self.database_url:
            return


        try:
            from ..db.engine import get_engine
            engine = get_engine(self.database_url)

            with engine.connect() as conn:
                conn.execute(text("""
                    UPDATE agent_state
                    SET status = :status,
                        current_task = :task,
                        last_heartbeat = :heartbeat
                    WHERE agent_id = :agent_id
                """), {
                    'agent_id': self.agent_id,
                    'status': status,
                    'task': json.dumps(task, default=json_serializer) if task else None,
                    'heartbeat': datetime.utcnow()
                })
                conn.commit()
        except Exception as e:
            print(f"⚠️  Could not update status: {e}")
    
    def store_decision(
        self,
        decision_type: str,
        context: dict[str, Any],
        reasoning: str,
        action: dict[str, Any],
        confidence: float
    ) -> AgentDecision:
        """
        Store an agent decision for audit and learning.
        
        Args:
            decision_type: Type of decision (e.g., 'fraud_flag', 'budget_alert')
            context: Input data that led to decision
            reasoning: Agent's thought process
            action: What action was taken
            confidence: Confidence score (0.0 to 1.0)
        
        Returns:
            AgentDecision object
        """
        decision = AgentDecision(
            decision_id=str(uuid.uuid4()),
            decision_type=decision_type,
            context=context,
            reasoning=reasoning,
            action=action,
            confidence=confidence,
            created_at=datetime.utcnow()
        )
        
        if not self.database_url:
            return decision
        
        
        try:
            from ..db.engine import get_engine
            engine = get_engine(self.database_url)
            
            with engine.connect() as conn:
                # Check if table exists
                result = conn.execute(text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'agent_decisions'
                    )
                """))
                
                if result.scalar():
                    conn.execute(text("""
                        INSERT INTO agent_decisions
                        (decision_id, agent_id, decision_type, context, reasoning, 
                         action, confidence, created_at)
                        VALUES (:decision_id, :agent_id, :decision_type, :context, 
                                :reasoning, :action, :confidence, :created_at)
                    """), {
                        'decision_id': decision.decision_id,
                        'agent_id': self.agent_id,
                        'decision_type': decision_type,
                        'context': json.dumps(context),
                        'reasoning': reasoning,
                        'action': json.dumps(action),
                        'confidence': confidence,
                        'created_at': decision.created_at
                    })
                    conn.commit()
            
            # Emit WebSocket event for dashboard
            try:
                from ..web.agent_dashboard import emit_agent_decision
                emit_agent_decision(
                    agent_id=self.agent_id,
                    decision_type=decision_type,
                    confidence=confidence,
                    reasoning=reasoning
                )
            except Exception:
                # Don't fail if WebSocket not available
                pass
                
        except Exception as e:
            print(f"⚠️  Could not store decision: {e}")
        
        return decision
    
    def think(self, user_input: str, context: dict | None = None) -> str:
        """
        Agent's thinking process using the LLM.
        
        Args:
            user_input: User's message or task
            context: Additional context (e.g., retrieved memories, data)
        
        Returns:
            Agent's response
        """
        self.update_status("thinking", {"input": user_input})
        
        # Build messages
        messages = [SystemMessage(content=self.system_prompt)]
        
        # Add conversation history
        for msg in self.conversation_history[-5:]:  # Last 5 messages
            messages.append(msg)
        
        # Add current input with context
        user_message = user_input
        if context:
            user_message += f"\n\nContext: {json.dumps(context, indent=2)}"
        
        messages.append(HumanMessage(content=user_message))
        
        # Get LLM response
        try:
            response = self.llm.invoke(messages)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Update conversation history
            self.conversation_history.append(HumanMessage(content=user_input))
            self.conversation_history.append(AIMessage(content=response_text))
            
            self.update_status("idle")
            return response_text
        
        except Exception as e:
            self.update_status("idle")
            return f"Error in thinking: {str(e)}"
    
    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Execute a tool by name.
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Tool arguments
        
        Returns:
            Tool execution result
        """
        self.update_status("acting", {"tool": tool_name, "args": kwargs})
        
        tool = next((t for t in self.tools if t.name == tool_name), None)
        if not tool:
            self.update_status("idle")
            raise ValueError(f"Tool '{tool_name}' not found")
        
        try:
            result = tool.func(**kwargs)
            self.update_status("idle")
            return result
        except Exception as e:
            self.update_status("idle")
            raise RuntimeError(f"Tool execution failed: {str(e)}")
    
    def store_memory(
        self,
        user_id: str,
        memory_type: str,
        content: str,
        metadata: dict[str, Any] | None = None
    ) -> str | None:
        """
        Store a memory with vector embedding for semantic search.
        
        Args:
            user_id: User ID this memory belongs to
            memory_type: Type of memory ('conversation', 'decision', 'pattern', 'preference')
            content: The memory content (will be embedded)
            metadata: Additional metadata
        
        Returns:
            memory_id if successful, None otherwise
        """
        if not self.database_url:
            return None
        
        
        try:
            # Generate embedding for the content
            from banko_ai.utils.embeddings import load_embedding_model
            model = load_embedding_model()
            embedding = model.encode(content).tolist()

            from ..db.engine import get_engine
            engine = get_engine(self.database_url)

            memory_id = str(uuid.uuid4())
            
            with engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO agent_memory
                    (memory_id, agent_id, user_id, memory_type, content, embedding, metadata, created_at, accessed_at)
                    VALUES (:memory_id, :agent_id, :user_id, :memory_type, :content, :embedding, :metadata, :created_at, :accessed_at)
                """), {
                    'memory_id': memory_id,
                    'agent_id': self.agent_id,
                    'user_id': user_id,
                    'memory_type': memory_type,
                    'content': content,
                    'embedding': str(embedding),
                    'metadata': json.dumps(metadata or {}),
                    'created_at': datetime.utcnow(),
                    'accessed_at': datetime.utcnow()
                })
                conn.commit()
            return memory_id
        
        except Exception as e:
            print(f"⚠️  Could not store memory: {e}")
            return None
    
    def retrieve_memory(
        self,
        user_id: str,
        query: str,
        memory_type: str | None = None,
        limit: int = 5
    ) -> list[dict[str, Any]]:
        """
        Retrieve memories using semantic search.
        
        Args:
            user_id: User ID to search memories for
            query: Search query (will be embedded)
            memory_type: Filter by memory type (optional)
            limit: Maximum number of results
        
        Returns:
            List of memory objects with similarity scores
        """
        if not self.database_url:
            return []
        
        
        try:
            # Generate embedding for the query
            from banko_ai.utils.embeddings import load_embedding_model
            model = load_embedding_model()
            query_embedding = model.encode(query).tolist()

            from ..db.engine import get_engine
            engine = get_engine(self.database_url)

            with engine.connect() as conn:
                # Build query with optional memory_type filter
                sql = """
                    SELECT 
                        memory_id,
                        memory_type,
                        content,
                        metadata,
                        created_at,
                        (embedding <=> :query_embedding::VECTOR) as distance
                    FROM agent_memory
                    WHERE user_id = :user_id
                """
                
                params = {
                    'user_id': user_id,
                    'query_embedding': str(query_embedding)
                }
                
                if memory_type:
                    sql += " AND memory_type = :memory_type"
                    params['memory_type'] = memory_type
                
                sql += " ORDER BY distance LIMIT :limit"
                params['limit'] = limit
                
                result = conn.execute(text(sql), params)
                
                memories = []
                for row in result.fetchall():
                    memories.append({
                        'memory_id': str(row[0]),
                        'memory_type': row[1],
                        'content': row[2],
                        'metadata': json.loads(row[3]) if row[3] else {},
                        'created_at': row[4].isoformat() if row[4] else None,
                        'similarity': 1.0 - float(row[5])  # Convert distance to similarity
                    })
            return memories
        
        except Exception as e:
            print(f"⚠️  Could not retrieve memory: {e}")
            return []
    
    def create_task(
        self,
        target_agent_id: str,
        task_type: str,
        payload: dict[str, Any],
        priority: int = 5
    ) -> str | None:
        """
        Create a task for another agent.
        
        Args:
            target_agent_id: ID of the agent that should handle this task
            task_type: Type of task ('analyze', 'process', 'report', 'escalate')
            payload: Task data
            priority: Priority (1-10, higher = more urgent)
        
        Returns:
            task_id if successful, None otherwise
        """
        if not self.database_url:
            return None
        
        
        try:
            from ..db.engine import get_engine
            engine = get_engine(self.database_url)
            
            task_id = str(uuid.uuid4())
            
            with engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO agent_tasks
                    (task_id, source_agent_id, target_agent_id, task_type, priority, payload, status, region, created_at)
                    VALUES (:task_id, :source_agent_id, :target_agent_id, :task_type, :priority, :payload, :status, :region, :created_at)
                """), {
                    'task_id': task_id,
                    'source_agent_id': self.agent_id,
                    'target_agent_id': target_agent_id,
                    'task_type': task_type,
                    'priority': priority,
                    'payload': json.dumps(payload, default=json_serializer),
                    'status': 'pending',
                    'region': self.region,
                    'created_at': datetime.utcnow()
                })
                conn.commit()
            return task_id
        
        except Exception as e:
            print(f"⚠️  Could not create task: {e}")
            return None
    
    def get_pending_tasks(self) -> list[dict[str, Any]]:
        """
        Get all pending tasks assigned to this agent.
        
        Returns:
            List of task objects
        """
        if not self.database_url:
            return []
        
        
        try:
            from ..db.engine import get_engine
            engine = get_engine(self.database_url)
            
            with engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT 
                        task_id,
                        source_agent_id,
                        task_type,
                        priority,
                        payload,
                        created_at
                    FROM agent_tasks
                    WHERE target_agent_id = :agent_id
                    AND status = 'pending'
                    ORDER BY priority DESC, created_at ASC
                """), {
                    'agent_id': self.agent_id
                })
                
                tasks = []
                for row in result.fetchall():
                    tasks.append({
                        'task_id': str(row[0]),
                        'source_agent_id': str(row[1]),
                        'task_type': row[2],
                        'priority': row[3],
                        'payload': json.loads(row[4]) if row[4] else {},
                        'created_at': row[5].isoformat() if row[5] else None
                    })
            return tasks
        
        except Exception as e:
            print(f"⚠️  Could not get pending tasks: {e}")
            return []
    
    def update_task_status(
        self,
        task_id: str,
        status: str,
        result: dict[str, Any] | None = None
    ) -> bool:
        """
        Update the status of a task.
        
        Args:
            task_id: Task ID
            status: New status ('processing', 'completed', 'failed')
            result: Task result (optional)
        
        Returns:
            True if successful, False otherwise
        """
        if not self.database_url:
            return False
        
        
        try:
            from ..db.engine import get_engine
            engine = get_engine(self.database_url)
            
            with engine.connect() as conn:
                if status not in ('processing', 'completed', 'failed'):
                    raise ValueError(f"Invalid task status: {status}")
                
                if status == 'processing':
                    conn.execute(text("""
                        UPDATE agent_tasks
                        SET status = :status, result = :result, started_at = :timestamp
                        WHERE task_id = :task_id
                    """), {
                        'task_id': task_id, 'status': status,
                        'result': json.dumps(result, default=json_serializer) if result else None,
                        'timestamp': datetime.utcnow()
                    })
                else:
                    conn.execute(text("""
                        UPDATE agent_tasks
                        SET status = :status, result = :result, completed_at = :timestamp
                        WHERE task_id = :task_id
                    """), {
                        'task_id': task_id, 'status': status,
                        'result': json.dumps(result, default=json_serializer) if result else None,
                        'timestamp': datetime.utcnow()
                    })
                conn.commit()
            return True
        
        except Exception as e:
            print(f"⚠️  Could not update task status: {e}")
            return False
    
    def get_info(self) -> dict[str, Any]:
        """Get agent information"""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'region': self.region,
            'status': self.status,
            'current_task': self.current_task,
            'tools': [t.name for t in self.tools],
            'conversation_length': len(self.conversation_history)
        }
    
    def __repr__(self) -> str:
        return f"<{self.agent_type.title()}Agent id={self.agent_id[:8]}... region={self.region} status={self.status}>"
