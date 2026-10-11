# Banko AI Assistant API Documentation

## Core Endpoints

### Health & Status

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check (DB + AI provider status) |
| `/api/ai-providers` | GET | List available AI providers |
| `/api/models` | GET | List available models for current provider |
| `/api/models` | POST | Switch to a different model |
| `/api/agents/status` | GET | Agent dashboard data |
| `/cache-stats` | GET | Cache performance metrics |

### Search & RAG

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/search` | POST | Vector search expenses (SQL-based) |
| `/api/vectorstore-search` | POST | Vector search via langchain-cockroachdb VectorStore |
| `/api/rag` | POST | RAG-based Q&A with AI insights |

### Receipt & Agent Workflows

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/upload-receipt` | POST | Upload receipt image/PDF for agent processing |

### Spending Coach (streaming signals)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/cdc/signals` | POST | CDC webhook receiver. Body is a CockroachDB changefeed wrapped envelope; requires an `X-Banko-Signature` header (hex HMAC-SHA256 of the raw body with `CDC_WEBHOOK_HMAC_SECRET`). Returns 202 with queued and replayed signal ids |
| `/api/coach/nudges` | GET | Recent nudges for the current persona (`?limit=N`) |
| `/api/coach/nudges/<nudge_id>` | GET | One nudge with its tool trace (the evidence panel) |
| `/api/coach/chat` | POST | Conversational coach; JSON body `{"message": ..., "thread_id": ...}`, state checkpointed in CockroachDB |
| `/health/coach` | GET | Coach subsystem health: DB reachability, webhook secret, Kafka flag, last nudge time |

The same signals can arrive over Kafka instead of the webhook: set
`COACH_KAFKA_ENABLED=true` and `KAFKA_BOOTSTRAP_SERVERS`, topic
`banko.spending_signals`. See PIPELINE_CONTRACT.md for the full contract.

### Chat History

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat-history/<session_id>` | GET | Retrieve persistent chat history |
| `/api/chat-history/<session_id>` | DELETE | Clear chat history for a session |
| `/api/user-summary` | GET | Spending summary for the current persona |

### Data Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/generate-data` | POST | Generate sample expense data |
| `/api/stop-generation` | POST | Stop an in-progress generation run |
| `/api/reset-generation` | POST | Reset generation state |
| `/cache-cleanup` | POST | Clear expired cache entries |
| `/data-generator` | GET | Data generator UI |

### Web Interface

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main web interface |

## Response Formats

### Health Response
```json
{
    "success": true,
    "ai_provider": "connected",
    "ai_provider_available": true,
    "ai_service": "watsonx",
    "current_model": "openai/gpt-oss-120b",
    "database": "connected"
}
```

### Models Response
```json
{
    "success": true,
    "provider": "aws",
    "current": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    "models": [
        "anthropic.claude-sonnet-4-20250514-v1:0",
        "us.anthropic.claude-3-5-haiku-20241022-v1:0"
    ]
}
```

### Search Response
```json
{
    "success": true,
    "query": "coffee expenses",
    "results": [
        {
            "expense_id": "uuid",
            "merchant": "Starbucks",
            "amount": 20.47,
            "date": "2025-08-07",
            "description": "Spent $20.47 on coffee at Starbucks using Debit Card.",
            "similarity_score": 0.41,
            "metadata": {
                "shopping_type": "Coffee",
                "payment_method": "Debit Card"
            }
        }
    ]
}
```

### RAG Response
```json
{
    "success": true,
    "response": "Based on your expense data...",
    "sources": [],
    "metadata": {
        "provider": "aws",
        "model": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
        "cached": false,
        "language": "en"
    }
}
```

### Cache Stats Response
```json
{
    "cache_enabled": true,
    "time_period": "24 hours",
    "overall_metrics": {
        "total_requests": 45,
        "total_hits": 23,
        "overall_hit_rate": 51.11,
        "total_tokens_saved": 1250,
        "estimated_cost_savings_usd": 0.025
    }
}
```

## Usage Examples

### cURL

```bash
# Health check
curl http://localhost:5000/api/health

# Vector search
curl -X POST http://localhost:5000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "coffee shop expenses", "limit": 5}'

# RAG query
curl -X POST http://localhost:5000/api/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "What are my biggest expenses this month?"}'

# List models
curl http://localhost:5000/api/models

# Switch model
curl -X POST http://localhost:5000/api/models \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o"}'

# Upload receipt
curl -X POST http://localhost:5000/api/upload-receipt \
  -F "receipt=@receipt.png"

# Agent status
curl http://localhost:5000/api/agents/status
```

### Python

```python
import requests

BASE = "http://localhost:5000"

# Health check
r = requests.get(f"{BASE}/api/health")
print(r.json())

# RAG query
r = requests.post(f"{BASE}/api/rag", json={
    "query": "Show me my grocery spending"
})
print(r.json()["response"])

# Switch model
r = requests.post(f"{BASE}/api/models", json={
    "model": "gpt-4o-mini"
})
print(r.json())
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | CockroachDB connection string | `cockroachdb://root@localhost:26257/defaultdb?sslmode=disable` |
| `AI_SERVICE` | AI provider (`watsonx`, `openai`, `aws`, `gemini`, `ollama`) | `watsonx` |
| `SECRET_KEY` | Flask session key (auto-generated if not set) | Random |

See [README.md](../README.md) for full configuration including provider-specific settings, cache tuning, and database pool configuration.

## Error Responses

| Code | Description |
|------|-------------|
| 400 | Bad request (missing parameters) |
| 500 | Internal server error / AI provider unavailable |

```json
{
    "success": false,
    "error": "Error description"
}
```
