# Banko AI Assistant API Documentation

## Core Endpoints

### Health & Status

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check (DB + AI provider status). `components` holds each warm-up step and its state; `display` and `actual` show the provider on screen and the one really serving. `?refresh=1` re-tests the provider now |
| `/api/health/live` | GET | Liveness: `{"status": "alive"}` as soon as the server listens, even during warm-up |
| `/api/health/ready` | GET | Readiness: 200 once the schema and the demo personas are ready, 503 with their current state before that |
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

## API v1

`/api/v1` is the JSON API over the banking data: accounts, the ledger,
budgets, goals and transfers. Every route answers for the signed-in persona.
With no session (curl, the pipeline) it answers for the persona that
`COACH_DEFAULT_USER_ID` points at, which is maya out of the box. Like every
`/api/` path, it returns 503 `{"error": "warming_up"}` until the schema and
the personas are ready.

### Conventions

- Money is a JSON number rounded to cents and signed from your side: cash
  balances are positive, credit card balances are negative, and net worth is
  the sum of the balances.
- Dates and timestamps are ISO 8601 strings.
- Money moves (`/transfers`, `/goals/<id>/contribute`) need an
  `idempotency_key` of at most 128 characters. Send the same request twice
  and the second answer is `replay` with the same `transfer_id`; nothing
  moves twice.
- Errors:
  - 409 `{"error": "insufficient_funds"}` when the source account can't
    cover it;
  - 409 `{"error": "key_conflict"}` when an `idempotency_key` you already
    used comes back with a different amount, account pair or kind;
  - 404 `{"error": "not_found"}` for anything outside your data, including
    ids that aren't UUIDs and other users' accounts;
  - 400 with a readable `error` for bad input, such as an amount like
    `12.345`, a month like `2026-13` or a body that isn't a JSON object.
- Admin jobs return 202 right away and run in the background. There is no
  admin check on them; this is a demo.

### Routes

| Method | Path | Purpose | Success | Errors |
|--------|------|---------|---------|--------|
| GET | `/api/v1/me` | The current persona: name, avatar color, spending style, home region and seeding progress | 200 | 404 |
| GET | `/api/v1/ai` | Provider, name, model, icon and model list as shown on screen (follows `AI_DISPLAY_PROVIDER`) | 200 | |
| POST | `/api/v1/ai/model` | Switch model, `{"model": "..."}`; the same switch as `POST /api/models` | 200 | 400, 500 |
| GET | `/api/v1/dashboard` | Net worth and its history, spending pace, budgets, bills due in the next 14 days, six recent transactions, latest coach insight | 200 | |
| GET | `/api/v1/accounts` | Accounts grouped into cash and credit, with totals, net worth and card utilization | 200 | |
| GET | `/api/v1/accounts/<id>` | One account | 200 | 404 |
| GET | `/api/v1/accounts/<id>/statements` | No `month`: the months that have a statement. `?month=YYYY-MM`: opening, credits, charges, closing and entries. Add `&format=csv` for a download | 200 | 400, 404 |
| GET | `/api/v1/transactions` | Newest first, keyset paged: `limit` (1 to 200, default 50), `cursor` (from `next_cursor`), `category`, `account`, `kind`, `from`, `to` (years 1900 to 2999), `min`, `max`. With `q`, a semantic search instead, ranked by vector distance (`limit` default 20, plus `category`, `from`, `to`) | 200 | 400 |
| GET | `/api/v1/transactions/<id>` | One transaction, by ledger entry id or expense id | 200 | 404 |
| GET | `/api/v1/cashflow` | Income, spending, net and savings rate per month, plus totals by category and merchant; `?months=` 1 to 24 (default 6) | 200 | 400 |
| GET | `/api/v1/budgets` | This month's budgets with spending against each | 200 | |
| PUT | `/api/v1/budgets` | Set one budget, `{"category": "groceries", "amount": "425"}`; `0` removes it | 200 | 400 |
| GET | `/api/v1/recurring` | Recurring bills and subscriptions found in the history | 200 | |
| GET | `/api/v1/goals` | Savings goals with progress and projected finish | 200 | |
| POST | `/api/v1/goals/<id>/contribute` | Move money into a goal, `{"from_account", "amount", "idempotency_key"}` | 201, 200 replay | 400, 404, 409 |
| POST | `/api/v1/transfers` | Move money between two of your accounts, `{"from_account", "to_account", "amount", "idempotency_key"}`; `"kind": "card_payment"` pays a card | 201, 200 replay | 400, 404, 409 |
| POST | `/api/v1/admin/reset` | Delete and reseed one persona, `{"persona": "maya"}`, or all of them, `{"all": true, "confirm": "all users"}` | 202 | 400, 404 |
| POST | `/api/v1/admin/seed` | Create and seed generated demo users, `{"users": 1-200, "months": 1-24}` | 202 | 400 |

A reset holds the same per-user lease as seeding. If a seed of that persona
is still running, the reset waits up to 30 seconds for it and tries again.
If the lease is still taken, it skips that persona without deleting anything,
and the server log says so. One persona failing doesn't stop the others.

#### Assistant (threads, memories, actions, receipts)

Threads hold conversations; each turn streams Server-Sent Events (SSE). Receipts also stream. Memories persist facts the assistant learns, and actions are proposed changes that wait for confirmation.

| Method | Path | Purpose | Success | Errors |
|--------|------|---------|---------|--------|
| GET | `/api/v1/assistant/threads` | List your threads, newest first | 200 | |
| POST | `/api/v1/assistant/threads` | Create a thread, `{"title": "...", "context": {"kind": "nudge|transaction", "id": "<uuid>"}}` (both fields optional) | 201 | 400 |
| GET | `/api/v1/assistant/threads/<thread_id>` | One thread with its messages and active receipt | 200 | 404 |
| PATCH | `/api/v1/assistant/threads/<thread_id>` | Rename a thread, `{"title": "..."}` | 200 | 404 |
| DELETE | `/api/v1/assistant/threads/<thread_id>` | Delete a thread and its checkpoints | 204 | 404 |
| POST | `/api/v1/assistant/threads/<thread_id>/messages` | Ask a question, `{"message": "...", "language": "es-ES"}` (language optional); streams SSE | SSE | 404, 400 |
| POST | `/api/v1/actions/<action_id>/confirm` | Confirm a proposed action | 200 | 404, 400 |
| POST | `/api/v1/actions/<action_id>/cancel` | Cancel a proposed action | 200 | 404 |
| GET | `/api/v1/memories` | List your memories | 200 | |
| DELETE | `/api/v1/memories/<memory_id>` | Delete a memory | 204 | 404 |
| POST | `/api/v1/receipts` | Upload a receipt, multipart `receipt` file plus optional `thread_id` form field; streams SSE | SSE | 404, 400 |

**SSE events from POST `/api/v1/assistant/threads/<thread_id>/messages`:**

- `step`: `{"label": "Aggregation SQL", "detail": "$922.23 across 16 transactions (Restaurant), computed by CockroachDB", "ms": 23.6, "status": "success"}`. One per piece of work that ran (intent routing, region pruning, SQL, caches, tools, EXPLAIN ANALYZE, the model's answer). `status` is `info`, `success`, `hit`, `miss` or `warning`; `ms` is a number, or null when the step was not timed.
- `token`: `{"text": "partial answer text"}`. Tokens arrive while the answer is written. The `answer` event carries the final text, so replace what you streamed with it.
- `citation`: `{"expense_id": "...", "merchant": "Blue Bottle", "amount": 13.58, "date": "2026-10-11"}`, one per purchase the answer cites. Amounts are positive.
- `proposal`: `{"action_id": "<uuid>", "kind": "budget", "summary": "Set your Groceries budget to $450.00 a month", "status": "pending"}`. `kind` is `budget` or `recategorize`. Nothing changes until `POST /api/v1/actions/<action_id>/confirm`, and the answer of a proposal turn ends with "Waiting for your confirmation: <summary>."
- `answer`: `{"answer": "full answer", "citations": [...], "proposals": [...], "cached": false, "route": "aggregate", "meta": {"region": "us-east-1", "db_ms": 23}}`. `route` is `aggregate`, `receipt`, `remember`, `forget`, `offtopic` or `plan`. The only markers left in `answer` are `[txn:<expense_id>]` for the purchases in `citations`. On aggregation and planner answers, `meta` holds the region that served the query (null on a single-region cluster) and the database time in ms; receipt, memory and off-topic answers, and answers from the answer cache, send `{}`.
- `error`: `{"message": "..."}`, a neutral message (still warming up, reconnecting, or a general failure), never the exception text.
- `done`: `{}`, always the last event.

**SSE events from POST `/api/v1/receipts`:**

- `step`: `{"step": "ocr", "status": "done", "detail": "221 characters", "ms": 746.6}`. The steps run in order: `ocr`, `extract`, `category`, `duplicate`, `post`, `fraud`, `budget`. Each sends `start`, then `done` or `failed`; for a duplicate, `post` is `skipped` and the fraud and budget checks don't run. `ms` is null on `start` and `skipped`.
- `receipt`: `{"document_id": "...", "status": "posted", "filename": "receipt.png", "merchant": "TRADER JOE'S", "amount": 25.94, "date": "2026-10-13", "category": "Groceries", "items": [...], "payment_method": "VISA", "expense_id": "...", "created_at": "...", "error": null, "thread_id": "..."}`. `status` is `posted` or `duplicate` (then `expense_id` is the purchase it matches and nothing posted). `amount` is a number, `category` is the catalog label, and `payment_method` is null when the receipt shows none. `thread_id` is the thread the receipt became the active receipt of (a new one unless the upload named one), or null if that failed after the purchase posted.
- `error`: `{"message": "..."}`, a neutral message.
- `done`: `{}`, always the last event.

**Kept for compatibility:** `/api/rag` (POST, RAG-based Q&A) and `/api/upload-receipt` (POST, receipt upload without SSE) still work. New code should use the `/api/v1/assistant` routes.

### Examples

Accounts (trimmed to one account):

```bash
curl http://localhost:5000/api/v1/accounts
```

```json
{
  "accounts": [
    {"account_id": "d1e33baf-77b6-4190-b531-e2f8922a615d", "kind": "credit",
     "name": "Everyday Card", "institution": "Roachathon Bank", "mask": "1659",
     "balance": -825.71, "credit_limit": 5000.0, "apr": 21.99, "opened_at": "2026-09-12"}
  ],
  "groups": {"cash": ["..."], "credit": ["..."]},
  "cash_total": 11918.58,
  "credit_total": -825.71,
  "net_worth": 11092.87,
  "utilization": 0.1651
}
```

Semantic search over your own purchases:

```bash
curl "http://localhost:5000/api/v1/transactions?q=coffee&limit=1"
```

```json
{
  "items": [
    {"id": "a4276f49-6652-5ddd-b36b-6fcd95fccfc2", "expense_id": "a4276f49-6652-5ddd-b36b-6fcd95fccfc2",
     "kind": "purchase", "posted_at": "2026-09-18", "amount": -42.81, "merchant": "Subway",
     "category": "Restaurant", "category_label": "Restaurant", "category_color": "#F97316",
     "category_icon": "🍽️", "description": "Spent $42.81 on restaurant at Subway using Credit Card.",
     "account_id": null, "distance": 0.6937}
  ]
}
```

A transfer, sent twice with the same key:

```bash
curl -X POST http://localhost:5000/api/v1/transfers \
  -H "Content-Type: application/json" \
  -d '{"from_account": "afbc553b-ddf2-488c-86fb-b62e98aa9e23",
       "to_account": "8b85e1e5-a58b-45e1-84c4-0bc5003c100d",
       "amount": "25.00", "idempotency_key": "rent-split-2026-10"}'
```

```json
{"status": "ok", "transfer_id": "64229f3f-b447-43f9-ac91-f693eb25ff84"}
```

The first call answers 201 with that body. The second answers 200 with
`"status": "replay"` and the same `transfer_id`. The same key with
`"amount": "30.00"` answers 409 `{"error": "key_conflict"}`.

A statement as CSV:

```bash
curl -OJ "http://localhost:5000/api/v1/accounts/afbc553b-ddf2-488c-86fb-b62e98aa9e23/statements?month=2026-10&format=csv"
```

That saves `statement-6981-2026-10.csv` (the account's last four digits and
the month). It starts with the opening balance, lists every entry, and ends
with a `Closing balance` row:

```csv
Date,Description,Kind,Amount
Opening balance,,,6020.24
2026-10-01,Spent $1500.00 on housing at Oak Park Rentals using Debit Card.,purchase,-1500.00
2026-10-02,Goal contribution: Vacation,transfer_out,-200.00
2026-10-03,Spent $101.66 on utilities at Electric Company using Debit Card.,purchase,-101.66
2026-10-10,Payroll deposit,income,2500.00
2026-10-12,Transfer,transfer_out,-25.00
Closing balance,,,6693.58
```

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
curl http://localhost:5000/api/health/ready

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
