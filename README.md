[![PyPI version](https://img.shields.io/pypi/v/banko-ai-assistant)](https://pypi.org/project/banko-ai-assistant/)
[![Python versions](https://img.shields.io/pypi/pyversions/banko-ai-assistant)](https://pypi.org/project/banko-ai-assistant/)
[![License](https://img.shields.io/pypi/l/banko-ai-assistant)](https://pypi.org/project/banko-ai-assistant/)
[![Docker Pulls](https://img.shields.io/docker/pulls/virag/banko-ai-assistant)](https://hub.docker.com/r/virag/banko-ai-assistant)

# Banko AI Assistant

A demo banking assistant that combines vector RAG, a LangGraph multi-agent
pipeline, and runtime-switchable LLM providers — all backed by a single
CockroachDB cluster.

![Banko AI Assistant](https://raw.githubusercontent.com/cockroachlabs-field/banko-ai-assistant/main/docs/images/banko-ai-assistant-watsonx.gif)

## What it can do

- **Ask in natural language**. Ask "what did I spend on dining last month?"
  or "show me anything unusual" and answers stream token by token on every
  provider. Math questions (totals, averages, counts) run SQL instead of
  asking the LLM to compute. Threads remember receipts across restarts,
  memories you create are searchable and deletable, and proposed budget
  changes wait for you to confirm.
- **Upload a receipt**: an image or PDF goes through OCR and field
  extraction, posts through the ledger (a duplicate posts nothing), then
  runs the fraud and budget agents, each step showing up as it finishes.
  The receipt joins a conversation checkpointed in CockroachDB, so you can
  still ask about it after a restart.
- **Multi-provider LLM** — watsonx (default), OpenAI, AWS Bedrock, or
  Google Gemini. The provider is chosen at startup via `AI_SERVICE`;
  within a running app you can swap models inside the active provider from
  Settings (model lists come from each provider's API, not a hardcoded
  enum). Switching providers requires a restart with the new env value.
- **See agents work in real time** — the dashboard streams actual agent
  state from the backend (no canned activity).
- **Three-layer semantic cache** — query / embedding / vector-search caches
  with tunable similarity thresholds.

## How it works

![Banko AI Architecture](https://raw.githubusercontent.com/cockroachlabs-field/banko-ai-assistant/main/docs/images/banko-ai-architecture.png)

Five layers, one database:

| Layer | What it does | Where |
|-------|--------------|-------|
| **Web** | Flask + SocketIO UI, REST API, real-time agent status | `banko_ai/web/` |
| **Agents** | A LangGraph assistant graph (routing, SQL aggregation, receipts, memories, a tool planner), each conversation checkpointed by `CockroachDBSaver`; receipt, fraud and budget agents check every upload | `banko_ai/assistant/`, `banko_ai/agents/` |
| **AI providers** | One abstraction over watsonx, OpenAI, Bedrock, Gemini — all LLM calls go through it | `banko_ai/ai_providers/` |
| **Vector search** | `CockroachDBVectorStore` with C-SPANN cosine indexes, 384-dim `all-MiniLM-L6-v2` embeddings (local, no API key) | `banko_ai/vector_search/` |
| **Persistence** | CockroachDB stores SQL rows, vectors, and agent state in one cluster | `banko_ai/utils/` |

All CockroachDB-specific pieces come from
[`langchain-cockroachdb`](https://github.com/cockroachdb/langchain-cockroachdb):
`CockroachDBEngine` (psycopg3 pool), `CockroachDBVectorStore`,
`CockroachDBChatMessageHistory`, and `CockroachDBSaver`.

## Run it

### Prerequisites

- Python 3.10+ (3.12 recommended)
- CockroachDB v25.4.0+ (vector indexes are GA)
- An API key for at least one provider (watsonx, OpenAI, AWS, or Gemini),
  or a local model through Ollama (no key, no internet)

### Install

```bash
pip install banko-ai-assistant            # PyPI
# or
docker-compose up -d                      # Docker
# or
git clone https://github.com/cockroachlabs-field/banko-ai-assistant
cd banko-ai-assistant
uv pip install -e ".[dev]"                # local development
```

### Start CockroachDB

```bash
brew install cockroachdb/tap/cockroach    # macOS
cockroach start-single-node --insecure \
  --store=./cockroach-data \
  --listen-addr=localhost:26257 \
  --http-addr=localhost:8080 --background
```

### Start the app: the bare minimum, per provider

Everything below assumes CockroachDB on `localhost:26257` (that is the
built-in default, so `DATABASE_URL` can be omitted; set it to point
anywhere else). Two optional extras for any of them:
`CDC_WEBHOOK_HMAC_SECRET=<any shared secret>` turns on the Coach webhook
and the signup welcome nudge, and `TOKENIZERS_PARALLELISM=false` only
silences a harmless HuggingFace warning.

**Installed CLI** (`pip install banko-ai-assistant`):

```bash
# Ollama — local model, no keys; needs `ollama serve` running
AI_SERVICE=ollama banko-ai run

# watsonx (the default provider, AI_SERVICE not needed)
WATSONX_API_KEY=... WATSONX_PROJECT_ID=... banko-ai run

# OpenAI — or any compatible endpoint via OPENAI_BASE_URL
AI_SERVICE=openai OPENAI_API_KEY=sk-... banko-ai run

# AWS Bedrock — note aws, not bedrock; a profile OR a key pair
AI_SERVICE=aws AWS_PROFILE=your-profile banko-ai run
AI_SERVICE=aws AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... banko-ai run

# Gemini — Vertex service account OR the Generative AI API key
AI_SERVICE=gemini GOOGLE_PROJECT_ID=... GOOGLE_APPLICATION_CREDENTIALS=sa.json banko-ai run
AI_SERVICE=gemini GOOGLE_PROJECT_ID=... GOOGLE_API_KEY=... banko-ai run
```

**From source** — identical variables, `uv run` in front:

```bash
git clone https://github.com/cockroachlabs-field/banko-ai-assistant
cd banko-ai-assistant && uv sync
AI_SERVICE=ollama uv run banko-ai run
```

**Docker** — identical variables via `-e`, plus two container facts:
host services are reached at `host.docker.internal`, and `SECRET_KEY`
keeps sessions across container restarts (newer images generate a
temporary one when it is omitted):

```bash
docker run --rm -p 5000:5000 \
  -e DATABASE_URL="cockroachdb://root@host.docker.internal:26257/defaultdb?sslmode=disable" \
  -e AI_SERVICE=ollama -e OLLAMA_HOST=http://host.docker.internal:11434 \
  -e SECRET_KEY=pick-something-stable \
  virag/banko-ai-assistant:latest
```

Swap the `AI_SERVICE` line and credentials for any provider above; only
Ollama needs the `OLLAMA_HOST` override.

Useful flags for every mode: `banko-ai run --port 5001` (macOS AirPlay
holds 5000), `banko-ai run --no-data` (skip the sample-data generator),
`banko-ai run --generate-data 2000` (how many sample records an empty
database gets; the default is 5000, or `SAMPLE_DATA_COUNT`).

Open <http://localhost:5000>. The server listens in about two seconds and
does the slow work afterwards. Until the schema and the demo personas are
ready, pages show a getting-ready card that refreshes itself, and
`/api/health` lists each warm-up step with its state.

### Show a different provider

Sometimes the venue has no network, but the audience came to see
watsonx. `AI_SERVICE` picks the provider that really answers;
`AI_DISPLAY_PROVIDER` picks the one the UI shows, with `AI_DISPLAY_MODEL`
for the model name on screen and `AI_DISPLAY_MODELS` for what the model
dropdown offers (picking one never changes the real model). For example
`AI_SERVICE=ollama AI_DISPLAY_PROVIDER=watsonx
AI_DISPLAY_MODEL=ibm/granite-3-8b-instruct banko-ai run` serves from a
local Ollama while the badges say watsonx. The truth is never hidden from
you: the terminal prints both at startup, and `/api/health` reports the
real provider and model under `actual`.

### Sign up and it's yours

The first visit asks for a username. New names pick a spending style
(diner, subscriber, saver, balanced) and get a seeded history in that
style, embeddings included; returning names land straight back in their
own data, and the Coach greets new users with a real nudge within
seconds. Everything on screen is scoped to that identity: SQL
aggregations filter by your user id, vector search runs through the
per-user vector index, and the header shows who is signed in. Questions
that ask for totals or counts ("how much did I spend on restaurants in
the past 60 days?") are answered by SQL directly, so the figures are
exact and identical no matter which AI provider is active; the model
adds the narrative and suggestions around them. Every answer wears a
badge with the real database time, and an expandable panel shows the
actual EXPLAIN ANALYZE that produced it.

On a multi-region cluster the same app goes region aware with no extra
configuration: signup offers a home region (detected live from the
cluster), rows are pinned to it via REGIONAL BY ROW, reads prune to the
user's partition, and the answer badge names the regions that actually
served each query. Kill a region and the badge visibly moves to the
survivors while answers keep flowing. Single-region deployments see
none of this; nothing fake is ever rendered. The legacy personas
(maya, sam, riley) still exist for existing databases, and
`banko-ai clear-demo-users` resets a demo machine while keeping them.

`banko-ai --help` lists the rest (`generate-data`, `clear-data`, `status`,
`search`, etc.). The first run creates the schema (expense, agent, cache,
checkpoint tables), generates sample data with embeddings, and initializes
the selected provider.

### Run it offline (airgap)

The whole stack runs without internet: embeddings are computed locally,
and the LLM is a local model served by Ollama. Bring the stack up once
while online and it preloads itself (an init container pulls the Ollama
model into a named volume, skipping the pull when it is already cached);
every start after that works with the network off:

```bash
docker compose -f docker-compose.airgap.yml up -d   # first run online, then works offline
```

To cache ahead of time instead (say, the night before a talk),
`scripts/airgap/preload-models.sh` pulls the same models explicitly.

Default model is granite3.3:8b (override with `OLLAMA_MODEL`). For a
non-Docker setup, `ollama serve` plus `AI_SERVICE=ollama banko-ai run`
does the same thing. Any OpenAI-compatible endpoint also works via
`OPENAI_BASE_URL` with `AI_SERVICE=openai`.

Two settings keep Ollama from taking over a laptop. `OLLAMA_NUM_CTX`
(default 8192) is the context window the model loads with; the daemon's
own 32k default makes granite3.3:8b take about 10 GB. `OLLAMA_KEEP_ALIVE`
(default `5m`) is how long the model stays loaded after the last
request, and `0` unloads it right away. Banko sends both on every call,
chat and agents alike, and the airgap compose file sets the same values
on the Ollama container. On a Mac, Ollama inside Docker runs on the CPU
because the Linux VM has no access to the Apple GPU, so native
`ollama serve` is the faster choice there.

## Configuration

The important knobs:

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | CockroachDB connection string | `cockroachdb://root@localhost:26257/defaultdb?sslmode=disable` |
| `RLS_MODE` | `auto` puts row-level security on the money tables and serves requests as the `banko_app` role when it can log in. `off` stops routing through that role but leaves the policies in place, which is harmless because the admin role bypasses them. `/api/health` says which one is active and why | `auto` |
| `DATABASE_APP_URL` | Connection string for the `banko_app` role. Empty derives it from `DATABASE_URL` on an insecure cluster; a secure cluster needs it set, or row-level security stays off | empty |
| `AI_SERVICE` | `watsonx`, `openai`, `aws`, `gemini`, or `ollama` | `watsonx` |
| `SECRET_KEY` | Flask session key. Auto-generated per boot in dev, which signs everyone out on restart; set it for stable demo sessions | random |
| `CDC_WEBHOOK_HMAC_SECRET` | Signs Coach webhook posts; unset disables `/api/cdc/signals` and the welcome nudge | unset |
| `SAMPLE_DATA_COUNT` | Sample expenses generated when the database is empty; `--generate-data N` overrides it | `5000` |
| `OLLAMA_NUM_CTX` | Context window Ollama loads the model with | `8192` |
| `OLLAMA_KEEP_ALIVE` | How long Ollama keeps the model loaded when idle; `0` unloads right away | `5m` |
| `WARMUP_WAIT_SECONDS` | How long a request waits for warm-up before getting the getting-ready reply | `20` |
| `MAX_UPLOAD_MB` | Largest request body accepted, receipt uploads included | `10` |
| `SOCKETIO_CORS_ORIGINS` | Comma-separated origins allowed to open Socket.IO; empty means same origin only | empty |
| `SESSION_COOKIE_SECURE` | Mark the session cookie Secure; turn it on behind HTTPS | `false` |
| `CATEGORIES_FILE`, `PERSONAS_FILE`, `SEED_PROFILES_FILE` | Your own YAML in place of the packaged categories, personas or seed profiles (`banko_ai/config/defaults/`) | packaged files |
| `SEED_HISTORY_MONTHS` | Months of history seeded for each persona or new user, unless its profile says otherwise | `6` |
| `SEED_BATCH_SIZE` | Ledger entries written per seeding transaction | `200` |
| `SEED_PARALLELISM` | Users seeded at the same time | `3` |
| `CHECKING_OVERDRAFT_LIMIT` | How far below zero checking may go; `0` means no overdraft | `0` |
| `MAX_TRANSFER_AMOUNT` | Largest single purchase, deposit or transfer the API accepts, and the largest goal target | `1000000` |
| `INTEGRITY_INTERVAL_SECONDS` | Seconds between checks that every balance matches its ledger; `0` turns the checker off | `300` |
| `INTEGRITY_MAX_ACCOUNTS` | The integrity checker skips itself above this many accounts | `5000` |
| `REPORTING_FOLLOWER_READS` | Read cash flow, net worth history and statements with follower reads, a few seconds stale | `true` |
| `RECURRING_MIN_OCCURRENCES` | Charges needed before a merchant counts as a recurring bill | `3` |
| `RECURRING_PRICE_CHANGE_PCT` | Price change, in percent, that flags a subscription as changed | `5` |

### Make the demo yours

The demo ships with sample spending data: categories and merchant rules in `YAML`, personas (spending styles) to log in as, and seeded transaction history. Point `CATEGORIES_FILE` to your own category tree, `PERSONAS_FILE` to your personas, or `SEED_PROFILES_FILE` to custom spending profiles. The app reads them on boot and names the file if one is invalid. A persona's display name and color refresh on boot; a changed spending style shapes that persona's history from its next reset. Seeding stays deterministic and resumable.

Per-provider credentials are in "Start the app" above; `AWS_REGION`
defaults to `us-east-1`. Override model lists with `WATSONX_MODELS`, `OPENAI_MODELS`, `AWS_MODELS`,
`GEMINI_MODELS` (comma-separated). That is not every setting: `banko-ai
config` prints all of them with the value in effect and where it came
from, and `banko-ai config --docs` prints the full reference (cache, pool,
coach and the rest) as Markdown.

## API

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/health` | GET | DB + AI status, plus every warm-up step |
| `/api/health/live` | GET | 200 as soon as the server listens |
| `/api/health/ready` | GET | 200 once the schema and personas are ready, 503 before |
| `/api/ai-providers` | GET | List providers |
| `/api/models` | GET / POST | List or switch models |
| `/api/search` | POST | Vector search |
| `/api/rag` | POST | RAG-based Q&A |
| `/api/upload-receipt` | POST | Run a receipt through the agent pipeline |
| `/api/agents/status` | GET | Agent dashboard data |
| `/api/chat-history/<id>` | GET / DELETE | Persistent chat per session |

The banking data has its own JSON API under `/api/v1`: dashboard, accounts
with statements and CSV export, paged and semantic transaction search, cash
flow, budgets, recurring bills, goals and transfers, all scoped to the
signed-in persona. Transfers take an idempotency key, so a double submit
moves money once. Routes, conventions and examples are in
[`docs/API.md`](docs/API.md#api-v1).

Full list with examples in [`docs/API.md`](docs/API.md). Quick check:

```bash
curl -X POST http://localhost:5000/api/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "What are my biggest expenses this month?"}'
```

## The Spending Coach (streaming + agentic)

Everything above waits for a question. The Coach reacts to events: a
spending signal arrives, an agent investigates with real budget and
transaction tools, and a nudge appears live on the `/coach` tab, for
example "you've used 82% of your dining budget with 9 days left." Signals
and nudges live in CockroachDB with row-level TTLs, and the agent's
conversation state checkpoints there too.

Kick it off the quick way, by posting a synthetic signal at the webhook:

```bash
export CDC_WEBHOOK_HMAC_SECRET=dev-only-secret   # same value as the app
uv run python scripts/coach/mock_signals.py --type=budget_threshold
# also: --type=anomaly, --type=recurring_drift
```

The coach page shows the signed-in user's nudges. By default the script
targets the maya persona, so sign in as `maya` to watch the nudge land,
or pass `--user-id` (or set `COACH_DEFAULT_USER_ID`) for the account you
signed up with.

Or run the real thing: CockroachDB changefeeds streaming through Debezium
and Kafka into the app. One script brings up Kafka, Kafka Connect, and the
Debezium CockroachDB connector, then registers a source on the
`spending_signals` table:

```bash
scripts/coach/cdc-demo/run-cdc-demo.sh
# then run the app with the Kafka transport on:
COACH_KAFKA_ENABLED=true KAFKA_BOOTSTRAP_SERVERS=localhost:29092 \
  AI_SERVICE=watsonx banko-ai run
```

With that stack up, sending a change event is just SQL. Insert a row and
CockroachDB streams it to the Coach, no webhook involved. The coach page
shows the signed-in user's nudges, so target the username you signed up
with:

```sql
INSERT INTO spending_signals
  (user_id, signal_type, severity, payload, idempotency_key)
VALUES
  ((SELECT user_id FROM users WHERE username = 'YOUR-USERNAME'),
   'budget_threshold', 'warn',
   '{"category": "dining", "pct_used": 0.82, "monthly_budget": 400.0,
     "spent_so_far": 328.0, "days_remaining": 9}',
   'demo:' || gen_random_uuid()::STRING);
```

Watch `/coach` while you run it. To verify the whole path automatically:

```bash
uv run python scripts/coach/assert_nudges.py             # webhook transport
uv run python scripts/coach/assert_nudges.py --via sql   # the CDC pipeline
```

The producer contract (payload shapes, idempotency, both transports) is
in [PIPELINE_CONTRACT.md](PIPELINE_CONTRACT.md).

## Where the data plane comes from

This repo is the agent side. A companion repo,
[`viragtripathi/cockroachdb-watsonx-data-pipeline`](https://github.com/viragtripathi/cockroachdb-watsonx-data-pipeline),
streams CockroachDB CDC events into Apache Iceberg on IBM watsonx.data
(both via webhook and Debezium → Kafka). Neither repo requires the other —
this app works against a local CockroachDB with its own sample data — but
the two together demo the end-to-end transactional + lakehouse path.

## Testing

```bash
python -m pytest tests/ -v                # all tests
ruff check banko_ai/                      # lint
```

Integration tests need a populated CockroachDB; they're skipped in CI when
the DB isn't available.

## Troubleshooting

- **CockroachDB version** — must be v25.4.0+ for vector indexes
  (`cockroach version`).
- **DB connection error** — confirm the single-node command above is
  running, then `cockroach sql --insecure --execute "SHOW TABLES;"`.
- **AI provider issues** — confirm keys are exported, then hit
  `/api/health?refresh=1`, which reports provider and database status.
- **Port 5000 in use (macOS)** — AirPlay Receiver claims port 5000.
  Either disable it in System Settings → AirDrop & Handoff, or run
  `banko-ai run --port 5001`.

## License

MIT
