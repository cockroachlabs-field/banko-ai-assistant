# CLAUDE.md — banko-ai-assistant

This file orients any Claude Code session (or human contributor) picking up this repo cold.

## What this project is

Banko AI Assistant is a **production-ready agentic RAG demo** for a fictional bank. It demonstrates retrieval-augmented generation, multi-agent orchestration on LangGraph, durable workflow checkpointing on CockroachDB, multi-provider LLM support, and (in v1.1 of the Spending Coach work) streaming, observability, MCP exposure, and airgap deployment.

It is **not** "just a RAG demo." The repo was renamed from `banko-ai-assistant-rag-demo` in April 2026 because "RAG" undersells what it does — the agentic and multi-provider story is the headline.

**Primary audience for the demo**: AWS re:Invent, IBM TechXchange / Think, and enterprise customer POCs (regulated industries and airgap segments included).

## High-level architecture

```
Flask + SocketIO web app
  ├── Multi-agent layer (LangGraph 1.x, CockroachDBSaver checkpointing)
  │     ├── Receipt agent (OCR via tesseract + pdf2image)
  │     ├── Fraud agent
  │     ├── Budget agent
  │     ├── Coach agent (v1a shipped July 2026 — reactive nudges + chat)
  │     └── Supervisor (planned — LLM-routed dispatch, coach v1b)
  ├── AI provider abstraction (banko_ai/ai_providers/)
  │     ├── watsonx.ai (DEFAULT, also default classifier model)
  │     ├── OpenAI
  │     ├── AWS Bedrock (internally identified as "aws", not "bedrock")
  │     ├── Google Gemini
  │     └── Ollama (local models; airgap default granite3.3:8b)
  ├── Vector RAG (langchain-cockroachdb 0.2.x, C-SPANN cosine indexes,
  │     384-dim all-MiniLM-L6-v2 embeddings — local, no API key)
  ├── Cache layer (3 tiers: query, embedding, vector_search, all in CRDB)
  ├── CDC config (banko_ai/pipeline/config.py — webhook + Kafka modes;
  │     consumers planned in Coach spec)
  └── CockroachDB (v25.4.0+, vector indexes are GA, row-level TTL for
        checkpoints and signal/nudge tables)
```

A fuller architecture diagram lives in the internal design docs (local-only, see Repo layout).

## Repo layout

| Path | Purpose |
|---|---|
| `banko_ai/agents/` | LangGraph specialist agents + workflows |
| `banko_ai/ai_providers/` | Provider abstraction; ALL LLM calls go through here, no direct SDK use anywhere else |
| `banko_ai/config/` | Env-driven settings (~15 knobs) |
| `banko_ai/vector_search/` | Semantic search, embedding enrichment, sample data generator |
| `banko_ai/utils/` | DatabaseManager, cache manager, migration runner, retry/pooling, chat history |
| `banko_ai/web/` | Flask app (~1800 LOC in `app.py`), agent dashboard, auth stub |
| `banko_ai/pipeline/` | CDC config (Kafka, webhook, COS) — consumer impls planned |
| `banko_ai/coach/` | Spending Coach v1a: agent, signal handler, insights, tools, Kafka consumer |
| `banko_ai/agents/supervisor.py` | **Planned** — LLM-routed multi-agent supervisor |
| `banko_ai/observability/` | **Planned** — OTel instrumentation |
| `tests/` | 16+ test suites; integration tests need a populated DB (skipped in CI when DB is absent) |
| `tests/eval/` | **Planned** — eval harness (LLM-as-judge on fixtures) |
| `scripts/` | Dev/ops helpers (build-docker, demo_standalone_search, watch-queries) |
| `scripts/coach/` | mock_signals.py (synthetic signals at the webhook), assert_nudges.py (automated e2e smoke), cdc-demo/ (Kafka + Debezium bootstrap) |
| `scripts/airgap/` | preload-models.sh (cache the Ollama model for offline runs) |
| `docs/` | API.md, DOCKER.md (public). `docs/superpowers/` (specs, plans, handoffs) is local-only and gitignored — internal working docs never go to the public repo |
| `PIPELINE_CONTRACT.md` | Contract for the sibling pipeline repo (payload shapes, idempotency, both transports) |

## Sibling repo (you will touch its outputs, not its code)

`~/idea_workspace/cockroachdb-watsonx-data-pipeline` (GitHub: `viragtripathi/cockroachdb-watsonx-data-pipeline`) — the streaming/lakehouse side. It produces CDC events from CockroachDB into Apache Iceberg on IBM watsonx.data, with two paths: webhook (demo) and Debezium-Kafka (prod). It is developed in a **separate Claude Code session**. The contract between the two repos lives in `PIPELINE_CONTRACT.md` at this repo's root. The pipeline repo writes to a `spending_signals` table; this repo consumes it via webhook or Kafka.

Schema for `expenses` table is **shared** between both repos — keep them in sync if you ever change it here.

## Tech stack quick reference

- **Python**: 3.10+ (3.12 recommended)
- **CockroachDB**: 25.4.0+ (vector indexes are GA)
- **Embeddings**: `all-MiniLM-L6-v2` via sentence-transformers 5.x (384-dim, local). The 3.x to 5.x jump landed July 2026 with embedding parity verified (identical vectors), so nothing stored in CRDB needed re-embedding.
- **Agent framework**: LangGraph 1.x
- **Vector store / chat history / checkpointer**: `langchain-cockroachdb` 0.2.x
- **Web**: Flask 3.1.3 + Flask-SocketIO + eventlet (prod) / threading (dev)
- **Dep manager**: `uv` (lockfile is `uv.lock`, refreshed wholesale July 2026; the deprecated vertexai SDK is gone, google-genai 2.x covers both Vertex and the public API)
- **Testing**: pytest, ruff, mypy (mypy is advisory, see Open bugs)
- **Docker**: multi-arch (amd64/arm64), images on Docker Hub at `virag/banko-ai-assistant`

## Conventions and gotchas — do not relearn these the hard way

These come from prior development pain (mostly from droid sessions Feb-Apr 2026). Honoring them prevents reintroducing fixed bugs and avoids known footguns.

### LLM and AI

- **Default provider is watsonx**. Override via `AI_SERVICE=openai|aws|gemini|ollama` env var. Aggregation questions (how much, how many, average) never reach the LLM for math: the intent classifier routes them to SQL and the model only narrates the pre-computed figures. The UI provider switcher must key on `AI_SERVICE`, not model name — there was a real bug where the watsonx logo kept showing even when OpenAI was active because display logic keyed on model.
- AWS Bedrock is identified internally as `"aws"`, NOT `"bedrock"` — docstring is inconsistent in places but the env value is `aws`.
- **All LLM calls go through `banko_ai/ai_providers/`** — never import `openai`, `anthropic`, `boto3.client('bedrock-runtime')`, etc. directly anywhere else. This is what makes airgap mode (Ollama) and provider switching work cleanly.
- Dynamic model discovery is the pattern for every provider — never hardcode model lists.
- LangGraph checkpoints auto-expire after 7 days via CRDB row-level TTL (`CHECKPOINT_TTL_DAYS`).

### CockroachDB and vector search

- Vector indexes are C-SPANN, cosine ops (`<=>` operator). **Do not use L2** — the codebase migrated away from it.
- No `CAST` in vector queries — the `<=>` operator on a `VECTOR(384)` column does the right thing directly.
- Pre-v25.4.0, the `feature.vector_index.enabled` cluster setting was required. GA now — no longer needed but harmless to set.
- User-specific vector index `idx_expenses_user_embedding` exists; all per-user vector queries should filter by `user_id` to use it.

### Receipt OCR

- Tesseract MUST be in PATH or receipt upload silently returns "Unknown" for all fields. This bit us — see `scripts/setup_system_deps.sh` for installer.

### Caching

- 3-layer cache: `query_cache` (semantic, 0.75 similarity threshold), `embedding_cache` (per-provider — provider is part of the unique key, this was a bug), `vector_search_cache`.
- Cache hits are reported via `/cache-stats` (not `/api/cache/stats`, which does not exist).
- TTL default 24h; configurable.

### Frontend

- Agent dashboard activity feed must reflect REAL agent activity. The user has been bitten by canned/hardcoded demo lights — never ship a dashboard with synthetic events.

### Documentation

- ONE end-to-end doc for any new feature, not a sprawl of `FEATURE_X.md` + `FEATURE_X_NOTES.md` + `FEATURE_X_FIXES.md`. The user has explicitly deleted droid-era doc piles. Keep `docs/` lean.
- README target: ~250 lines, with embedded architecture diagram. Currently 469 lines — slim in the Coach v1 (see spec §8.2).
- Demo marketing tone: Cisco / CockroachDB customer-story register. **Reject validation-speak** like "we prove it at scale with an industry-standard benchmark." The user has pushed back on that phrasing.

### Foundation layers (v2)

- Settings: `banko_ai/config/settings.py` is the only reader of environment variables (a guardrail test enforces it). Add a knob as a field there; `banko-ai config --docs` renders it.
- Database: use `banko_ai.db.engine.get_engine()`, `run_transaction()` or `user_session()`. Never call `create_engine()` or `.dispose()` elsewhere (guardrail test). Bulk writes are set-based; never `executemany` on an autocommit connection (psycopg 3 turns it into one implicit transaction).
- psycopg 3 binds server-side, and CockroachDB rejects placeholders in `EXPLAIN`. Any EXPLAIN with parameters goes through `psycopg.ClientCursor` (see `explain_aggregation` and the vector search explain).
- Schema: every table, index and locality lives in `banko_ai/db/schema.py`. Boot runs one SELECT when the fingerprint matches. New columns on existing tables must be nullable or have defaults.
- Boot: slow work goes in the warm-up steps in `create_app`, reported through the app's `Readiness`; nothing slow runs before the server listens. Pages wait for the `schema` and `personas` components.
- Persona: anything that shows provider identity reads it from `banko_ai/ai_providers/display.py`. Errors a user can see use `neutral_error()` or `NEUTRAL_FAILED`, never the exception text.
- Tests: the plain suite runs against a throwaway single-node CockroachDB on port 26258. Always pass `DATABASE_URL` explicitly; the shell profile points at the chaos cluster on 26257, which only tests gated on `BANKO_TEST_MR_URL` may touch.
- Money: every write goes through `banko_ai/services/ledger.py` (one short transaction per movement, idempotent where a user can repeat it). All money SQL lives in `banko_ai/services/`; read services use `read_session(user_id, follower=...)`, follower reads for reports only. Money is `Decimal` until `jsonable()` at the API edge. Inserts into REGIONAL BY ROW money tables pin `crdb_region` with `money_region` and `region_columns` (`has_region_column` guards `user_budgets` while it converts).
- Content: categories, personas and seed profiles are YAML in `banko_ai/config/defaults/`, overridable with `CATEGORIES_FILE`, `PERSONAS_FILE`, `SEED_PROFILES_FILE`. Persona ids a1, a2, a3 are fixed (coach default user).
- Seeding: deterministic per user and day, resumable via `banko_meta` key `seed:<user_id>` holding the window; reset and seeding take a per-user lease on `seed_lock:<user_id>`. The integrity worker must report 0 drift.
- Vector search: semantic transaction search pins the per-user vector index with a hint (`FROM expenses@idx_expenses_user_embedding`) because freshly seeded users are not yet in table statistics.
- RLS: policies on accounts, ledger_entries, transfers, goals, user_budgets only, for the `banko_app` role; request-path services resolve `url` with `rls.request_url()`. Never add RLS to a vector table.
- Assistant: every chat turn runs the LangGraph graph in `banko_ai/assistant/graph.py` (ingest, route, aggregate | receipt | memory | offtopic | plan, respond, learn), checkpointed per thread with `CockroachDBSaver` (`assistant/checkpoint.py` owns its psycopg pool). Routing is deterministic (`assistant/router.py`); only open questions reach the JSON planner. Models are reached through `ai_providers/streaming.py` (`ProviderChat`), and tests use `tests/_chat.py` (`ScriptedChat`), never a live model.
- Assistant data: `receipts` (not the legacy `documents`), `chat_threads` (RLS), `memories`, `answer_cache`, `search_cache`, `assistant_actions`. Every per-user vector lookup goes through `services/_vector.nearest`, which pins the index and falls back to an exact per-user scan (cockroachlabs/cockroach#13756). Cache keys include the user's data version.
- Receipts: `services/receipts.process_receipt` posts through `ledger.post_purchase`; a receipt ends posted, duplicate or failed, never "processed" after a swallowed error. Duplicates are caught within `duplicate_window_days` (default 3 days, `DUPLICATE_WINDOW_DAYS`).

### Git workflow

- See `~/.claude/projects/.../memory/feedback_local_testing_before_push.md` — **NEVER commit or push until thoroughly tested locally against every relevant LLM provider.** This applies to bot PRs too (dependabot batches go through local rebase + smoke before push — see spec §8.1).
- **Do NOT add bot/agent co-author trailers** to commits or PR bodies (`Co-Authored-By: Claude ...`, "Generated with Claude Code", etc.). Commits should look like Virag wrote them. See `~/.claude/projects/.../memory/feedback_no_bot_commit_trailers.md`.
- Default branch is `main`. Branch sync confusion (`agentic_ai` vs `main`) has bitten before — confirm which branch you're on.

### Open bugs flagged but not yet fixed

- Coach on Gemini is code-complete but never smoke-tested: the GCP project (cockroach-harsh-432021) lost Vertex access mid-test on July 15 (billing). When the project recovers, run the coach smoke (mock_signals x3 plus /api/coach/chat) with AI_SERVICE=gemini before claiming four-provider coverage.
- The watsonx model dropdown lists every chat-capable model IBM returns, including code-tuned ones (`ibm/granite-8b-code-instruct`) that echo prompt templates on JSON extraction tasks. The `ReceiptExtraction` Pydantic gate now catches the bad payloads (422 instead of 500), but the dropdown still presents the foot-gun. Consider annotating models with a "suitable for structured extraction" hint, or filtering code-tuned models out of the JSON-extraction code paths.
- mypy carries findings that predate the July 2026 dep landing; the measured baseline is 222 findings (verified identical against the May lockfile, so not upgrade fallout). The `make types` target is advisory, same as CI. Someone needs to budget a real typing pass; until then do not treat mypy output as a regression signal.
- The ibm-watsonx-ai SDK prints a third-party-license WatsonxAPIWarning on every agent LLM call while the default model is non-IBM (openai/gpt-oss-120b). Informational, not an error; switching WATSONX_MODEL to an ibm/granite model silences it.

## Deployment modes (all three must keep working)

| Mode | LLM | Embeddings | DB | Use case |
|---|---|---|---|---|
| **Cloud** | watsonx / OpenAI / Bedrock / Gemini | local (sentence-transformers) | CRDB | SaaS, conferences |
| **Hybrid** | cloud LLM | local | on-prem CRDB + on-prem CDC pipeline | enterprise customers |
| **Airgap** | Ollama (granite3.3:8b default) | local | on-prem CRDB + on-prem CDC | gov, regulated industries; "disconnect wifi during the talk" demo |

See `memory/project_airgap_first_class.md` — airgap is not optional, it is a first-class deployment target. Any new code must work in all three modes.

Airgap status as of late July 2026: fully working. `AI_SERVICE=ollama` is a first-class provider (dynamic model discovery from the daemon, granite3.3:8b default), `OPENAI_BASE_URL` points the OpenAI provider at any compatible endpoint, and `docker-compose.airgap.yml` plus `scripts/airgap/preload-models.sh` run the whole stack offline.

## Running locally

```bash
docker compose up -d                                # CRDB + banko (cloud-ready)
# OR
docker compose -f docker-compose.standalone.yml up  # single container demo
# OR
scripts/airgap/preload-models.sh                    # once, while online
docker compose -f docker-compose.airgap.yml up -d   # offline demo (Ollama)
```

Then visit http://localhost:5000. App generates 5000 sample expense records on first run.

## Testing locally (the required path before any push)

```bash
make test-local        # ruff + advisory mypy + pytest (same ignore list as CI)
# Then boot the app and smoke every implemented provider:
#   watsonx, openai, aws, gemini, ollama
```

The pytest target skips the same script style test files CI skips; they need a running app or live creds. The coach end to end smoke is automated: boot the app, then `uv run python scripts/coach/assert_nudges.py` (webhook transport) or `--via sql` (exercises the real CDC path when the cdc-demo stack is up). The rest of the manual pass list: grounded chat answer, correct provider logo, receipt upload extracts fields, real dashboard activity, cache stats climb on a repeat query, relevant vector search, clean server log.

CI is **not enough**. CI gates on lint/unit/integration, but the multi-provider smoke is a human-driven, locally-run check.

## Active design docs (local-only, gitignored)

Internal specs and plans live under `docs/superpowers/` on Virag's machine and are never committed or referenced in public artifacts (README, blog posts, PR bodies). Current state in brief: Coach v1a (reactive nudges, conversational mode, webhook + Kafka transports, Live Coach UI) merged to main July 2026 as v1.1.0; still open are v1b (Supervisor, MCP server, eval harness) and v1c (OTel, airgap packaging, docs), plus the July 2026 demo accuracy and local models round. Read the local specs directory for the details.

## CockroachDB skills

The official best-practice skills from
https://github.com/cockroachlabs/cockroachdb-skills should be installed
when working on this repo (`npx skills add cockroachlabs/cockroachdb-skills`,
or copy the skill directories into `~/.claude/skills/`). The three that
govern application code here: `cockroachdb-sql` (schema and query
patterns), `designing-application-transactions` (retries, pooling,
set-based writes), and `designing-multi-region-applications` (REGIONAL
BY ROW, survival goals). Consult them before schema changes, new SQL,
or multi-region work.

## Where the auto-memory lives

Per-project Claude memory is at `~/.claude/projects/-Users-viragtripathi-idea-workspace-banko-ai-assistant/memory/`. Read `MEMORY.md` there for the index of user preferences, project facts, and feedback rules. Update it as you learn new things.
