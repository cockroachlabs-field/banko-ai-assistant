# PIPELINE_CONTRACT.md

The contract between this repo (consumer) and
`viragtripathi/cockroachdb-watsonx-data-pipeline` (producer) for the
Proactive Spending Coach's streaming signals. This file is canonical: it is
derived from the implemented consumer code, with file references so drift is
checkable. If either side needs a change, update this file and coordinate
both repos in the same change window.

Consumer implementation: `banko_ai/coach/signals.py` (parsing),
`banko_ai/web/app.py` `/api/cdc/signals` (webhook receiver),
`banko_ai/coach/kafka_consumer.py` (Kafka transport),
`banko_ai/db/schema.py` (DDL).

## The spending_signals table

The producer writes rows here. Banko also creates this table at app startup
(CREATE IF NOT EXISTS), so either side can boot first.

```sql
CREATE TABLE IF NOT EXISTS spending_signals (
  signal_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id         UUID NOT NULL,
  signal_type     STRING NOT NULL,    -- budget_threshold | anomaly | recurring_drift
  severity        STRING NOT NULL,    -- info | warn | critical
  payload         JSONB NOT NULL,
  produced_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
  consumed_at     TIMESTAMPTZ,
  idempotency_key STRING NOT NULL UNIQUE,
  INDEX (user_id, produced_at DESC)
) WITH (ttl_expire_after = '30 days')
```

## The signal row

Both transports normalize to the same object. Required fields, enforced by
`Signal.from_dict`:

| field | type | notes |
|---|---|---|
| signal_id | UUID string | correlation ID across logs, DB rows, traces |
| user_id | UUID string | |
| signal_type | string | one of `budget_threshold`, `anomaly`, `recurring_drift`; anything else is rejected |
| severity | string | `info`, `warn`, or `critical` |
| payload | JSON object | type-specific, see below |
| idempotency_key | string | MUST be deterministic from the signal's content; banko dedups on it. Recommended: `{signal_type}:{user_id}:{period_or_merchant}` |

`produced_at` is optional; the consumer stamps arrival time if absent.

## Payload shapes (v1)

These match the fixtures in `scripts/coach/mock_signals.py`, which are the
forcing function: if the real producer's shapes drift from these, banko's
tests catch it. Amounts are plain decimal numbers, not cents.

budget_threshold:
```json
{"category": "dining", "pct_used": 0.82, "monthly_budget": 400.0,
 "spent_so_far": 328.0, "days_remaining": 9}
```

anomaly:
```json
{"merchant": "Uber", "amount": 850.0, "expected_max": 75.0,
 "z_score": 4.2, "transaction_id": "<uuid>"}
```

recurring_drift:
```json
{"subscription": "Netflix", "old_amount": 15.99, "new_amount": 22.99,
 "pct_change": 0.44, "merchant_id": "<uuid>"}
```

Severity guidance: `info` for a gentle mid-month nudge, `warn` for
needs-attention-this-week (90% of budget, 2-3 sigma anomaly), `critical`
for needs-attention-today (overspend, 4+ sigma, subscription jump over 50%).

## Transport 1: webhook (demo and dev)

```
POST {BANKO_URL}/api/cdc/signals
Content-Type: application/json
X-Banko-Signature: <hex hmac-sha256 of the raw request body>
```

The HMAC secret is shared via the `CDC_WEBHOOK_HMAC_SECRET` env var on both
sides. An unset secret on the banko side rejects everything (fail closed).

The body is a CockroachDB CHANGEFEED wrapped envelope. One or more rows:

```json
{"payload": [
  {"after": { ...signal row as above... }, "updated": "<ts>"}
]}
```

Rows with `"after": null` (deletes) are skipped. Because a raw CRDB webhook
sink cannot sign requests, the producer side owns delivery: either a small
sender that polls `spending_signals` for `consumed_at IS NULL` rows and
POSTs signed envelopes, or any process that can compute the HMAC.

Responses:

| status | meaning |
|---|---|
| 202 | accepted; body has `queued_signal_ids` and `replayed_signal_ids` |
| 200 | idempotent no-op (all rows replayed, or envelope had no inserts) |
| 400 | malformed JSON or invalid signal row |
| 401 | missing or wrong signature |
| 503 | `{"error": "warming_up"}`: the app is still creating its schema at boot; retry later |

Processing is asynchronous; a 202 means queued, not nudged. Redelivery is
safe: claimed signal_ids and idempotency keys are deduped.

## Transport 2: Kafka (prod)

| | |
|---|---|
| topic | `banko.spending_signals` |
| key | `user_id` (keeps per-user ordering) |
| value | the bare signal row JSON, exactly the webhook envelope's `after` object |
| consumer group | `banko-coach-v1` |
| poison messages | published to `banko.spending_signals.dlq` with an `error` header, then committed |
| handler failure | not committed; Kafka redelivers, idempotency dedups |

Enable on the banko side with `COACH_KAFKA_ENABLED=true` and
`KAFKA_BOOTSTRAP_SERVERS=<brokers>`.

Producing via Debezium: the CockroachDB source connector emits Debezium
envelopes, so configure the unwrap and routing transforms to match this
contract:

```
transforms=unwrap,route
transforms.unwrap.type=io.debezium.transforms.ExtractNewRecordState
transforms.route.type=org.apache.kafka.connect.transforms.RegexRouter
transforms.route.regex=.*spending_signals
transforms.route.replacement=banko.spending_signals
```

Reference pipelines: https://github.com/viragtripathi/debezium-cockroachdb-examples

## Rules that keep this working

- The `expenses` table schema is shared between both repos; change it in
  lockstep or not at all.
- Never write to any banko table other than `spending_signals`.
- New signal shapes get a new `signal_type`; never re-version an existing
  one mid-stream.
- Test end to end locally before pushing, on both sides.
