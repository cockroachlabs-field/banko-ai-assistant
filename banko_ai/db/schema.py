"""Every table, index and locality banko needs, as one ordered registry.

Boot reads a fingerprint of the rendered DDL from banko_meta. When it
matches, no DDL runs at all: one SELECT. When it differs, the registry runs
once. Fresh tables are created with their indexes and locality inline, which
takes seconds per table even on a 9-node multi-region cluster (CREATE TABLE
followed by CREATE INDEX and SET LOCALITY took 82 s for one empty table).
Existing tables only get the columns and indexes they are missing. Locality
changes on tables that already hold data are returned to the caller so a
background thread can run them while the app serves.

New columns added to an existing table must be nullable or have a default.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from sqlalchemy import text
from sqlalchemy.engine import Connection
from sqlalchemy.exc import DBAPIError, ProgrammingError

from ..config.settings import get_config
from .engine import get_engine
from .rls import policy_statements, policy_tables, rls_statements, role_statements, supports_rls

log = logging.getLogger("banko.schema")
narration = logging.getLogger("banko.migration")  # the lines the demo console shows

REGISTRY_VERSION = 2
FINGERPRINT_KEY = "schema_fingerprint"
LOCK_KEY = "schema_lock"
LOCK_TTL_SECONDS = 300


@dataclass(frozen=True)
class Topology:
    """regional is true when the database has a primary region, which is
    what REGIONAL BY ROW and GLOBAL need."""

    regional: bool
    regions: tuple[str, ...] = ()


@dataclass(frozen=True)
class Column:
    name: str
    definition: str


@dataclass(frozen=True)
class Index:
    name: str
    columns: str
    vector: bool = False
    storing: str = ""

    def _body(self) -> str:
        body = f"({self.columns})"
        return f"{body} STORING ({self.storing})" if self.storing else body

    def inline(self) -> str:
        kind = "VECTOR INDEX" if self.vector else "INDEX"
        return f"{kind} {self.name} {self._body()}"

    def create(self, table: str) -> str:
        kind = "VECTOR INDEX" if self.vector else "INDEX"
        return f"CREATE {kind} IF NOT EXISTS {self.name} ON {table} {self._body()}"


@dataclass(frozen=True)
class Table:
    name: str
    columns: tuple[Column, ...]
    primary_key: str
    indexes: tuple[Index, ...] = ()
    constraints: tuple[str, ...] = ()
    locality: str | None = None  # "REGIONAL BY ROW", "GLOBAL", or None for the database default
    options: str = ""            # for example WITH (ttl_expire_after = '30 days')

    def create_sql(self, topo: Topology) -> str:
        parts = [f"  {c.name} {c.definition}" for c in self.columns]
        parts.append(f"  PRIMARY KEY ({self.primary_key})")
        parts += [f"  {c}" for c in self.constraints]
        parts += [f"  {i.inline()}" for i in self.indexes]
        sql = f"CREATE TABLE IF NOT EXISTS {self.name} (\n" + ",\n".join(parts) + "\n)"
        if self.options:
            sql += f" {self.options}"
        if self.locality and topo.regional:
            sql += f" LOCALITY {self.locality}"
        return sql


def _c(*pairs: tuple[str, str]) -> tuple[Column, ...]:
    return tuple(Column(name, definition) for name, definition in pairs)


def tables() -> tuple[Table, ...]:
    """The registry, in creation order (foreign key targets first)."""
    cfg = get_config()
    return (
        Table("banko_meta", _c(
            ("key", "STRING NOT NULL"),
            ("value", "STRING NOT NULL"),
            ("updated_at", "TIMESTAMPTZ NOT NULL DEFAULT now()"),
        ), primary_key="key", locality="GLOBAL"),
        Table("categories", _c(
            ("key", "STRING NOT NULL"),
            ("label", "STRING NOT NULL"),
            ("icon", "STRING NOT NULL"),
            ("color", "STRING NOT NULL"),
            ("grp", "STRING NOT NULL"),
            ("sort", "INT8 NOT NULL DEFAULT 0"),
        ), primary_key="key", constraints=(
            "CHECK (grp IN ('needs', 'wants', 'savings', 'income', 'transfer'))",
        ), locality="GLOBAL"),
        Table("expenses", _c(
            ("expense_id", "UUID NOT NULL DEFAULT gen_random_uuid()"),
            ("user_id", "UUID NOT NULL"),
            ("expense_date", "DATE NOT NULL"),
            ("expense_amount", "DECIMAL(10,2) NOT NULL"),
            ("shopping_type", "STRING NOT NULL"),
            ("description", "STRING"),
            ("merchant", "STRING"),
            ("payment_method", "STRING NOT NULL"),
            ("recurring", "BOOL DEFAULT false"),
            ("tags", "STRING[]"),
            ("embedding", "VECTOR(384)"),
            ("created_at", "TIMESTAMPTZ DEFAULT now()"),
        ), primary_key="expense_id", indexes=(
            Index("idx_expenses_embedding", "embedding vector_cosine_ops", vector=True),
            Index("idx_expenses_user_embedding", "user_id, embedding vector_cosine_ops", vector=True),
            Index("idx_expenses_user_date", "user_id ASC, expense_date DESC"),
            Index("idx_expenses_merchant", "merchant ASC"),
            Index("idx_expenses_shopping_type", "shopping_type ASC"),
        ), locality="REGIONAL BY ROW"),
        Table("agent_state", _c(
            ("agent_id", "UUID NOT NULL DEFAULT gen_random_uuid()"),
            ("agent_type", "STRING NOT NULL"),
            ("region", "STRING NOT NULL"),
            ("status", "STRING NOT NULL DEFAULT 'idle'"),
            ("current_task", "JSONB"),
            ("last_heartbeat", "TIMESTAMPTZ DEFAULT now()"),
            ("metadata", "JSONB"),
            ("created_at", "TIMESTAMPTZ DEFAULT now()"),
        ), primary_key="agent_id", indexes=(
            Index("idx_agent_state_region_status", "region, status"),
            Index("idx_agent_state_type", "agent_type, status"),
        )),
        Table("agent_memory", _c(
            ("memory_id", "UUID NOT NULL DEFAULT gen_random_uuid()"),
            ("agent_id", "UUID REFERENCES agent_state(agent_id) ON DELETE CASCADE"),
            ("user_id", "STRING"),
            ("memory_type", "STRING NOT NULL"),
            ("content", "STRING NOT NULL"),
            ("embedding", "VECTOR(384)"),
            ("metadata", "JSONB"),
            ("created_at", "TIMESTAMPTZ DEFAULT now()"),
            ("accessed_at", "TIMESTAMPTZ DEFAULT now()"),
            ("access_count", "INT8 DEFAULT 0"),
        ), primary_key="memory_id"),
        Table("agent_decisions", _c(
            ("decision_id", "UUID NOT NULL DEFAULT gen_random_uuid()"),
            ("agent_id", "UUID REFERENCES agent_state(agent_id)"),
            ("decision_type", "STRING NOT NULL"),
            ("context", "JSONB NOT NULL"),
            ("reasoning", "STRING"),
            ("action", "JSONB"),
            ("confidence", "FLOAT8"),
            ("user_feedback", "STRING"),
            ("created_at", "TIMESTAMPTZ DEFAULT now()"),
            ("feedback_at", "TIMESTAMPTZ"),
        ), primary_key="decision_id"),
        Table("agent_tasks", _c(
            ("task_id", "UUID NOT NULL DEFAULT gen_random_uuid()"),
            ("source_agent_id", "UUID REFERENCES agent_state(agent_id)"),
            ("target_agent_id", "UUID REFERENCES agent_state(agent_id)"),
            ("task_type", "STRING NOT NULL"),
            ("priority", "INT8 DEFAULT 5"),
            ("payload", "JSONB"),
            ("status", "STRING DEFAULT 'pending'"),
            ("region", "STRING"),
            ("created_at", "TIMESTAMPTZ DEFAULT now()"),
            ("started_at", "TIMESTAMPTZ"),
            ("completed_at", "TIMESTAMPTZ"),
            ("result", "JSONB"),
            ("error", "STRING"),
        ), primary_key="task_id", indexes=(
            Index("idx_agent_tasks_status", "status, priority DESC, created_at"),
        )),
        Table("documents", _c(
            ("document_id", "UUID NOT NULL DEFAULT gen_random_uuid()"),
            ("user_id", "STRING NOT NULL"),
            ("document_type", "STRING NOT NULL"),
            ("s3_key", "STRING"),
            ("original_filename", "STRING"),
            ("extracted_text", "STRING"),
            ("extracted_data", "JSONB"),
            ("embedding", "VECTOR(384)"),
            ("processing_status", "STRING DEFAULT 'pending'"),
            ("created_at", "TIMESTAMPTZ DEFAULT now()"),
            ("processed_at", "TIMESTAMPTZ"),
        ), primary_key="document_id", indexes=(
            Index("idx_documents_user_type", "user_id, document_type, created_at DESC"),
        )),
        Table("query_cache", _c(
            ("cache_id", "UUID NOT NULL DEFAULT gen_random_uuid()"),
            ("query_hash", "STRING NOT NULL"),
            ("query_text", "STRING NOT NULL"),
            ("query_embedding", "VECTOR(384)"),
            ("response_text", "STRING NOT NULL"),
            ("response_tokens", "INT8 DEFAULT 0"),
            ("prompt_tokens", "INT8 DEFAULT 0"),
            ("ai_service", "STRING NOT NULL"),
            ("language", "STRING NOT NULL DEFAULT 'en'"),
            ("expense_data_hash", "STRING"),
            ("scope", "STRING NOT NULL DEFAULT ''"),
            ("created_at", "TIMESTAMPTZ DEFAULT now()"),
            ("expires_at", "TIMESTAMPTZ"),
            ("hit_count", "INT8 DEFAULT 0"),
            ("last_accessed", "TIMESTAMPTZ DEFAULT now()"),
        ), primary_key="cache_id", constraints=(
            "UNIQUE (query_hash, ai_service, language)",
        ), indexes=(
            Index("idx_query_hash", "query_hash"),
            Index("idx_expires_at", "expires_at"),
            Index("idx_ai_service", "ai_service"),
            Index("idx_expense_hash", "expense_data_hash"),
        )),
        Table("embedding_cache", _c(
            ("embedding_id", "UUID NOT NULL DEFAULT gen_random_uuid()"),
            ("text_hash", "STRING NOT NULL"),
            ("text_content", "STRING NOT NULL"),
            ("embedding", "VECTOR(384) NOT NULL"),
            ("model_name", "STRING NOT NULL DEFAULT 'all-MiniLM-L6-v2'"),
            ("created_at", "TIMESTAMPTZ DEFAULT now()"),
            ("access_count", "INT8 DEFAULT 0"),
        ), primary_key="embedding_id", constraints=(
            "UNIQUE (text_hash, model_name)",
        ), indexes=(
            Index("idx_text_hash", "text_hash"),
        )),
        Table("vector_search_cache", _c(
            ("search_id", "UUID NOT NULL DEFAULT gen_random_uuid()"),
            ("query_embedding_hash", "STRING NOT NULL UNIQUE"),
            ("search_results", "JSONB NOT NULL"),
            ("result_count", "INT8"),
            ("similarity_threshold", "FLOAT8"),
            ("created_at", "TIMESTAMPTZ DEFAULT now()"),
            ("expires_at", "TIMESTAMPTZ"),
            ("access_count", "INT8 DEFAULT 0"),
        ), primary_key="search_id", indexes=(
            Index("idx_embedding_hash", "query_embedding_hash"),
            Index("idx_expires_at", "expires_at"),
        )),
        Table("cache_stats", _c(
            ("stat_id", "UUID NOT NULL DEFAULT gen_random_uuid()"),
            ("cache_type", "STRING NOT NULL"),
            ("operation", "STRING NOT NULL"),
            ("tokens_saved", "INT8 DEFAULT 0"),
            ("timestamp", "TIMESTAMPTZ DEFAULT now()"),
            ("details", "JSONB"),
        ), primary_key="stat_id"),
        Table("users", _c(
            ("user_id", "UUID NOT NULL DEFAULT gen_random_uuid()"),
            ("username", "STRING NOT NULL UNIQUE"),
            ("spending_style", "STRING NOT NULL"),
            ("home_region", "STRING"),
            ("created_at", "TIMESTAMPTZ NOT NULL DEFAULT now()"),
            ("demo_user", "BOOL NOT NULL DEFAULT true"),
            ("display_name", "STRING"),
            ("avatar_color", "STRING"),
        ), primary_key="user_id", locality="GLOBAL"),
        Table("spending_signals", _c(
            ("signal_id", "UUID NOT NULL DEFAULT gen_random_uuid()"),
            ("user_id", "UUID NOT NULL"),
            ("signal_type", "STRING NOT NULL"),
            ("severity", "STRING NOT NULL"),
            ("payload", "JSONB NOT NULL"),
            ("produced_at", "TIMESTAMPTZ NOT NULL DEFAULT now()"),
            ("consumed_at", "TIMESTAMPTZ"),
            ("idempotency_key", "STRING NOT NULL UNIQUE"),
        ), primary_key="signal_id", indexes=(
            Index("spending_signals_user_id_produced_at_idx", "user_id, produced_at DESC"),
        ), locality="REGIONAL BY ROW",
            options=f"WITH (ttl_expire_after = '{cfg.coach_signal_ttl_days} days')"),
        Table("coach_nudges", _c(
            ("nudge_id", "UUID NOT NULL DEFAULT gen_random_uuid()"),
            ("signal_id", "UUID REFERENCES spending_signals(signal_id)"),
            ("user_id", "UUID NOT NULL"),
            ("message", "STRING NOT NULL"),
            ("tool_trace", "JSONB"),
            ("provider_used", "STRING"),
            ("trace_id", "STRING"),
            ("created_at", "TIMESTAMPTZ NOT NULL DEFAULT now()"),
        ), primary_key="nudge_id", indexes=(
            Index("coach_nudges_user_id_created_at_idx", "user_id, created_at DESC"),
        ), locality="REGIONAL BY ROW",
            options=f"WITH (ttl_expire_after = '{cfg.coach_nudge_ttl_days} days')"),
        Table("user_budgets", _c(
            ("user_id", "UUID NOT NULL"),
            ("category", "STRING NOT NULL"),
            ("amount", "DECIMAL NOT NULL"),
            ("updated_at", "TIMESTAMPTZ NOT NULL DEFAULT now()"),
        ), primary_key="user_id, category", locality="REGIONAL BY ROW"),
        Table("accounts", _c(
            ("account_id", "UUID NOT NULL DEFAULT gen_random_uuid()"),
            ("user_id", "UUID NOT NULL"),
            ("kind", "STRING NOT NULL"),
            ("name", "STRING NOT NULL"),
            ("institution", "STRING NOT NULL"),
            ("mask", "STRING NOT NULL"),
            ("payment_method", "STRING"),
            ("opening_balance", "DECIMAL(14,2) NOT NULL DEFAULT 0"),
            ("balance", "DECIMAL(14,2) NOT NULL DEFAULT 0"),
            ("credit_limit", "DECIMAL(14,2)"),
            ("apr", "DECIMAL(5,2)"),
            ("opened_at", "DATE NOT NULL"),
            ("created_at", "TIMESTAMPTZ NOT NULL DEFAULT now()"),
        ), primary_key="account_id", constraints=(
            "CHECK (kind IN ('checking', 'savings', 'credit'))",
        ), indexes=(
            Index("accounts_user_idx", "user_id", storing="kind, name, institution, mask, balance, credit_limit"),
        ), locality="REGIONAL BY ROW"),
        Table("transfers", _c(
            ("transfer_id", "UUID NOT NULL DEFAULT gen_random_uuid()"),
            ("user_id", "UUID NOT NULL"),
            ("kind", "STRING NOT NULL"),
            ("from_account", "UUID NOT NULL"),
            ("to_account", "UUID NOT NULL"),
            ("amount", "DECIMAL(14,2) NOT NULL"),
            ("goal_id", "UUID"),
            ("idempotency_key", "STRING NOT NULL"),
            ("created_at", "TIMESTAMPTZ NOT NULL DEFAULT now()"),
        ), primary_key="transfer_id", constraints=(
            "CHECK (kind IN ('transfer', 'card_payment', 'goal_contribution'))",
            "CHECK (amount > 0)",
            "UNIQUE (user_id, idempotency_key)",
        ), locality="REGIONAL BY ROW"),
        Table("ledger_entries", _c(
            ("entry_id", "UUID NOT NULL DEFAULT gen_random_uuid()"),
            ("user_id", "UUID NOT NULL"),
            ("account_id", "UUID NOT NULL"),
            ("posted_at", "TIMESTAMPTZ NOT NULL"),
            ("amount", "DECIMAL(14,2) NOT NULL"),
            ("kind", "STRING NOT NULL"),
            ("description", "STRING NOT NULL"),
            ("expense_id", "UUID"),
            ("transfer_id", "UUID"),
            ("goal_id", "UUID"),
        ), primary_key="entry_id", constraints=(
            "CHECK (kind IN ('purchase', 'income', 'transfer_in', 'transfer_out', "
            "'card_payment', 'interest', 'refund', 'adjustment'))",
        ), indexes=(
            Index("ledger_account_idx", "account_id, posted_at DESC", storing="amount, kind"),
            Index("ledger_user_idx", "user_id, posted_at DESC",
                  storing="account_id, amount, kind, description, expense_id"),
            Index("ledger_expense_idx", "expense_id"),
        ), locality="REGIONAL BY ROW"),
        Table("goals", _c(
            ("goal_id", "UUID NOT NULL DEFAULT gen_random_uuid()"),
            ("user_id", "UUID NOT NULL"),
            ("name", "STRING NOT NULL"),
            ("icon", "STRING NOT NULL"),
            ("target_amount", "DECIMAL(14,2) NOT NULL"),
            ("target_date", "DATE"),
            ("account_id", "UUID NOT NULL"),
            ("created_at", "TIMESTAMPTZ NOT NULL DEFAULT now()"),
        ), primary_key="goal_id", constraints=(
            "CHECK (target_amount > 0)",
        ), indexes=(
            Index("goals_user_idx", "user_id"),
        ), locality="REGIONAL BY ROW"),
        Table("receipts", _c(
            ("document_id", "UUID NOT NULL DEFAULT gen_random_uuid()"),
            ("user_id", "UUID NOT NULL"),
            ("kind", "STRING NOT NULL DEFAULT 'receipt'"),
            ("filename", "STRING"),
            ("mime", "STRING"),
            ("status", "STRING NOT NULL"),
            ("error", "STRING"),
            ("ocr_text", "STRING"),
            ("extracted", "JSONB"),
            ("expense_id", "UUID"),
            ("thumbnail", "BYTES"),
            ("embedding", "VECTOR(384)"),
            ("created_at", "TIMESTAMPTZ NOT NULL DEFAULT now()"),
        ), primary_key="document_id", constraints=(
            "CHECK (status IN ('processing', 'posted', 'duplicate', 'failed'))",
        ), indexes=(
            Index("receipts_user_idx", "user_id, created_at DESC"),
            Index("receipts_user_embedding", "user_id, embedding vector_cosine_ops", vector=True),
            Index("receipts_expense_idx", "expense_id"),
        ), locality="REGIONAL BY ROW"),
        Table("chat_threads", _c(
            ("thread_id", "UUID NOT NULL DEFAULT gen_random_uuid()"),
            ("user_id", "UUID NOT NULL"),
            ("title", "STRING NOT NULL DEFAULT 'New chat'"),
            ("message_count", "INT8 NOT NULL DEFAULT 0"),
            ("created_at", "TIMESTAMPTZ NOT NULL DEFAULT now()"),
            ("last_message_at", "TIMESTAMPTZ NOT NULL DEFAULT now()"),
        ), primary_key="thread_id", indexes=(
            Index("chat_threads_user_idx", "user_id, last_message_at DESC", storing="title, message_count"),
        ), locality="REGIONAL BY ROW"),
        Table("memories", _c(
            ("memory_id", "UUID NOT NULL DEFAULT gen_random_uuid()"),
            ("user_id", "UUID NOT NULL"),
            ("kind", "STRING NOT NULL"),
            ("content", "STRING NOT NULL"),
            ("embedding", "VECTOR(384) NOT NULL"),
            ("source_thread_id", "UUID"),
            ("use_count", "INT8 NOT NULL DEFAULT 0"),
            ("created_at", "TIMESTAMPTZ NOT NULL DEFAULT now()"),
            ("last_used_at", "TIMESTAMPTZ"),
        ), primary_key="memory_id", constraints=(
            "CHECK (kind IN ('fact', 'preference', 'goal'))",
        ), indexes=(
            Index("memories_user_embedding", "user_id, embedding vector_cosine_ops", vector=True),
            Index("memories_user_idx", "user_id, created_at DESC"),
        ), locality="REGIONAL BY ROW"),
        Table("answer_cache", _c(
            ("cache_id", "UUID NOT NULL DEFAULT gen_random_uuid()"),
            ("user_id", "UUID NOT NULL"),
            ("data_version", "INT8 NOT NULL"),
            ("language", "STRING NOT NULL"),
            ("query_text", "STRING NOT NULL"),
            ("query_embedding", "VECTOR(384) NOT NULL"),
            ("answer", "STRING NOT NULL"),
            ("tokens", "INT8 NOT NULL DEFAULT 0"),
            ("hits", "INT8 NOT NULL DEFAULT 0"),
            ("created_at", "TIMESTAMPTZ NOT NULL DEFAULT now()"),
            ("expires_at", "TIMESTAMPTZ NOT NULL"),
        ), primary_key="cache_id", indexes=(
            Index("answer_cache_user_embedding", "user_id, query_embedding vector_cosine_ops", vector=True),
            Index("answer_cache_user_idx", "user_id, data_version"),
        ), locality="REGIONAL BY ROW", options="WITH (ttl_expiration_expression = 'expires_at')"),
        Table("search_cache", _c(
            ("cache_key", "STRING NOT NULL"),
            ("user_id", "UUID NOT NULL"),
            ("results", "JSONB NOT NULL"),
            ("hits", "INT8 NOT NULL DEFAULT 0"),
            ("created_at", "TIMESTAMPTZ NOT NULL DEFAULT now()"),
            ("expires_at", "TIMESTAMPTZ NOT NULL"),
        ), primary_key="cache_key", locality="REGIONAL BY ROW",
            options="WITH (ttl_expiration_expression = 'expires_at')"),
        Table("assistant_actions", _c(
            ("action_id", "UUID NOT NULL DEFAULT gen_random_uuid()"),
            ("user_id", "UUID NOT NULL"),
            ("thread_id", "UUID"),
            ("kind", "STRING NOT NULL"),
            ("args", "JSONB NOT NULL"),
            ("summary", "STRING NOT NULL"),
            ("status", "STRING NOT NULL DEFAULT 'pending'"),
            ("created_at", "TIMESTAMPTZ NOT NULL DEFAULT now()"),
            ("expires_at", "TIMESTAMPTZ NOT NULL"),
        ), primary_key="action_id", constraints=(
            "CHECK (kind IN ('recategorize', 'budget'))",
            "CHECK (status IN ('pending', 'done', 'cancelled'))",
        ), indexes=(
            Index("assistant_actions_user_idx", "user_id, created_at DESC"),
        ), locality="REGIONAL BY ROW", options="WITH (ttl_expiration_expression = 'expires_at')"),
        Table("user_data_versions", _c(
            ("user_id", "UUID NOT NULL"),
            ("version", "INT8 NOT NULL DEFAULT 0"),
            ("updated_at", "TIMESTAMPTZ NOT NULL DEFAULT now()"),
        ), primary_key="user_id", locality="REGIONAL BY ROW"),
    )


def fingerprint(topo: Topology, *, rls: bool = False) -> str:
    rendered = "\n".join(t.create_sql(topo) for t in tables())
    digest = hashlib.sha256(f"{REGISTRY_VERSION}|{topo.regional}|rls={rls}|{rendered}".encode()).hexdigest()
    return digest[:16]


def plan_locality_conversions(current: dict[str, str], topo: Topology) -> list[str]:
    """ALTERs for existing tables whose locality differs from the registry.
    Tables that do not exist yet are created inline and need none."""
    if not topo.regional:
        return []
    return [f"ALTER TABLE {t.name} SET LOCALITY {t.locality}"
            for t in tables()
            if t.locality and t.name in current and current[t.name] != t.locality]


# Conditional upgrades for databases created by older releases.
Upgrade = Callable[[Connection, Topology], list[str]]


def _naive_timestamp_upgrade(conn: Connection, topo: Topology) -> list[str]:
    ours = {t.name for t in tables()}
    rows = conn.execute(text("""
        SELECT table_name, column_name FROM information_schema.columns
        WHERE table_schema = 'public' AND data_type = 'timestamp without time zone'
    """)).fetchall()
    todo = [(t, c) for t, c in rows if t in ours]
    if not todo:
        return []
    statements = ["SET enable_experimental_alter_column_type_general = true"]
    statements += [f'ALTER TABLE {t} ALTER COLUMN "{c}" TYPE TIMESTAMPTZ' for t, c in todo]
    statements.append("RESET enable_experimental_alter_column_type_general")
    return statements


def _create_ddl(conn: Connection, table: str) -> str:
    row = conn.execute(text(f"SHOW CREATE TABLE {table}")).fetchone()
    return "" if row is None else str(row[1])


def _users_ttl_upgrade(conn: Connection, topo: Topology) -> list[str]:
    # Row-level TTL on users swept personas and orphaned their expenses; the
    # boot-time demo-user sweep replaced it.
    try:
        ddl = _create_ddl(conn, "users")
        return ["ALTER TABLE users RESET (ttl)"] if "ttl_expire_after" in ddl else []
    except ProgrammingError:  # users table doesn't exist yet
        return []


_TTL = re.compile(r"ttl_expire_after = '(\d+) days'")


def _ttl_drift_upgrade(conn: Connection, topo: Topology) -> list[str]:
    cfg = get_config()
    out = []
    for table, days in (("spending_signals", cfg.coach_signal_ttl_days), ("coach_nudges", cfg.coach_nudge_ttl_days)):
        try:
            match = _TTL.search(_create_ddl(conn, table))
            if not match or int(match.group(1)) != days:
                out.append(f"ALTER TABLE {table} SET (ttl_expire_after = '{days} days')")
        except ProgrammingError:  # table doesn't exist yet
            continue
    return out


UPGRADES: tuple[Upgrade, ...] = (_naive_timestamp_upgrade, _users_ttl_upgrade, _ttl_drift_upgrade)


@dataclass
class SchemaResult:
    changed: bool
    fingerprint: str
    statements: int = 0
    pending_conversions: list[str] = field(default_factory=list)
    seconds: float = 0.0


def _stored(conn: Connection) -> dict | None:
    try:
        row = conn.execute(text("SELECT value FROM banko_meta WHERE key = :k"), {"k": FINGERPRINT_KEY}).fetchone()
    except ProgrammingError:  # banko_meta does not exist yet
        return None
    return json.loads(row[0]) if row else None


def _cluster_regions(conn: Connection) -> list[str]:
    return [str(r[0]) for r in conn.execute(text("SHOW REGIONS FROM CLUSTER")).fetchall()]


def _database_regions(conn: Connection) -> list[tuple[str, bool]]:
    return [(str(r[0]), bool(r[1])) for r in conn.execute(
        text('SELECT region, "primary" FROM [SHOW REGIONS FROM DATABASE]')).fetchall()]


def detect_topology(conn: Connection) -> Topology:
    db = _database_regions(conn)
    return Topology(regional=any(primary for _, primary in db), regions=tuple(sorted(r for r, _ in db)))


def _bootstrap_regions(conn: Connection, preferred: str | None) -> list[str]:
    """A fresh database on a multi-region cluster gets the cluster's
    regions, a primary, and region survival when there are three or more.
    A database someone else configured (the chaos demo's init.sql owns
    defaultdb there) is left alone."""
    if any(primary for _, primary in _database_regions(conn)):
        return []
    regions = _cluster_regions(conn)
    if len(regions) < 2:
        return []
    db = conn.execute(text("SELECT current_database()")).scalar()
    gateway = conn.execute(text("SELECT gateway_region()")).scalar()
    primary = preferred if preferred in regions else (gateway if gateway in regions else regions[0])
    statements = [f'ALTER DATABASE "{db}" SET PRIMARY REGION "{primary}"']
    statements += [f'ALTER DATABASE "{db}" ADD REGION IF NOT EXISTS "{r}"' for r in regions if r != primary]
    if len(regions) >= 3:
        statements.append(f'ALTER DATABASE "{db}" SURVIVE REGION FAILURE')
    return statements


def _tolerant(conn: Connection, sql: str) -> None:
    """Run DDL that a concurrent boot may already have run."""
    try:
        conn.execute(text(sql))
        narration.info("%s", sql)
    except DBAPIError as e:
        if "already" in str(e).lower():
            log.debug("skipped (already applied): %s", sql)
        else:
            raise


def current_localities(conn: Connection) -> dict[str, str]:
    return {str(r[0]): str(r[1]) for r in conn.execute(
        text("SELECT table_name, locality FROM [SHOW TABLES] WHERE schema_name = 'public'")).fetchall()}


def _retrying(fn: Callable[[], object], attempts: int = 4) -> object:
    """Concurrent boots race on the same rows and DDL; under SERIALIZABLE
    the loser gets 40001 and simply tries again."""
    for attempt in range(attempts):
        try:
            return fn()
        except DBAPIError as e:
            if getattr(getattr(e, "orig", None), "sqlstate", "") == "40001" and attempt < attempts - 1:
                time.sleep(0.2 * (attempt + 1))
                continue
            raise
    raise AssertionError("unreachable")


def _ensure_meta(conn: Connection, topo: Topology) -> None:
    meta = next(t for t in tables() if t.name == "banko_meta")

    def create() -> None:
        try:
            conn.execute(text(meta.create_sql(topo)))
        except DBAPIError as e:
            if "already exists" not in str(e).lower():
                raise

    _retrying(create)


def _acquire_lock(conn: Connection, owner: str) -> bool:
    def attempt() -> bool:
        row = conn.execute(text(f"""
            INSERT INTO banko_meta (key, value, updated_at) VALUES (:k, :owner, now())
            ON CONFLICT (key) DO UPDATE SET value = excluded.value, updated_at = now()
            WHERE banko_meta.updated_at < now() - INTERVAL '{LOCK_TTL_SECONDS} seconds'
            RETURNING value
        """), {"k": LOCK_KEY, "owner": owner}).fetchone()
        return row is not None and row[0] == owner

    return bool(_retrying(attempt))


def _release_lock(conn: Connection, owner: str) -> None:
    conn.execute(text("DELETE FROM banko_meta WHERE key = :k AND value = :owner"), {"k": LOCK_KEY, "owner": owner})


def _wait_for(conn: Connection, want: str, owner: str) -> bool:
    """Wait for another process to finish applying the schema, or take over
    a stale lease. Returns True if we took over and should apply ourselves."""
    deadline = time.monotonic() + LOCK_TTL_SECONDS
    while time.monotonic() < deadline:
        stored = _stored(conn)
        if stored and stored.get("fingerprint") == want:
            return False  # schema is current, no need to apply
        # Try to acquire the lock in case the other process died
        if _acquire_lock(conn, owner):
            return True  # we got the lock, caller should apply
        time.sleep(1.0)
    raise TimeoutError("gave up waiting for another process to finish applying the schema")


def _refresh_lease(conn: Connection, owner: str) -> None:
    """Refresh the lease's updated_at between phases. Raises if the lease was
    taken over, preventing two appliers from running DDL concurrently."""
    result = conn.execute(text(
        "UPDATE banko_meta SET updated_at = now() WHERE key = :k AND value = :owner"),
        {"k": LOCK_KEY, "owner": owner})
    if result.rowcount == 0:
        raise RuntimeError(f"lease was taken over by another process (owner {owner} no longer holds it)")


def _apply_policies_in_parallel(conn: Connection, per_table: dict[str, list[str]]) -> int:
    """Run each table's statements in order on its own autocommit connection,
    all tables at once. The first failure re-raises here."""
    engine = conn.engine

    def run(statements: list[str]) -> int:
        with engine.connect() as worker:
            worker.execution_options(isolation_level="AUTOCOMMIT")
            for sql in statements:
                log.info("schema: %s", sql)
                worker.execute(text(sql))
        return len(statements)

    with ThreadPoolExecutor(max_workers=max(1, len(per_table)), thread_name_prefix="banko-rls") as pool:
        futures = [pool.submit(run, statements) for statements in per_table.values()]
        return sum(f.result() for f in futures)


def _apply(conn: Connection, topo: Topology, owner: str, *, rls_on: bool = False) -> int:
    existing = set(current_localities(conn))
    is_fresh = len(existing) <= 1  # only banko_meta exists
    count = 0
    roles_batched = False

    # Batch-create all missing tables in one transaction for fresh databases
    missing_tables = [t for t in tables() if t.name not in existing]
    if missing_tables:
        create_started = time.monotonic()
        if is_fresh:
            # Fresh database: batch all table creates in one transaction
            log.info("schema: batch-creating %d tables", len(missing_tables))
            # The role and its GRANTs ride in the same transaction. The
            # policy DDL can't (see policy_statements), so it runs after.
            roles = role_statements([t.name for t in tables()], regional=topo.regional) if rls_on else []
            # Use a fresh connection from the shared engine in a transaction
            engine = conn.engine
            with engine.begin() as txn:
                # Disable autocommit_before_ddl so statements don't commit individually
                txn.execute(text("SET LOCAL autocommit_before_ddl = false"))
                for table in missing_tables:
                    sql = table.create_sql(topo)
                    log.info("schema: %s", sql.splitlines()[0])
                    txn.execute(text(sql))
                for sql in roles:
                    log.info("schema: %s", sql)
                    txn.execute(text(sql))
            count += len(missing_tables) + len(roles)
            roles_batched = bool(roles)
            log.info("schema: created %d tables in %.1fs", len(missing_tables), time.monotonic() - create_started)
            # Refresh lease after batch transaction commits
            _refresh_lease(conn, owner)
        else:
            # Incremental: create tables one by one
            for table in missing_tables:
                sql = table.create_sql(topo)
                log.info("schema: %s", sql.splitlines()[0])
                conn.execute(text(sql))
                count += 1
                _refresh_lease(conn, owner)

    # For existing tables, add missing columns and indexes (skip on fresh database)
    if not is_fresh:
        for table in tables():
            if table.name in existing:
                have_columns = {str(r[0]) for r in conn.execute(text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema = 'public' AND table_name = :t"), {"t": table.name}).fetchall()}
                for c in table.columns:
                    if c.name not in have_columns:
                        sql = f"ALTER TABLE {table.name} ADD COLUMN IF NOT EXISTS {c.name} {c.definition}"
                        log.info("schema: %s", sql)
                        conn.execute(text(sql))
                        count += 1
                        _refresh_lease(conn, owner)
                have_indexes = {str(r[0]) for r in conn.execute(
                    text(f"SELECT DISTINCT index_name FROM [SHOW INDEXES FROM {table.name}]")).fetchall()}
                for i in table.indexes:
                    if i.name not in have_indexes:
                        sql = i.create(table.name)
                        log.info("schema: %s", sql)
                        conn.execute(text(sql))
                        count += 1
                        _refresh_lease(conn, owner)

    # Run upgrade functions (skip on fresh database as tables don't exist yet)
    if not is_fresh:
        upgrade_started = time.monotonic()
        for upgrade in UPGRADES:
            statements = upgrade(conn, topo)
            # A trailing RESET undoes a session setting the upgrade needed;
            # it runs even when one of the statements before it fails, so
            # the pooled connection never goes back with the setting on.
            cleanup = [sql for sql in statements if sql.startswith("RESET ")]
            try:
                for sql in statements:
                    if sql in cleanup:
                        continue
                    log.info("schema: %s", sql.splitlines()[0])
                    conn.execute(text(sql))
                    count += 1
                    _refresh_lease(conn, owner)
            finally:
                for sql in cleanup:
                    conn.execute(text(sql))
                    count += 1
        if count > len(missing_tables):
            log.info("schema: ran upgrades in %.1fs", time.monotonic() - upgrade_started)

    # Row-level security on the money tables, never on vector tables.
    if rls_on and roles_batched:
        # Fresh database: no request can be live yet, and each money table's
        # pair of statements is independent of the others, so the tables run
        # side by side. On a multi-region cluster each statement is a schema
        # change job of several seconds; one after another they took ~50 s.
        rls_started = time.monotonic()
        count += _apply_policies_in_parallel(conn, policy_statements())
        log.info("schema: row-level security in %.1fs", time.monotonic() - rls_started)
        _refresh_lease(conn, owner)
    elif rls_on:
        # Tables that already have the policy get ALTER POLICY: no window
        # where a live banko_app request meets RLS without a policy.
        for sql in rls_statements([t.name for t in tables()], regional=topo.regional,
                                  existing=policy_tables(conn)):
            log.info("schema: %s", sql)
            conn.execute(text(sql))
            count += 1
        _refresh_lease(conn, owner)

    return count


def _write_fingerprint(conn: Connection, fp: str, topo: Topology, *, rls_on: bool = False) -> None:
    value = json.dumps({"fingerprint": fp, "regional": topo.regional, "version": REGISTRY_VERSION,
                        "rls": rls_on, "rls_mode": get_config().rls_mode})
    conn.execute(text("UPSERT INTO banko_meta (key, value, updated_at) VALUES (:k, :v, now())"),
                 {"k": FINGERPRINT_KEY, "v": value})


def ensure_schema(url: str | None = None, *, force: bool = False,
                  primary_region: str | None = None,
                  on_progress: Callable[[str], None] | None = None) -> SchemaResult:
    """Make the database match the registry. One SELECT when it already does.
    on_progress hears a short human sentence when there is real work to do,
    so a one-time upgrade on a big database doesn't look like a hang."""
    started = time.monotonic()

    def progress(detail: str) -> None:
        if on_progress is not None:
            on_progress(detail)

    with get_engine(url).connect() as conn:
        conn.execution_options(isolation_level="AUTOCOMMIT")
        stored = None if force else _stored(conn)
        # The fast path stays one SELECT: whether RLS was applied comes from
        # the stored value, and a changed RLS_MODE forces the slow path.
        if (stored and stored.get("rls_mode") == get_config().rls_mode
                and stored.get("fingerprint") == fingerprint(Topology(regional=bool(stored.get("regional"))),
                                                             rls=bool(stored.get("rls")))):
            return SchemaResult(changed=False, fingerprint=stored["fingerprint"],
                                seconds=time.monotonic() - started)

        # Phase 1: Bootstrap regions (if multi-region cluster)
        bootstrap_started = time.monotonic()
        bootstrap_stmts = _bootstrap_regions(conn, primary_region)
        for sql in bootstrap_stmts:
            _tolerant(conn, sql)
        if bootstrap_stmts:
            log.info("schema: bootstrap phase %.1fs (%d statements)",
                     time.monotonic() - bootstrap_started, len(bootstrap_stmts))

        topo = detect_topology(conn)
        _ensure_meta(conn, topo)
        # After _ensure_meta: supports_rls probes SHOW POLICIES on banko_meta.
        rls_on = get_config().rls_mode != "off" and supports_rls(conn)
        want = fingerprint(topo, rls=rls_on)

        owner = uuid.uuid4().hex
        if not _acquire_lock(conn, owner):
            log.info("schema: another process is applying the schema, waiting for it")
            progress("Waiting for another Banko process to finish the schema.")
            should_apply = _wait_for(conn, want, owner)
            if not should_apply:
                return SchemaResult(changed=False, fingerprint=want, seconds=time.monotonic() - started)
            # We took over a stale lease, proceed to apply
        try:
            stored = None if force else _stored(conn)
            if stored and stored.get("fingerprint") == want:
                return SchemaResult(changed=False, fingerprint=want, seconds=time.monotonic() - started)
            log.info("schema: applying registry v%d (%s)", REGISTRY_VERSION,
                     "multi-region" if topo.regional else "single-region")

            # Refresh lease after acquiring it (especially after bootstrap phase if regions were added)
            if bootstrap_stmts:
                _refresh_lease(conn, owner)

            # Phase 2: Apply schema (tables, columns, indexes, upgrades)
            if len(current_localities(conn)) > 1:  # more than banko_meta
                progress("Upgrading the schema (one-time). On a populated database this can take a minute.")
            else:
                progress("Creating the schema.")
            apply_started = time.monotonic()
            count = _apply(conn, topo, owner, rls_on=rls_on)
            log.info("schema: apply phase %.1fs (%d statements)",
                     time.monotonic() - apply_started, count)

            # Phase 3: Plan locality conversions and write fingerprint
            fp_started = time.monotonic()
            conversions = plan_locality_conversions(current_localities(conn), topo)
            _write_fingerprint(conn, want, topo, rls_on=rls_on)
            log.info("schema: fingerprint phase %.1fs", time.monotonic() - fp_started)

            return SchemaResult(changed=True, fingerprint=want, statements=count,
                                pending_conversions=conversions, seconds=time.monotonic() - started)
        finally:
            _release_lock(conn, owner)


def run_pending_conversions(url: str | None = None) -> list[str]:
    """Background work after boot: re-apply the registry if the database
    became multi-region since the fingerprint was written, then convert
    any table whose locality still differs. Narrates like the old migration."""
    with get_engine(url).connect() as conn:
        conn.execution_options(isolation_level="AUTOCOMMIT")
        topo = detect_topology(conn)
        stored = _stored(conn)

        # If topology changed, or cluster has regions but DB doesn't, reapply
        cluster_regions = _cluster_regions(conn)
        db_has_primary = any(primary for _, primary in _database_regions(conn))
        if stored is not None and bool(stored.get("regional")) != topo.regional:
            log.info("schema: database topology changed since the schema was applied, re-applying")
            ensure_schema(url, force=True)
        elif len(cluster_regions) >= 2 and not db_has_primary:
            log.info("schema: cluster has regions but database doesn't, bootstrapping")
            ensure_schema(url, force=True)

        if not topo.regional:
            return []
        current = current_localities(conn)
        planned = plan_locality_conversions(current, topo)
        planned_tables = {sql.split()[2] for sql in planned}
        for table in tables():
            if table.locality and table.name in current and table.name not in planned_tables:
                narration.info("%s already %s", table.name, table.locality)
        done = []
        for sql in planned:
            name, locality = sql.split()[2], sql.split("SET LOCALITY ", 1)[1]
            narration.info("converting %s to %s (a real schema change, can take a minute or two "
                           "on a multi-node cluster)", name, locality)
            try:
                conn.execute(text(sql))
                narration.info("set %s to %s", name, locality)
                done.append(sql)
            except DBAPIError as e:
                if "already" in str(e).lower():
                    narration.info("%s already %s (concurrent boot)", name, locality)
                    done.append(sql)
                else:
                    raise
        narration.info("multi-region migration completed successfully")
        from .regions import clear_region_caches
        clear_region_caches()
        return done
