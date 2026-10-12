"""Banko settings.

One typed model is the only place the app reads configuration. Values come
from the environment first, then an optional YAML file named by BANKO_CONFIG,
then the defaults below. `banko-ai config` prints what is in effect and where
each value came from, and the README reference is generated from these same
definitions, so the documentation can't drift from what the code honors.
"""

from __future__ import annotations

import logging
import os
import secrets
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import yaml
from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

log = logging.getLogger("banko.config")

PROVIDERS = ("watsonx", "openai", "aws", "gemini", "ollama")

GROUPS = {
    "server": "Server",
    "database": "Database",
    "ai": "AI providers",
    "display": "Provider persona",
    "embeddings": "Embeddings",
    "cache": "Caching",
    "data": "Demo data",
    "coach": "Coach and CDC",
}

# Old names that still work, and the setting each one now feeds.
DEPRECATED_ALIASES = {
    "WATSONX_MODEL_ID": "WATSONX_MODEL",
    "AWS_MODEL_ID": "AWS_MODEL",
    "CACHE_THRESHOLD": "CACHE_SIMILARITY_THRESHOLD",
    "WATSONX_URL": "WATSONX_API_URL",
    "GEMINI_API_KEY": "GOOGLE_API_KEY",
}

# Settings that existed before v2 and now do nothing. Setting one is not an
# error, but people should hear about it once instead of wondering why it
# has no effect.
REMOVED_SETTINGS = {
    "DATABASE_HOST": "DATABASE_URL carries the host",
    "DATABASE_PORT": "DATABASE_URL carries the port",
    "DATABASE_NAME": "DATABASE_URL carries the database",
    "DATABASE_USER": "DATABASE_URL carries the user",
    "DATABASE_PASSWORD": "DATABASE_URL carries the password",
    "DATABASE_SSL_MODE": "put sslmode= in DATABASE_URL",
    "VECTOR_DIMENSIONS": "the schema fixes VECTOR(384)",
    "SIMILARITY_THRESHOLD": "use CACHE_SIMILARITY_THRESHOLD",
    "CACHE_ENABLED": "caching is always on",
    "CACHE_TTL": "use CACHE_TTL_HOURS",
    "DEFAULT_RECORD_COUNT": "use SAMPLE_DATA_COUNT or banko-ai run --generate-data",
    "DEFAULT_USER_COUNT": "nothing reads it",
    "COACH_RATE_LIMIT_PER_5MIN": "nothing reads it",
    "COACH_VELOCITY_HORIZON_DAYS": "nothing reads it",
    "COACH_TOP_MERCHANTS_K": "nothing reads it",
    "COACH_SUBSCRIPTION_MIN_OCCURRENCES": "use RECURRING_MIN_OCCURRENCES",
}

_WATSONX_CHAT_PATH = "/ml/v1/text/chat?version=2023-05-29"

_warned: set[str] = set()


@dataclass(frozen=True)
class FieldMeta:
    env: str
    group: str
    secret: bool
    aliases: tuple[str, ...]


# The same metadata as json_schema_extra, keyed by env name, with real types.
_META: dict[str, FieldMeta] = {}


def _field(default: Any, *, group: str, env: str, description: str,
           secret: bool = False, aliases: tuple[str, ...] = ()) -> Any:
    _META[env] = FieldMeta(env=env, group=group, secret=secret, aliases=aliases)
    return Field(
        default,
        validation_alias=AliasChoices(env, *aliases),
        description=description,
        json_schema_extra={"group": group, "env": env, "secret": secret, "aliases": list(aliases)},
    )


def field_meta(field: FieldInfo) -> FieldMeta:
    """Typed metadata for one Settings field."""
    alias = field.validation_alias
    if not isinstance(alias, AliasChoices) or not isinstance(alias.choices[0], str):
        raise TypeError("Settings fields are declared through _field()")
    return _META[alias.choices[0]]


@dataclass(frozen=True)
class SettingInfo:
    env: str
    group: str
    value: str
    source: str
    description: str


class Settings(BaseSettings):
    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore", populate_by_name=True)

    # Server
    secret_key: str = _field("", group="server", env="SECRET_KEY", secret=True, description="Flask session secret. Required under gunicorn or FLASK_ENV=production; a random one is generated per run otherwise.")
    flask_env: str = _field("development", group="server", env="FLASK_ENV", description="Set to production to make a missing SECRET_KEY fatal.")
    debug: bool = _field(False, group="server", env="DEBUG", description="Flask debug mode. Leave it off for demos.")
    host: str = _field("0.0.0.0", group="server", env="HOST", description="Address `banko-ai run` binds.")
    port: int = _field(5000, group="server", env="PORT", description="Port `banko-ai run` and the Docker image listen on.")
    socketio_cors_origins: str = _field("", group="server", env="SOCKETIO_CORS_ORIGINS", description="Comma-separated origins allowed to open Socket.IO connections. Empty means same origin only.")
    max_upload_mb: int = _field(10, group="server", env="MAX_UPLOAD_MB", description="Largest request body accepted, in megabytes. It caps every request, receipt uploads included.")
    session_cookie_secure: bool = _field(False, group="server", env="SESSION_COOKIE_SECURE", description="Mark the session cookie Secure. Turn it on when serving over HTTPS.")
    warmup_wait_seconds: int = _field(20, group="server", env="WARMUP_WAIT_SECONDS", description="How long a request waits for background warm-up (schema, embedding model) before getting a friendly still-warming-up reply.")

    # Database
    database_url: str = _field("cockroachdb://root@localhost:26257/defaultdb?sslmode=disable", group="database", env="DATABASE_URL", secret=True, description="CockroachDB connection URL. postgresql:// and cockroachdb:// both work.")
    db_pool_size: int = _field(20, group="database", env="DB_POOL_SIZE", description="Connections kept open in the shared pool.")
    db_max_overflow: int = _field(10, group="database", env="DB_MAX_OVERFLOW", description="Extra connections allowed during bursts.")
    db_pool_timeout: int = _field(30, group="database", env="DB_POOL_TIMEOUT", description="Seconds to wait for a pooled connection.")
    db_pool_recycle: int = _field(1800, group="database", env="DB_POOL_RECYCLE", description="Recycle pooled connections after this many seconds.")
    db_pool_pre_ping: bool = _field(True, group="database", env="DB_POOL_PRE_PING", description="Check a connection before using it, so a restarted node or a haproxy reset stays invisible.")
    db_connect_timeout: int = _field(10, group="database", env="DB_CONNECT_TIMEOUT", description="Seconds to wait when opening a connection.")
    db_retry_max_attempts: int = _field(5, group="database", env="DB_RETRY_MAX_ATTEMPTS", description="Attempts for a transaction that hits a retryable error (40001, a dropped connection).")
    rls_mode: str = _field("auto", group="database", env="RLS_MODE",
        description="auto turns on row-level security for the money tables when the app role can log in; off keeps it off.")
    database_app_url: str = _field("", group="database", env="DATABASE_APP_URL", secret=True,
        description="Connection URL for the non-admin app role used with row-level security. Empty derives it from DATABASE_URL on insecure clusters.")

    # AI providers
    ai_service: str = _field("watsonx", group="ai", env="AI_SERVICE", description="Provider that actually serves requests: watsonx, openai, aws, gemini or ollama.")
    watsonx_api_key: str | None = _field(None, group="ai", env="WATSONX_API_KEY", secret=True, description="IBM Cloud API key.")
    watsonx_project_id: str | None = _field(None, group="ai", env="WATSONX_PROJECT_ID", description="watsonx.ai project id.")
    watsonx_model: str = _field("openai/gpt-oss-120b", group="ai", env="WATSONX_MODEL", aliases=("WATSONX_MODEL_ID",), description="Default watsonx model.")
    watsonx_models: str = _field("", group="ai", env="WATSONX_MODELS", description="Comma-separated models for the dropdown. Empty means discover them.")
    watsonx_api_url: str = _field("https://us-south.ml.cloud.ibm.com" + _WATSONX_CHAT_PATH, group="ai", env="WATSONX_API_URL", aliases=("WATSONX_URL",), description="watsonx endpoint for your region. A bare host such as https://eu-de.ml.cloud.ibm.com also works.")
    watsonx_token_url: str = _field("https://iam.cloud.ibm.com/identity/token", group="ai", env="WATSONX_TOKEN_URL", description="IBM IAM token endpoint.")
    watsonx_timeout: int = _field(30, group="ai", env="WATSONX_TIMEOUT", description="watsonx request timeout in seconds.")
    openai_api_key: str | None = _field(None, group="ai", env="OPENAI_API_KEY", secret=True, description="OpenAI API key (any value works for local compatible endpoints).")
    openai_base_url: str = _field("", group="ai", env="OPENAI_BASE_URL", description="OpenAI-compatible endpoint, for example http://localhost:11434/v1. Empty means api.openai.com.")
    openai_model: str = _field("gpt-4o-mini", group="ai", env="OPENAI_MODEL", description="Default OpenAI model.")
    openai_models: str = _field("", group="ai", env="OPENAI_MODELS", description="Comma-separated models for the dropdown. Empty means discover them.")
    ollama_host: str = _field("http://localhost:11434", group="ai", env="OLLAMA_HOST", description="Ollama daemon URL.")
    ollama_model: str = _field("granite3.3:8b", group="ai", env="OLLAMA_MODEL", description="Default Ollama model.")
    ollama_timeout_seconds: int = _field(120, group="ai", env="OLLAMA_TIMEOUT_SECONDS", description="Ollama request timeout in seconds.")
    ollama_num_ctx: int = _field(8192, group="ai", env="OLLAMA_NUM_CTX", description="Context window Ollama loads the model with. The daemon default (32k) makes granite3.3:8b take about 10 GB; 8192 fits the demo prompts in far less memory.")
    ollama_keep_alive: str = _field("5m", group="ai", env="OLLAMA_KEEP_ALIVE", description="How long Ollama keeps the model loaded after the last request, for example 5m or 1h. 0 unloads it right after each answer.")
    aws_access_key_id: str | None = _field(None, group="ai", env="AWS_ACCESS_KEY_ID", secret=True, description="AWS access key id (or use AWS_PROFILE).")
    aws_secret_access_key: str | None = _field(None, group="ai", env="AWS_SECRET_ACCESS_KEY", secret=True, description="AWS secret access key.")
    aws_profile: str | None = _field(None, group="ai", env="AWS_PROFILE", description="AWS profile name, for SSO.")
    aws_region: str = _field("us-east-1", group="ai", env="AWS_REGION", description="Bedrock region.")
    aws_model: str = _field("us.anthropic.claude-haiku-4-5-20251001-v1:0", group="ai", env="AWS_MODEL", aliases=("AWS_MODEL_ID",), description="Default Bedrock model or inference profile.")
    aws_models: str = _field("", group="ai", env="AWS_MODELS", description="Comma-separated models for the dropdown. Empty means discover them.")
    google_project_id: str | None = _field(None, group="ai", env="GOOGLE_PROJECT_ID", description="Google Cloud project for Vertex AI.")
    google_location: str = _field("us-central1", group="ai", env="GOOGLE_LOCATION", description="Vertex AI region.")
    google_model: str = _field("gemini-2.0-flash-001", group="ai", env="GOOGLE_MODEL", description="Default Gemini model.")
    gemini_models: str = _field("", group="ai", env="GEMINI_MODELS", description="Comma-separated models for the dropdown. Empty means discover them.")
    google_application_credentials: str | None = _field(None, group="ai", env="GOOGLE_APPLICATION_CREDENTIALS", description="Path to a service account JSON for Vertex AI.")
    google_api_key: str | None = _field(None, group="ai", env="GOOGLE_API_KEY", aliases=("GEMINI_API_KEY",), secret=True, description="Gemini API key, used when Vertex AI is unavailable.")

    # Provider persona
    ai_display_provider: str = _field("", group="display", env="AI_DISPLAY_PROVIDER", description="Provider the UI shows (watsonx, openai, aws, gemini or ollama). Empty means show the real one.")
    ai_display_model: str = _field("", group="display", env="AI_DISPLAY_MODEL", description="Model name the UI shows under the persona.")
    ai_display_models: str = _field("", group="display", env="AI_DISPLAY_MODELS", description="Comma-separated models the dropdown offers under the persona. Picking one never changes the real model.")

    # Embeddings
    embedding_model: str = _field("all-MiniLM-L6-v2", group="embeddings", env="EMBEDDING_MODEL", description="sentence-transformers model. It must produce 384-dim vectors to match the schema.")
    embedding_device: str = _field("cpu", group="embeddings", env="EMBEDDING_DEVICE", description="Torch device for embeddings. cpu is fastest for one query at a time and avoids MPS reserving a gigabyte on Macs.")

    # Caching
    cache_similarity_threshold: float = _field(0.75, group="cache", env="CACHE_SIMILARITY_THRESHOLD", aliases=("CACHE_THRESHOLD",), description="How similar a question must be to reuse a cached answer, from 0 to 1.")
    cache_ttl_hours: int = _field(24, group="cache", env="CACHE_TTL_HOURS", description="Hours a cached answer stays valid.")
    cache_strict_mode: bool = _field(True, group="cache", env="CACHE_STRICT_MODE", description="Only reuse an answer when the underlying expense rows are identical.")

    # Demo data
    banko_skip_autosetup: bool = _field(False, group="data", env="BANKO_SKIP_AUTOSETUP", description="Skip generating sample data on an empty database.")
    sample_data_count: int = _field(5000, group="data", env="SAMPLE_DATA_COUNT", description="Sample expenses generated at boot when the database is empty. `banko-ai run --generate-data N` overrides it.")
    data_gen_batch_size: int = _field(50, group="data", env="DATA_GEN_BATCH_SIZE", description="Rows per insert batch in the data generator.")
    demo_user_ttl_days: int = _field(0, group="data", env="DEMO_USER_TTL_DAYS", description="Delete signed-up demo users older than this many days at boot. 0 keeps them.")
    monthly_budget_default: float = _field(10000.0, group="data", env="MONTHLY_BUDGET_DEFAULT", description="Monthly budget the budget agent assumes when a user has none.")
    fraud_duplicate_window_days: int = _field(60, group="data", env="FRAUD_DUPLICATE_WINDOW_DAYS", description="Days the fraud agent looks back for duplicate charges.")
    categories_file: str = _field("", group="data", env="CATEGORIES_FILE",
        description="YAML file replacing the packaged category list. Empty uses banko_ai/config/defaults/categories.yaml.")
    personas_file: str = _field("", group="data", env="PERSONAS_FILE",
        description="YAML file replacing the packaged demo personas. Empty uses banko_ai/config/defaults/personas.yaml.")
    seed_profiles_file: str = _field("", group="data", env="SEED_PROFILES_FILE",
        description="YAML file replacing the packaged seed profiles. Empty uses banko_ai/config/defaults/seed_profiles.yaml.")
    seed_history_months: int = _field(6, group="data", env="SEED_HISTORY_MONTHS",
        description="Months of history seeded for each persona or new user, unless the profile says otherwise.")
    seed_batch_size: int = _field(200, group="data", env="SEED_BATCH_SIZE",
        description="Ledger entries written per seeding transaction.")
    seed_parallelism: int = _field(3, group="data", env="SEED_PARALLELISM",
        description="Users seeded at the same time.")
    checking_overdraft_limit: float = _field(0.0, group="data", env="CHECKING_OVERDRAFT_LIMIT",
        description="How far below zero a checking account may go. 0 means no overdraft.")
    max_transfer_amount: float = _field(1000000.0, group="data", env="MAX_TRANSFER_AMOUNT",
        description="Largest single purchase, deposit or transfer the API accepts.")
    integrity_interval_seconds: int = _field(300, group="data", env="INTEGRITY_INTERVAL_SECONDS",
        description="Seconds between money integrity checks. 0 turns the checker off.")
    integrity_max_accounts: int = _field(5000, group="data", env="INTEGRITY_MAX_ACCOUNTS",
        description="The integrity checker skips itself above this many accounts.")
    reporting_follower_reads: bool = _field(True, group="data", env="REPORTING_FOLLOWER_READS",
        description="Read reports (cash flow, net worth history, statements) with follower reads, a few seconds stale.")
    recurring_min_occurrences: int = _field(3, group="data", env="RECURRING_MIN_OCCURRENCES",
        description="Charges needed before a merchant counts as a recurring bill or subscription.")
    recurring_price_change_pct: float = _field(5.0, group="data", env="RECURRING_PRICE_CHANGE_PCT",
        description="Percent change between a subscription's old and new price that counts as a price change.")

    # Coach and CDC
    cdc_webhook_hmac_secret: str = _field("", group="coach", env="CDC_WEBHOOK_HMAC_SECRET", secret=True, description="Shared secret for the /api/cdc/signals webhook. Empty rejects every webhook.")
    coach_default_user_id: str = _field("00000000-0000-0000-0000-0000000000a1", group="coach", env="COACH_DEFAULT_USER_ID", description="User that API calls without a session act as.")
    coach_agent_max_steps: int = _field(5, group="coach", env="COACH_AGENT_MAX_STEPS", description="Most tool steps the coach planner may take.")
    coach_socketio_room_prefix: str = _field("coach:", group="coach", env="COACH_SOCKETIO_ROOM_PREFIX", description="Socket.IO room prefix for per-user nudges.")
    coach_kafka_enabled: bool = _field(False, group="coach", env="COACH_KAFKA_ENABLED", description="Consume spending signals from Kafka as well as the webhook.")
    kafka_bootstrap_servers: str = _field("localhost:9092", group="coach", env="KAFKA_BOOTSTRAP_SERVERS", description="Kafka brokers for the coach transport.")
    coach_kafka_topic: str = _field("banko.spending_signals", group="coach", env="COACH_KAFKA_TOPIC", description="Kafka topic carrying spending signals.")
    coach_tx_default_limit: int = _field(25, group="coach", env="COACH_TX_DEFAULT_LIMIT", description="Transactions a coach tool returns by default.")
    coach_agg_lookback_days: int = _field(30, group="coach", env="COACH_AGG_LOOKBACK_DAYS", description="Days coach aggregates look back by default.")
    coach_signal_ttl_days: int = _field(30, group="coach", env="COACH_SIGNAL_TTL_DAYS", description="Row-level TTL for spending_signals, in days.")
    coach_nudge_ttl_days: int = _field(90, group="coach", env="COACH_NUDGE_TTL_DAYS", description="Row-level TTL for coach_nudges, in days.")
    checkpoint_ttl_days: int = _field(7, group="coach", env="CHECKPOINT_TTL_DAYS", description="Row-level TTL for LangGraph checkpoints, in days. 0 keeps them.")

    @field_validator("ai_service", "ai_display_provider", mode="before")
    @classmethod
    def _normalize_provider(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        value = value.strip().lower()
        return "aws" if value == "bedrock" else value

    @field_validator("watsonx_api_url", mode="after")
    @classmethod
    def _complete_watsonx_url(cls, value: str) -> str:
        # WATSONX_URL used to mean the bare regional host; the chat
        # provider needs the full endpoint. Accept both.
        parts = urlsplit(value)
        if parts.path in ("", "/"):
            return f"{parts.scheme}://{parts.netloc}{_WATSONX_CHAT_PATH}"
        return value

    @field_validator("rls_mode", mode="after")
    @classmethod
    def _check_rls_mode(cls, value: str) -> str:
        allowed = {"auto", "off"}
        if value not in allowed:
            raise ValueError(f"RLS_MODE={value!r} must be one of {', '.join(sorted(allowed))}")
        return value

    @model_validator(mode="after")
    def _check_choices(self) -> Settings:
        if self.ai_service not in PROVIDERS:
            raise ValueError(f"AI_SERVICE={self.ai_service!r} is not one of {', '.join(PROVIDERS)}")
        if self.ai_display_provider and self.ai_display_provider not in PROVIDERS:
            raise ValueError(
                f"AI_DISPLAY_PROVIDER={self.ai_display_provider!r} is not one of {', '.join(PROVIDERS)} "
                "(leave it empty to show the real provider)")
        if (self.ai_display_model or self.ai_display_models) and not self.ai_display_provider:
            raise ValueError("AI_DISPLAY_MODEL and AI_DISPLAY_MODELS need AI_DISPLAY_PROVIDER set too")
        if self.ai_display_provider and not (self.ai_display_model or self.ai_display_models):
            # Without a model to show, the dropdown would fall back to the
            # real model names, which gives the persona away.
            raise ValueError("AI_DISPLAY_PROVIDER needs AI_DISPLAY_MODEL (or AI_DISPLAY_MODELS): the model name to show")
        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        sources: list[PydanticBaseSettingsSource] = [init_settings, env_settings]
        path = os.environ.get("BANKO_CONFIG")
        if path:
            sources.append(YamlConfigSettingsSource(settings_cls, yaml_file=path))
        return tuple(sources)

    @property
    def ollama_keep_alive_value(self) -> int | str:
        """OLLAMA_KEEP_ALIVE as Ollama wants it: bare digits are seconds."""
        value = self.ollama_keep_alive.strip()
        return int(value) if value.lstrip("-").isdigit() else value

    @property
    def watsonx_base_url(self) -> str:
        parts = urlsplit(self.watsonx_api_url)
        return f"{parts.scheme}://{parts.netloc}"

    def get_ai_config(self) -> dict[str, Any]:
        """Provider configuration in the shape the providers expect."""
        return {
            "service": self.ai_service,
            "openai": {"api_key": self.openai_api_key, "base_url": self.openai_base_url, "model": self.openai_model},
            "aws": {
                "access_key_id": self.aws_access_key_id,
                "secret_access_key": self.aws_secret_access_key,
                "profile_name": self.aws_profile,
                "region": self.aws_region,
                "model": self.aws_model,
            },
            "ollama": {"host": self.ollama_host, "model": self.ollama_model},
            "watsonx": {"api_key": self.watsonx_api_key, "project_id": self.watsonx_project_id, "model": self.watsonx_model},
            "gemini": {"project_id": self.google_project_id, "location": self.google_location, "model": self.google_model},
        }

    def validate(self) -> None:  # type: ignore[override]
        """Startup checks. Same behavior as before v2."""
        if not self.database_url:
            raise ValueError("DATABASE_URL is required")
        if not self.secret_key:
            running_under_gunicorn = "gunicorn" in os.environ.get("SERVER_SOFTWARE", "").lower()
            if self.flask_env.lower() == "production" or running_under_gunicorn:
                raise RuntimeError(
                    "SECRET_KEY must be set in production. Multi-worker deployments need a stable "
                    "secret so Flask session cookies validate across workers. Set SECRET_KEY "
                    "(for example `export SECRET_KEY=$(python -c 'import secrets; print(secrets.token_hex(32))')`).")
            self.secret_key = secrets.token_hex(32)
            print("Warning: SECRET_KEY not set. Generated a random key for this session.")
        if self.ai_service == "openai" and not self.openai_api_key:
            print("Warning: OPENAI_API_KEY not provided. AI features will be limited.")
        elif self.ai_service == "aws":
            if not (self.aws_access_key_id and self.aws_secret_access_key) and not self.aws_profile:
                raise ValueError("AWS requires either (AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY) or AWS_PROFILE")
        elif self.ai_service == "watsonx" and not self.watsonx_api_key:
            print("Warning: WATSONX_API_KEY not provided. AI features will be limited.")
        elif self.ai_service == "gemini" and not self.google_project_id:
            raise ValueError("GOOGLE_PROJECT_ID is required for Gemini service")

    def describe(self) -> list[SettingInfo]:
        """Every setting, its effective value (secrets masked) and its source."""
        env_present = {key.upper() for key in os.environ}
        file_keys = _yaml_keys()
        rows = []
        for name, field in type(self).model_fields.items():
            meta = field_meta(field)
            env = meta.env
            if any(n.upper() in env_present for n in (env, *meta.aliases)):
                source = "env"
            elif name in file_keys or env in file_keys:
                source = "file"
            else:
                source = "default"
            value = getattr(self, name)
            shown = _masked(name, value) if meta.secret else _shown(value)
            rows.append(SettingInfo(env=env, group=meta.group, value=shown,
                                    source=source, description=field.description or ""))
        return rows


def _shown(value: Any) -> str:
    return "" if value is None else str(value)


def _masked(name: str, value: Any) -> str:
    if not value:
        return "(not set)"
    if name == "database_url":
        from sqlalchemy.engine import make_url
        try:
            return make_url(str(value)).render_as_string(hide_password=True)
        except Exception:
            return "(set, unparseable)"
    return "(set)"


def _yaml_keys() -> set[str]:
    path = os.environ.get("BANKO_CONFIG")
    if not path:
        return set()
    try:
        with open(path) as handle:
            data = yaml.safe_load(handle) or {}
        return {str(key) for key in data}
    except OSError:
        return set()


def _warn_legacy_env() -> None:
    present = {key.upper() for key in os.environ}
    for old, new in DEPRECATED_ALIASES.items():
        if old in present and old not in _warned:
            _warned.add(old)
            log.warning("%s is deprecated; use %s (the old name still works for now)", old, new)
    for name, hint in REMOVED_SETTINGS.items():
        if name in present and name not in _warned:
            _warned.add(name)
            log.warning("%s is no longer used: %s", name, hint)


def render_config_docs() -> str:
    """Markdown reference for the README, generated from the field definitions."""
    lines: list[str] = []
    for group, title in GROUPS.items():
        fields = [f for f in Settings.model_fields.values() if field_meta(f).group == group]
        if not fields:
            continue
        lines += [f"### {title}", "", "| Variable | Default | What it does |", "|---|---|---|"]
        for field in fields:
            meta = field_meta(field)
            default = "" if field.default in (None, "") else f"`{field.default}`"
            description = field.description or ""
            aliases = meta.aliases
            if aliases:
                names = ", ".join(f"`{a}`" for a in aliases)
                description += f" The old name {names} still works." if len(aliases) == 1 else f" The old names {names} still work."
            lines.append(f"| `{meta.env}` | {default} | {description} |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


_config: Settings | None = None


def get_config() -> Settings:
    """The process-wide settings, loaded once."""
    global _config
    if _config is None:
        config = Settings()
        _warn_legacy_env()
        config.validate()
        _config = config
    return _config


def set_config(config: Settings) -> None:
    """Replace the process-wide settings (tests)."""
    global _config
    _config = config


def reset_config() -> None:
    """Forget the cached settings so the next get_config() rereads the environment (tests)."""
    global _config
    _config = None


# Older code imports Config.
Config = Settings
