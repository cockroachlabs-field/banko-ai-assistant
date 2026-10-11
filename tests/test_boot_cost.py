"""Boot must not pay for work it does not need.

Two costs that landed on every `banko-ai run` against the 9-node chaos
cluster: SET LOCALITY re-ran on tables that already had it (CockroachDB
runs a full schema-change job either way, about 9s each, so 35s of a 65s
boot went to changing nothing), and the embedding model landed on
Apple's MPS device, which reserves about 1 GB of Metal memory for a
22M-parameter model. Neither test needs a database.
"""

import sentence_transformers

from banko_ai.utils import db_retry, migration
from banko_ai.utils.embeddings import load_embedding_model

DB_URL = "cockroachdb://root@localhost:26257/defaultdb?sslmode=disable"
REGIONS = ["eu-de", "us-east", "us-south"]


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return self._rows


class _Conn:
    """Answers the topology queries migrate_regional_tables asks and
    records every statement it runs."""

    def __init__(self, localities, executed):
        self._localities = localities
        self._executed = executed

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, stmt, params=None):
        sql = str(stmt)
        self._executed.append(sql)
        if '[SHOW REGIONS FROM DATABASE] WHERE "primary"' in sql:
            return _Result([("us-south",)])
        if "[SHOW REGIONS FROM DATABASE]" in sql:
            return _Result([(r,) for r in REGIONS])
        if "[SHOW TABLES]" in sql:
            return _Result(list(self._localities.items()))
        return _Result([])

    def commit(self):
        pass


class _Engine:
    def __init__(self, localities, executed):
        self._conn = _Conn(localities, executed)

    def connect(self):
        return self._conn

    def dispose(self):
        pass


def _run_migration(monkeypatch, localities):
    executed: list[str] = []
    monkeypatch.setattr(migration, "detect_regions", lambda url: REGIONS)
    monkeypatch.setattr(db_retry, "create_resilient_engine",
                        lambda url, **kw: _Engine(localities, executed))
    assert migration.migrate_regional_tables(DB_URL, "us-south") is True
    return [s for s in executed if s.lstrip().upper().startswith("ALTER")]


def test_regional_migration_leaves_converted_tables_alone(monkeypatch):
    alters = _run_migration(monkeypatch, {
        "expenses": "REGIONAL BY ROW",
        "spending_signals": "REGIONAL BY ROW",
        "coach_nudges": "REGIONAL BY ROW",
        "users": "GLOBAL",
    })
    assert alters == []


def test_regional_migration_still_converts_what_differs(monkeypatch):
    alters = _run_migration(monkeypatch, {
        "expenses": "REGIONAL BY TABLE IN PRIMARY REGION",
        "spending_signals": "REGIONAL BY ROW",
        "coach_nudges": "REGIONAL BY ROW",
        "users": "REGIONAL BY TABLE IN PRIMARY REGION",
    })
    assert alters == [
        "ALTER TABLE expenses SET LOCALITY REGIONAL BY ROW",
        "ALTER TABLE users SET LOCALITY GLOBAL",
    ]


def test_embedding_model_loads_on_cpu(monkeypatch):
    calls = []

    def fake_model(*args, **kwargs):
        calls.append(kwargs)
        if kwargs.get("local_files_only"):
            raise OSError("not in the local cache")
        return object()

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", fake_model)
    monkeypatch.delenv("EMBEDDING_DEVICE", raising=False)
    load_embedding_model()
    # Both the cached load and the cold-cache download must pin CPU.
    assert [c.get("device") for c in calls] == ["cpu", "cpu"]
