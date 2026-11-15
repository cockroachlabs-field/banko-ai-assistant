"""Unit tests for SignalHandler. Uses an injected StubCoach so the handler
contract is tested without touching the LLM."""

import os

import pytest
from sqlalchemy import create_engine, text

from banko_ai.coach.handler import SignalHandler
from banko_ai.coach.signals import Signal, SignalType
from banko_ai.utils.migration import DatabaseMigration

TEST_USER = "00000000-0000-0000-0000-000000000eee"


@pytest.fixture(scope="module")
def db_url() -> str:
    # Scratch database on purpose: these tests INSERT and DELETE real
    # spending_signals rows, and on the shared database a changefeed
    # (cdc-demo) ships the inserts to Kafka. Consumers then meet events
    # whose parent row was deleted by test cleanup.
    url = os.getenv("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL not set")
    from sqlalchemy.engine.url import make_url
    scratch = str(make_url(url).set(database="banko_coach_test"))
    assert make_url(scratch).database == "banko_coach_test"
    admin = create_engine(url)
    with admin.connect() as conn:
        # Recreate from clean so a crashed prior run cannot leak rows
        # (fixed idempotency keys collide otherwise).
        conn.execute(text("DROP DATABASE IF EXISTS banko_coach_test CASCADE"))
        conn.execute(text("CREATE DATABASE banko_coach_test"))
        conn.commit()
    DatabaseMigration(database_url=scratch).migrate_to_coach_v1()
    yield scratch
    with admin.connect() as conn:
        conn.execute(text("DROP DATABASE IF EXISTS banko_coach_test CASCADE"))
        conn.commit()
    admin.dispose()


@pytest.fixture(autouse=True)
def _cleanup(db_url):
    eng = create_engine(db_url)
    for table in ("coach_nudges", "spending_signals"):
        try:
            with eng.begin() as conn:
                conn.execute(text(f"DELETE FROM {table} WHERE user_id = :u"),
                             {"u": TEST_USER})
        except Exception:
            pass
    eng.dispose()
    yield


class StubCoach:
    """Records calls; returns a canned nudge result."""
    def __init__(self):
        self.calls = []

    def react(self, signal: Signal) -> dict:
        self.calls.append(signal)
        return {
            "message": f"stub nudge for {signal.signal_type.value}",
            "tool_trace": [{"tool": "get_user_budget"}],
            "provider_used": "stub",
        }


class StubEmitter:
    def __init__(self):
        self.events = []

    def emit(self, event: str, payload: dict, room: str | None = None) -> None:
        self.events.append((event, payload, room))


def _make_signal(db_url: str, idem: str,
                 sig_type: SignalType = SignalType.BUDGET_THRESHOLD,
                 user_id: str = TEST_USER) -> Signal:
    # Inserts into the scratch database, never the shared one: the shared
    # spending_signals table is watched by the cdc-demo changefeed, and
    # test rows here used to leak into Kafka as phantom events.
    eng = create_engine(db_url)
    with eng.begin() as conn:
        row = conn.execute(text("""
            INSERT INTO spending_signals
              (user_id, signal_type, severity, payload, idempotency_key)
            VALUES (:u, :t, 'warn',
                    CAST(:payload AS JSONB), :k)
            RETURNING signal_id
        """), {"u": user_id, "t": sig_type.value, "k": idem,
               "payload": '{"category":"dining","pct_used":0.82}'}).fetchone()
    eng.dispose()
    return Signal(
        signal_id=str(row[0]),
        user_id=user_id,
        signal_type=sig_type,
        severity="warn",
        payload={"category": "dining", "pct_used": 0.82},
        idempotency_key=idem,
    )


def test_handler_invokes_coach_and_persists_nudge(db_url):
    coach = StubCoach()
    emitter = StubEmitter()
    handler = SignalHandler(coach=coach, emitter=emitter, database_url=db_url)

    sig = _make_signal(db_url, "h-1")
    result = handler.handle(sig)

    assert result["status"] == "delivered"
    assert len(coach.calls) == 1
    assert coach.calls[0].idempotency_key == "h-1"

    eng = create_engine(db_url)
    with eng.connect() as conn:
        nudges = conn.execute(text(
            "SELECT message, provider_used FROM coach_nudges WHERE user_id = :u"
        ), {"u": TEST_USER}).fetchall()
        consumed = conn.execute(text(
            "SELECT consumed_at FROM spending_signals WHERE signal_id = :s"
        ), {"s": sig.signal_id}).fetchone()
    eng.dispose()

    assert len(nudges) == 1
    assert nudges[0][1] == "stub"
    assert consumed[0] is not None
    assert len(emitter.events) == 1
    assert emitter.events[0][0] == "coach.nudge"


def test_handler_dedups_on_idempotency_key(db_url):
    coach = StubCoach()
    emitter = StubEmitter()
    handler = SignalHandler(coach=coach, emitter=emitter, database_url=db_url)

    sig = _make_signal(db_url, "h-dup")
    first = handler.handle(sig)
    second = handler.handle(sig)

    assert first["status"] == "delivered"
    assert second["status"] == "replayed"
    assert len(coach.calls) == 1


def test_handler_skips_suppressed_signal_type(db_url):
    coach = StubCoach()
    emitter = StubEmitter()
    handler = SignalHandler(
        coach=coach, emitter=emitter, database_url=db_url,
        suppressed_types={SignalType.RECURRING_DRIFT},
    )

    sig = _make_signal(db_url, "h-suppress", sig_type=SignalType.RECURRING_DRIFT)
    result = handler.handle(sig)

    assert result["status"] == "suppressed"
    assert coach.calls == []
    assert emitter.events == []


def test_a_load_customer_gets_no_model_written_nudge(db_url):
    """The load simulator's customers are named load-<n>-<hex> and nobody
    reads their nudges. With the pipeline attached to a load run every one of
    their signals would cost two model calls."""
    load_user = "00000000-0000-0000-0000-00000010ad01"
    eng = create_engine(db_url)
    with eng.begin() as conn:
        conn.execute(text("UPSERT INTO users (user_id, username, spending_style, demo_user) VALUES "
                          "(:u, 'load-00001-ab12', 'diner', true), (:p, 'h-person', 'diner', true)"),
                     {"u": load_user, "p": TEST_USER})
    try:
        coach, emitter = StubCoach(), StubEmitter()
        handler = SignalHandler(coach=coach, emitter=emitter, database_url=db_url)
        signal = _make_signal(db_url, "h-load-user", user_id=load_user)
        assert handler.handle(signal)["status"] == "suppressed"
        assert coach.calls == [] and emitter.events == []
        # Left unclaimed. A claimed signal with no nudge is what the boot
        # replay looks for, and every restart would go through each of a load
        # run's signals again.
        from banko_ai.coach.handler import replay_stranded_signals
        assert replay_stranded_signals(handler, db_url) == 0 and coach.calls == []
        with eng.connect() as conn:
            assert conn.execute(text("SELECT consumed_at FROM spending_signals WHERE signal_id = :s"),
                                {"s": signal.signal_id}).scalar() is None
        # A person's signal still reaches the coach.
        assert handler.handle(_make_signal(db_url, "h-person"))["status"] == "delivered"
        assert len(coach.calls) == 1
    finally:
        with eng.begin() as conn:
            conn.execute(text("DELETE FROM coach_nudges WHERE user_id IN (:u, :p)"), {"u": load_user, "p": TEST_USER})
            conn.execute(text("DELETE FROM spending_signals WHERE user_id IN (:u, :p)"), {"u": load_user, "p": TEST_USER})
            conn.execute(text("DELETE FROM users WHERE user_id IN (:u, :p)"), {"u": load_user, "p": TEST_USER})
        eng.dispose()


def test_handler_skips_stale_event_without_llm_call(db_url):
    # A Kafka event whose source row was deleted (test cleanup, TTL,
    # clear-demo-users) must be skipped before the LLM runs; it used to
    # invoke the model and then fail the nudge insert's foreign key.
    coach = StubCoach()
    emitter = StubEmitter()
    handler = SignalHandler(coach=coach, emitter=emitter, database_url=db_url)

    sig = _make_signal(db_url, "h-stale")
    eng = create_engine(db_url)
    with eng.begin() as conn:
        conn.execute(text("DELETE FROM spending_signals WHERE signal_id = :s"),
                     {"s": sig.signal_id})
    eng.dispose()

    result = handler.handle(sig)

    assert result["status"] == "stale"
    assert coach.calls == []
    assert emitter.events == []


def test_concurrent_delivery_processes_exactly_once(db_url):
    # The same signal arrives on both transports within a second (the
    # webhook claim INSERT flows through the changefeed to Kafka). The
    # claim is a compare-and-set on consumed_at, so the second delivery
    # must skip even though the first has not finished its LLM call:
    # simulate mid-flight by claiming, then delivering again.
    coach = StubCoach()
    emitter = StubEmitter()
    handler = SignalHandler(coach=coach, emitter=emitter, database_url=db_url)

    sig = _make_signal(db_url, "h-race")
    assert handler._try_claim(sig) is True       # transport A mid-processing
    second = handler.handle(sig)                 # transport B arrives now

    assert second["status"] == "replayed"
    assert coach.calls == []


def test_failed_persist_releases_the_claim(db_url):
    # A transient failure after the claim must release it, or redelivery
    # would skip the signal forever and the nudge would be lost.
    coach = StubCoach()
    emitter = StubEmitter()
    handler = SignalHandler(coach=coach, emitter=emitter, database_url=db_url)

    sig = _make_signal(db_url, "h-unclaim")
    original = handler._persist_nudge
    handler._persist_nudge = lambda s, n: (_ for _ in ()).throw(RuntimeError("db blip"))
    with pytest.raises(RuntimeError):
        handler.handle(sig)
    handler._persist_nudge = original

    retry = handler.handle(sig)
    assert retry["status"] == "delivered"


def test_evidence_always_includes_the_source_signal(db_url):
    # The panel showed a bare [] whenever the planner ran no tools; the
    # signal payload is the grounding and must lead the evidence.
    coach = StubCoach()
    emitter = StubEmitter()
    handler = SignalHandler(coach=coach, emitter=emitter, database_url=db_url)

    sig = _make_signal(db_url, "h-evidence")
    handler.handle(sig)

    eng = create_engine(db_url)
    with eng.connect() as conn:
        trace = conn.execute(text(
            "SELECT tool_trace FROM coach_nudges WHERE signal_id = :s"
        ), {"s": sig.signal_id}).scalar()
    eng.dispose()

    assert trace[0]["tool"] == "source_signal"
    assert trace[0]["payload"]["pct_used"] == 0.82
    assert trace[1]["tool"] == "get_user_budget"  # stub's own trace kept


def test_replay_recovers_claimed_signal_with_no_nudge(db_url):
    # A process dying between claim and persist strands the signal:
    # consumed_at set, no nudge row. Boot-time replay must release the
    # claim and produce the nudge.
    from banko_ai.coach.handler import replay_stranded_signals
    coach = StubCoach()
    emitter = StubEmitter()
    handler = SignalHandler(coach=coach, emitter=emitter, database_url=db_url)

    sig = _make_signal(db_url, "h-stranded")
    assert handler._try_claim(sig) is True  # claimed, then "process died"

    replayed = replay_stranded_signals(handler, db_url)

    assert replayed == 1
    assert len(coach.calls) == 1
    eng = create_engine(db_url)
    with eng.connect() as conn:
        count = conn.execute(text(
            "SELECT count(*) FROM coach_nudges WHERE signal_id = :s"
        ), {"s": sig.signal_id}).scalar()
    eng.dispose()
    assert count == 1
