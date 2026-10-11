"""SignalHandler — the transport-agnostic boundary between webhook/Kafka
adapters and the Coach.

Responsibilities:
  - idempotency dedup (per `idempotency_key`)
  - user-pref suppression (opted-out signal types)
  - invoke `coach.react(signal)`
  - persist the nudge to `coach_nudges`
  - mark `spending_signals.consumed_at`
  - emit `coach.nudge` over the supplied emitter (SocketIO in prod;
    StubEmitter in tests)

The handler is intentionally synchronous. The webhook route hands off via
a thread or queue so the HTTP ack returns fast; the handler itself is
straight-line so its failure modes are obvious.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

from sqlalchemy import text

from .signals import Signal, SignalType

log = logging.getLogger("banko.coach.handler")


class CoachProtocol(Protocol):
    def react(self, signal: Signal) -> dict[str, Any]: ...


class EmitterProtocol(Protocol):
    def emit(self, event: str, payload: dict[str, Any],
             room: str | None = None) -> None: ...


@dataclass
class SignalHandler:
    coach: CoachProtocol
    emitter: EmitterProtocol
    database_url: str
    suppressed_types: Iterable[SignalType] = field(default_factory=tuple)
    socketio_room_prefix: str = "coach:"

    def handle(self, signal: Signal) -> dict[str, Any]:
        """Process one signal. Returns a status dict; never raises for
        expected outcomes (dedup, suppression, coach failure). Unexpected
        DB errors bubble up to the transport adapter, which decides
        whether to retry."""
        log.info("handling signal", extra={"signal_id": signal.signal_id,
                                            "type": signal.signal_type.value,
                                            "user_id": signal.user_id})

        # Atomic claim: set consumed_at up front with a compare-and-set.
        # Both transports share this handler, and the same signal arrives
        # on both within a second (the webhook claim INSERTs the row, the
        # changefeed ships that insert to Kafka). Marking consumed only
        # after the LLM finished left a window the width of a model call
        # in which the other transport double-processed the signal.
        if not self._try_claim(signal):
            if self._signal_row_exists(signal):
                log.info("signal already claimed or consumed, skipping",
                         extra={"signal_id": signal.signal_id})
                return {"status": "replayed", "signal_id": signal.signal_id}
            # A Kafka event can outlive its source row (test cleanup,
            # TTL, clear-demo-users): nothing to claim, nothing to nudge.
            log.info("signal row no longer exists, skipping stale event",
                     extra={"signal_id": signal.signal_id})
            return {"status": "stale", "signal_id": signal.signal_id}

        if signal.signal_type in set(self.suppressed_types):
            return {"status": "suppressed", "signal_id": signal.signal_id}

        try:
            try:
                nudge = self.coach.react(signal)
            except Exception as e:
                log.exception("coach failed", extra={"signal_id": signal.signal_id})
                nudge = self._fallback_nudge(signal, error=str(e))
                nudge["provider_used"] = "fallback"

            # The signal payload is the nudge's grounding; lead the
            # evidence with it so the panel is never blank even when the
            # planner ran no tools (small local models skip them often).
            nudge["tool_trace"] = ([{
                "tool": "source_signal",
                "signal_type": signal.signal_type.value,
                "severity": signal.severity,
                "payload": signal.payload,
            }] + list(nudge.get("tool_trace") or []))

            nudge_id = self._persist_nudge(signal, nudge)
        except Exception:
            # Release the claim so a redelivery can retry; without this a
            # transient DB failure would strand the signal as consumed.
            self._unclaim(signal)
            raise
        self.emitter.emit(
            "coach.nudge",
            payload={
                "nudge_id": nudge_id,
                "signal_id": signal.signal_id,
                "user_id": signal.user_id,
                "signal_type": signal.signal_type.value,
                "severity": signal.severity,
                "message": nudge["message"],
                "provider_used": nudge.get("provider_used"),
            },
            room=f"{self.socketio_room_prefix}{signal.user_id}",
        )
        return {"status": "delivered", "signal_id": signal.signal_id,
                "nudge_id": nudge_id}

    def _engine(self):
        from ..db.engine import get_engine
        return get_engine(self.database_url)

    def _try_claim(self, signal: Signal) -> bool:
        """Compare-and-set on consumed_at: exactly one caller wins the
        signal no matter how many transports or processes deliver it."""
        eng = self._engine()
        with eng.begin() as conn:
            result = conn.execute(text(
                "UPDATE spending_signals SET consumed_at = now() "
                "WHERE idempotency_key = :k AND consumed_at IS NULL"
            ), {"k": signal.idempotency_key})
        return result.rowcount == 1

    def _unclaim(self, signal: Signal) -> None:
        eng = self._engine()
        with eng.begin() as conn:
            conn.execute(text(
                "UPDATE spending_signals SET consumed_at = NULL "
                "WHERE idempotency_key = :k"
            ), {"k": signal.idempotency_key})

    def _signal_row_exists(self, signal: Signal) -> bool:
        eng = self._engine()
        with eng.connect() as conn:
            row = conn.execute(text(
                "SELECT 1 FROM spending_signals WHERE signal_id = :s"
            ), {"s": signal.signal_id}).fetchone()
        return row is not None

    def _persist_nudge(self, signal: Signal, nudge: dict[str, Any]) -> str:
        eng = self._engine()
        with eng.begin() as conn:
            from ..utils.migration import regional_tables_ready
            from ..web.auth import resolve_user_region

            cols = ["signal_id", "user_id", "message", "tool_trace",
                    "provider_used", "trace_id"]
            placeholders = [":sig", ":u", ":msg", "CAST(:trace AS JSONB)",
                           ":prov", ":trace_id"]
            params = {
                "sig": signal.signal_id,
                "u": signal.user_id,
                "msg": nudge["message"],
                "trace": json.dumps(nudge.get("tool_trace") or []),
                "prov": nudge.get("provider_used"),
                "trace_id": nudge.get("trace_id"),
            }

            if regional_tables_ready(self.database_url):
                user_region = resolve_user_region(signal.user_id, self.database_url)
                if user_region:
                    cols.append("crdb_region")
                    placeholders.append(":region")
                    params["region"] = user_region

            sql = f"""
                INSERT INTO coach_nudges ({", ".join(cols)})
                VALUES ({", ".join(placeholders)})
                RETURNING nudge_id
            """
            row = conn.execute(text(sql), params).fetchone()
        return str(row[0])

    def _fallback_nudge(self, signal: Signal, error: str) -> dict[str, Any]:
        """Templated fallback when the LLM provider is unavailable. Keeps
        the user-facing channel alive while ops investigates."""
        if signal.signal_type == SignalType.BUDGET_THRESHOLD:
            pct = int(signal.payload.get("pct_used", 0) * 100)
            cat = signal.payload.get("category", "this category")
            msg = (f"Heads up: you're at {pct}% of your {cat} budget. "
                   "(Coach AI is temporarily offline; this is a templated nudge.)")
        elif signal.signal_type == SignalType.ANOMALY:
            merchant = signal.payload.get("merchant", "a merchant")
            amount = signal.payload.get("amount", 0)
            msg = (f"Unusual charge detected: ${amount:.2f} at {merchant}. "
                   "Review and confirm. (Coach AI is offline.)")
        else:
            sub = signal.payload.get("subscription", "a subscription")
            msg = (f"A recurring charge changed: {sub}. Review the new amount. "
                   "(Coach AI is offline.)")
        return {"message": msg, "tool_trace": [{"fallback": True,
                                                 "error": error}]}


def replay_stranded_signals(handler: SignalHandler, database_url: str,
                            window_minutes: int = 60) -> int:
    """Recover signals that were claimed but never produced a nudge.

    The claim is taken before the model call, so a process dying mid
    nudge (restart, crash, Ctrl+C on a demo box) leaves the signal
    consumed with nothing to show for it. On boot, release those claims
    and run them through the handler again. Suppressed types re-suppress
    instantly, so replaying them is free.
    """
    from sqlalchemy import text
    
    from ..db.engine import get_engine
    eng = get_engine(handler.database_url)
    with eng.connect() as conn:
        rows = conn.execute(text("""
            SELECT s.signal_id, s.user_id, s.signal_type, s.severity,
                   s.payload, s.idempotency_key
            FROM spending_signals s
            LEFT JOIN coach_nudges n ON n.signal_id = s.signal_id
            WHERE s.consumed_at IS NOT NULL
              AND n.nudge_id IS NULL
              AND s.produced_at > now() - (:mins * INTERVAL '1 minute')
        """), {"mins": window_minutes}).fetchall()

    replayed = 0
    for row in rows:
        try:
            signal = Signal.from_dict({
                "signal_id": str(row[0]),
                "user_id": str(row[1]),
                "signal_type": row[2],
                "severity": row[3],
                "payload": row[4] if isinstance(row[4], dict) else {},
                "idempotency_key": row[5],
            })
            handler._unclaim(signal)
            result = handler.handle(signal)
            log.info("replayed stranded signal %s -> %s",
                     signal.signal_id, result.get("status"))
            replayed += 1
        except Exception:
            log.exception("failed to replay stranded signal %s", row[0])
    return replayed
