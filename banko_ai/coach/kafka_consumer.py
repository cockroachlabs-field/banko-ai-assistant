"""SignalsKafkaConsumer — prod-mode signal transport.

Polls the `banko.spending_signals` topic (key = user_id, value = JSON
matching the same shape as the webhook envelope's `after` row). Normalizes
to `Signal` and hands to `SignalHandler` — identical contract to the
webhook receiver, so the Coach can't tell which transport produced the
event.

Failure modes (per spec §6.1):
  - poison message (unparseable / invalid Signal): publish to
    `<topic>.dlq` and commit (otherwise we'd block the partition forever).
  - handler raises: DO NOT commit — Kafka redelivers on next poll, with
    `SignalHandler`'s idempotency layer protecting against duplicates.
  - broker disconnect / startup failure: the run loop sleeps with
    exponential backoff and tries again.

Testability: the constructor takes `kafka_consumer_factory`, `commit_fn`,
and `dlq_send_fn` callables so tests can inject fakes (see
tests/coach/test_kafka_consumer.py).
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from .handler import SignalHandler
from .signals import Signal, SignalParseError

log = logging.getLogger("banko.coach.kafka")


@dataclass
class SignalsKafkaConsumer:
    handler: SignalHandler
    kafka_consumer_factory: Callable[[], Iterable[Any]]
    commit_fn: Callable[[Any], None]
    dlq_send_fn: Callable[[bytes, str], None] | None = None
    backoff_seconds: tuple[int, ...] = (1, 2, 5, 10, 30)

    def run_forever(self) -> None:
        """Outer loop with exponential backoff on broker failures. Inner
        loop is `run_once`."""
        attempt = 0
        while True:
            try:
                self.run_once()
                attempt = 0
            except Exception:
                wait = self.backoff_seconds[min(attempt,
                                                 len(self.backoff_seconds) - 1)]
                log.exception("kafka consumer crashed; backing off %ss", wait)
                time.sleep(wait)
                attempt += 1

    def run_once(self) -> None:
        """Drains the next batch from the broker (or fake) exactly once.
        Exposed for tests; production calls `run_forever`."""
        consumer = self.kafka_consumer_factory()
        for msg in consumer:
            self._process_one(msg)

    def _process_one(self, msg: Any) -> None:
        raw = msg.value
        try:
            data = json.loads(raw.decode("utf-8"))
            signal = Signal.from_dict(data)
        except (UnicodeDecodeError, json.JSONDecodeError, SignalParseError,
                KeyError) as e:
            log.error("poison message at offset=%s: %s",
                      getattr(msg, "offset", "?"), e)
            if self.dlq_send_fn is not None:
                self.dlq_send_fn(raw, str(e))
            self.commit_fn(msg)
            return

        try:
            self.handler.handle(signal)
        except Exception as e:
            # Constraint violations are permanent: the classic case is a
            # signal whose parent row was deleted between the changefeed
            # capturing it and us consuming it. Redelivering forever would
            # loop, re-running the LLM on every attempt. Ship it to the
            # DLQ and move on; genuinely transient failures (DB blips,
            # provider timeouts) still redeliver.
            from sqlalchemy.exc import IntegrityError
            if isinstance(e, IntegrityError):
                log.error("permanent failure at offset=%s, sending to DLQ: %s",
                          getattr(msg, "offset", "?"), e)
                if self.dlq_send_fn is not None:
                    self.dlq_send_fn(raw, str(e))
                self.commit_fn(msg)
                return
            log.exception("handler raised on offset=%s; will redeliver",
                          getattr(msg, "offset", "?"))
            return

        self.commit_fn(msg)


def build_production_consumer(handler: SignalHandler,
                               bootstrap_servers: str,
                               topic: str,
                               group_id: str = "banko-coach-v1"
                               ) -> SignalsKafkaConsumer:
    """Builds a SignalsKafkaConsumer wired to a real kafka-python broker
    consumer. Kept out of the dataclass so tests don't need a broker."""
    from kafka import KafkaConsumer, KafkaProducer
    kc = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        group_id=group_id,
        enable_auto_commit=False,
        auto_offset_reset="latest",
        value_deserializer=lambda b: b,
    )
    producer = KafkaProducer(bootstrap_servers=bootstrap_servers)
    dlq_topic = f"{topic}.dlq"

    def commit_fn(msg):
        from kafka import OffsetAndMetadata, TopicPartition
        tp = TopicPartition(msg.topic, msg.partition)
        # kafka-python 2.3 added a required leader_epoch field; -1 means
        # unknown, which the broker accepts for plain offset commits.
        kc.commit({tp: OffsetAndMetadata(msg.offset + 1, None, -1)})

    def dlq_send_fn(raw_value: bytes, error: str):
        producer.send(dlq_topic, raw_value,
                      headers=[("error", error.encode())])
        producer.flush()

    return SignalsKafkaConsumer(
        handler=handler,
        kafka_consumer_factory=lambda: kc,
        commit_fn=commit_fn,
        dlq_send_fn=dlq_send_fn,
    )
