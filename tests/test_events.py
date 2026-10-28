"""The event bus behind the Platform page's events card and chart markers."""

import pytest

from banko_ai.observability.events import EventBus


def test_events_since_and_capacity():
    bus = EventBus(capacity=3, clock=lambda: 42.0)
    ids = [bus.publish(f"e{i}")["id"] for i in range(5)]
    assert [e["text"] for e in bus.since(None)] == ["e2", "e3", "e4"]
    assert [e["text"] for e in bus.since(ids[3])] == ["e4"]
    assert bus.since(ids[-1]) == []


def test_an_event_carries_level_source_and_time():
    event = EventBus(clock=lambda: 7.5).publish("Region us-east-1 is down", level="bad", source="chaos")
    assert event == {"id": event["id"], "at": 7.5, "level": "bad", "source": "chaos",
                     "text": "Region us-east-1 is down"}


def test_unknown_levels_are_rejected():
    with pytest.raises(ValueError):
        EventBus().publish("x", level="loud")


def test_since_limits_to_the_newest():
    bus = EventBus()
    for i in range(10):
        bus.publish(f"e{i}")
    assert [e["text"] for e in bus.since(None, limit=2)] == ["e8", "e9"]


def test_each_bus_has_its_own_epoch():
    """Event ids start at 1 in every process; the epoch tells a page that
    the app restarted, so it stops waiting for ids it already passed."""
    assert EventBus().epoch != EventBus().epoch
