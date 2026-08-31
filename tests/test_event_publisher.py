"""Unit tests for Worker EventPublisher abstraction and transports."""
from datetime import datetime, timezone
import unittest
from apps.worker.app.events.publisher import (
    EventPublisher,
    LoggingEventPublisher,
    InMemoryEventPublisher,
    HTTPEventPublisher,
    get_event_publisher,
)


class TestEventPublisher(unittest.TestCase):
    """Test suite verifying EventPublisher interface and implementations."""

    def test_in_memory_event_publisher(self):
        """Verify InMemoryEventPublisher captures and stores published events."""
        publisher = InMemoryEventPublisher()
        self.assertIsInstance(publisher, EventPublisher)

        res = publisher.publish(
            stream_id="stream_gate_1",
            event_type="line_crossing_in",
            track_id=10,
            confidence=0.95,
            position=[320, 240],
            metadata={"source": "unit_test"},
        )
        self.assertTrue(res)
        self.assertEqual(len(publisher.published_events), 1)

        event = publisher.published_events[0]
        self.assertEqual(event["stream_id"], "stream_gate_1")
        self.assertEqual(event["event_type"], "line_crossing_in")
        self.assertEqual(event["track_id"], 10)
        self.assertEqual(event["position"], [320, 240])

        publisher.clear()
        self.assertEqual(len(publisher.published_events), 0)

    def test_logging_event_publisher(self):
        """Verify LoggingEventPublisher completes without raising exceptions."""
        publisher = LoggingEventPublisher()
        self.assertIsInstance(publisher, EventPublisher)

        res = publisher.publish(
            stream_id="stream_gate_2",
            event_type="line_crossing_out",
            track_id=42,
            confidence=0.88,
        )
        self.assertTrue(res)

    def test_http_event_publisher_graceful_offline(self):
        """Verify HTTPEventPublisher handles unavailable endpoints gracefully without crashing."""
        publisher = HTTPEventPublisher(api_url="http://127.0.0.1:59999/api/v1/events", timeout=0.5)
        self.assertIsInstance(publisher, EventPublisher)

        # Should return False on connection failure rather than raising an unhandled exception
        res = publisher.publish(
            stream_id="stream_gate_3",
            event_type="line_crossing_in",
            track_id=7,
        )
        self.assertFalse(res)

    def test_publisher_factory(self):
        """Verify get_event_publisher instantiates appropriate publisher types."""
        p_mem = get_event_publisher("memory")
        self.assertIsInstance(p_mem, InMemoryEventPublisher)

        p_log = get_event_publisher("log")
        self.assertIsInstance(p_log, LoggingEventPublisher)

        p_http = get_event_publisher("http", api_url="http://localhost:8000/api/v1/events")
        self.assertIsInstance(p_http, HTTPEventPublisher)


if __name__ == "__main__":
    unittest.main()
