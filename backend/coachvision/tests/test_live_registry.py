"""Unit tests for the live-session observer registry."""

import asyncio
import unittest

from coachvision.realtime.connection_manager import LiveSessionRegistry


class FakeSocket:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.sent: list[dict] = []

    async def send_json(self, payload: dict) -> None:
        if self.fail:
            raise RuntimeError("socket closed")
        self.sent.append(payload)


class LiveRegistryTest(unittest.TestCase):
    def test_session_lifecycle_and_user_lookup(self) -> None:
        registry = LiveSessionRegistry()
        registry.register_session("s1", "user-a", "squat")
        registry.register_session("s2", "user-b", "plank")

        self.assertIsNotNone(registry.get_session("s1"))
        live = registry.sessions_for_users({"user-a"})
        self.assertEqual([info.session_id for info in live], ["s1"])

        registry.unregister_session("s1")
        self.assertIsNone(registry.get_session("s1"))
        self.assertEqual(registry.sessions_for_users({"user-a"}), [])

    def test_broadcast_reaches_observers_and_drops_dead_sockets(self) -> None:
        registry = LiveSessionRegistry()
        registry.register_session("s1", "user-a", "squat")

        healthy = FakeSocket()
        dead = FakeSocket(fail=True)
        registry.add_observer("s1", healthy)
        registry.add_observer("s1", dead)
        self.assertEqual(registry.observer_count("s1"), 2)

        asyncio.run(registry.broadcast("s1", {"type": "metrics", "count": 3}))

        self.assertEqual(healthy.sent, [{"type": "metrics", "count": 3}])
        # The failing socket must have been evicted.
        self.assertEqual(registry.observer_count("s1"), 1)

    def test_broadcast_without_observers_is_a_noop(self) -> None:
        registry = LiveSessionRegistry()
        registry.register_session("s1", "user-a", "squat")
        asyncio.run(registry.broadcast("s1", {"type": "metrics"}))  # must not raise

    def test_unregister_clears_observers(self) -> None:
        registry = LiveSessionRegistry()
        registry.register_session("s1", "user-a", "squat")
        registry.add_observer("s1", FakeSocket())
        registry.unregister_session("s1")
        self.assertEqual(registry.observer_count("s1"), 0)


if __name__ == "__main__":
    unittest.main()
