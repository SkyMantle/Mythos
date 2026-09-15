"""Sweep must release the LOCK IQ reader before it retunes the same USB source."""
from __future__ import annotations

from tests.helpers import make_engine


class _FakeReader:
    def __init__(self):
        self.joined = False

    def join(self, timeout=None):
        self.joined = True


def test_sweep_command_stops_iq_reader_so_sweep_can_own_the_radio():
    """LOCK's reader thread and sweep retune share one bladeRF handle.

    If Sweep only flips mode and leaves `_reader_thread` running,
    `_do_sweep` `_grab()` races `sync_rx` — the same USB dual-access
    that drops LOCK after a short join. Auto-peek leftover must also
    clear, or the next confirmed hit would inherit a live reader.
    """
    eng = make_engine()
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng.state.auto = True
    reader = _FakeReader()
    eng._reader_thread = reader
    eng._ring = object()

    eng.command("sweep")
    eng._drain_commands()

    assert reader.joined, "Sweep must join the IQ reader before returning"
    assert eng._reader_thread is None
    assert eng._ring is None
    assert eng.state.mode == "SWEEP"
    assert eng.state.lock_target is None
    assert eng.state.auto is False
