"""LOCK start-reader dump must floor at 2048 samples, not the 5 ms chunk floor."""
from __future__ import annotations

from tests.helpers import MockSource, make_engine


class _CapturingThread:
    """Records constructor kwargs; start() must not run USB."""

    def __init__(self, target=None, args=(), kwargs=None, daemon=None):
        self.target = target
        self.args = args
        self.kwargs = kwargs or {}
        self.daemon = daemon
        self.started = 0

    def start(self):
        self.started += 1

    def join(self, timeout=None):
        pass


def test_start_reader_dump_floors_at_2048_not_chunk_or_ten_ms(monkeypatch):
    """#103 dump at 20 Msps is 10 ms (`max(2048, int(fs*0.01))`
    = 200_000) so the 2048 floor never fires. #108 USB *chunk*
    floors at 1024 at 100 ksps (5 ms would be 500). The missing
    half: the PLL dump is a different floor — 2048 — so a slow
    replay / neighbour clock still settles more than one USB
    packet. `_start_reader` is `retune_and_read(want,
    max(2048, int(fs * 0.01)))`. A leftover 10 ms without the
    floor (1000 at 100 ksps) is shorter than one `sync_rx`
    packet and shorter than the 1024 chunk floor: the first
    snapshot is 1 ms of unsettled IQ, tracking never shrinks,
    every field grabs full `capture_ms`. A leftover copy-paste
    of the chunk floor (1024) is still half a dump. A leftover
    `self.src.sample_rate` clock at 10 Msps asks 100_000 while
    `fs` is the LOCK rate. Direct `_start_reader`; occupancy/
    decode stay out. Freeze the IQ thread. File pacing (#102)
    is not pinned.
    """
    monkeypatch.setattr("fpvscan.engine.threading.Thread", _CapturingThread)
    src = MockSource(fs=10e6)
    ns: list[int] = []
    orig = src.retune_and_read

    def _rar(hz, n):
        ns.append(int(n))
        return orig(hz, n)

    src.retune_and_read = _rar
    eng = make_engine(src, capture_ms=80, ring_seconds=0.5)
    try:
        slow = 100e3
        eng._start_reader(5800e6, slow, 0.5)
        dump_n = max(2048, int(slow * 0.01))
        ten_ms = int(slow * 0.01)
        chunk_floor = max(1024, int(slow * 0.005))
        assert ns == [dump_n]
        assert ns[0] == 2048
        assert ns[0] != ten_ms
        assert ns[0] != 1000
        assert ns[0] != chunk_floor
        assert ns[0] != 1024
        assert ns[0] != int(src.sample_rate * 0.01)
        assert ns[0] != max(2048, int(src.sample_rate * 0.01))
        dump = eng._ring
        assert dump is not None
        assert dump.filled == 2048
        assert src.retune_calls == 1
        assert src._fc == 5800e6
    finally:
        eng._stop_reader()
