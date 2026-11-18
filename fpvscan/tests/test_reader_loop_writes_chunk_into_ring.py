"""LOCK USB reader must append the 5 ms chunk onto the primed ring."""
from __future__ import annotations

import numpy as np

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


def test_reader_loop_writes_five_ms_chunk_into_existing_ring(monkeypatch):
    """#105 `_start_reader` writes the 10 ms settled dump into the
    ring. #108 `_reader_loop` USB chunk is 5 ms, but the first
    `read` set the stop Event and `_ring` was None — size only,
    no write. The live path is `ring.write(iq)` on `self._ring`
    after that `sync_rx`. A leftover skip-write leaves
    `filled==len(dump)` — shipped `capture_ms=80` then hits
    `len(iq) < n` and sleeps `idle_ms` for the whole glance. A
    leftover write-zeros (or a new buffer not assigned to
    `_ring`) throws away PLL-settled IQ so DecodeState `abs_t0`
    jumps and tracking never shrinks. A leftover dump-sized
    write still looks like a live hop (`retune_calls==1`) while
    USB stays 2× the 5 ms budget. Direct `_reader_loop` after
    `_start_reader`; occupancy/decode stay out. Freeze the IQ
    thread so a later USB chunk cannot hide a skip-write. File
    pacing (#102) is not pinned.
    """
    monkeypatch.setattr("fpvscan.engine.threading.Thread", _CapturingThread)
    fs = 20e6
    src = MockSource(fs=10e6)
    eng = make_engine(src, capture_ms=80, ring_seconds=0.5)
    try:
        eng._start_reader(5800e6, fs, 0.5)
        ring = eng._ring
        ten_ms = max(2048, int(fs * 0.01))
        five_ms = max(1024, int(fs * 0.005))
        assert ring is not None
        assert ring.filled == ten_ms
        ns: list[int] = []

        def _read(n):
            ns.append(int(n))
            eng._reader_stop.set()
            return np.full(int(n), 2 + 0j, dtype=np.complex64)

        src.read = _read
        eng._reader_loop(fs)
        assert ns == [five_ms]
        assert ns[0] != ten_ms
        assert ns[0] != int(fs * 0.080)
        assert eng._ring is ring
        assert ring.filled == ten_ms + five_ms
        snap, abs_start = ring.snapshot(ten_ms + five_ms)
        assert abs_start == 0
        assert len(snap) == ten_ms + five_ms
        assert np.all(snap[:ten_ms] == 0)
        assert np.all(snap[ten_ms:] == np.complex64(2 + 0j))
        assert src.retune_calls == 1
    finally:
        eng._stop_reader()
