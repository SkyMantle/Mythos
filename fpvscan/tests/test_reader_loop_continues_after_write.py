"""LOCK USB reader must keep calling src.read after a successful chunk write."""
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


def test_reader_loop_reads_again_after_successful_write(monkeypatch):
    """#108 / #110 / #112 enter `src.read` once because the
    first `read` `.set()`s the stop Event (size / write /
    USB-exception). #113 already-set Event skips `read`
    entirely. The missing half: a successful `ring.write`
    must *not* `return` — `_reader_loop` is `while not
    is_set()` and the next iteration calls `src.read`
    again. A leftover `return` after write still looks
    like a live hop (`retune_calls==1`, filled==dump+chunk)
    while shipped `capture_ms=80` then hits `len(iq) < n`
    and sleeps `idle_ms` for the whole glance. Tracking
    never shrinks; DecodeState `abs_t0` is stuck on 15 ms
    of IQ. Direct `_reader_loop` after `_start_reader`;
    occupancy/decode stay out. Freeze the IQ thread. File
    pacing (#102) is not pinned.
    """
    monkeypatch.setattr("fpvscan.engine.threading.Thread", _CapturingThread)
    fs = 20e6
    src = MockSource(fs=10e6)
    eng = make_engine(src, capture_ms=80, ring_seconds=0.5)
    try:
        eng._start_reader(5800e6, fs, 0.5)
        dump = eng._ring
        ten_ms = max(2048, int(fs * 0.01))
        five_ms = max(1024, int(fs * 0.005))
        assert dump is not None
        assert dump.filled == ten_ms
        assert eng._reader_err is None
        ns: list[int] = []
        orig = src.read

        def _read(n):
            ns.append(int(n))
            if len(ns) >= 2:
                eng._reader_stop.set()
            return orig(n)

        src.read = _read
        eng._reader_loop(fs)
        assert ns == [five_ms, five_ms]
        assert ns[0] != ten_ms
        assert ns[0] != int(fs * 0.080)
        assert eng._reader_err is None
        assert eng._ring is dump
        assert dump.filled == ten_ms + 2 * five_ms
        assert src.retune_calls == 1
        assert eng._reader_stop.is_set()
    finally:
        eng._stop_reader()
