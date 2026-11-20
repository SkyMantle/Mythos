"""LOCK USB reader must exit cleanly when stop dropped the ring."""
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


def test_reader_loop_ring_none_exits_without_setting_err(monkeypatch):
    """#108 chunk *size* with `_ring is None` so no write (boot,
    never `_start_reader`). #110 writes a successful chunk onto
    the dump. #112 stashes a USB *exception* into `_reader_err`.
    The missing half: `_stop_reader` after a join-timed-out
    `sync_rx` nulls `_ring` while the orphan is still inside
    `src.read`. When that read returns, `_reader_loop` is
    `ring = self._ring; if ring is None: return` — no write,
    no `_reader_err`. A leftover `self._reader_err = e` /
    `RuntimeError("ring gone")` still looks like a live hop
    (`retune_calls==1`) while the next `_do_lock` re-raises
    after Sweep / AFC recenter — LOCK looks dead with a primed
    dump. A leftover write onto a newly minted ring races the
    next hop's IQ — DecodeState `abs_t0` jumps, tracking never
    shrinks. Direct `_reader_loop` after `_start_reader`; freeze
    the IQ thread. Did not copy #10 (short-join race) or #84
    (`_do_lock` still consults leftover err *before* start).
    File pacing (#102) is not pinned.
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
        eng._ring = None
        ns: list[int] = []
        orig = src.read

        def _read(n):
            ns.append(int(n))
            eng._reader_stop.set()
            return orig(n)

        src.read = _read
        eng._reader_loop(fs)
        assert ns == [five_ms]
        assert eng._reader_err is None
        assert eng._ring is None
        assert dump.filled == ten_ms
        assert src.retune_calls == 1
    finally:
        eng._stop_reader()
