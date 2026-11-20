"""LOCK USB reader must not call src.read after the stop Event is set."""
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


def test_reader_loop_set_event_skips_usb_read(monkeypatch):
    """#72 / #110 `_start_reader` mints a *new* clear Event after
    Sweep / peek-expiry `_stop_reader` `.set()`s the leftover.
    #108 / #112 enter `src.read` because the Event is still
    clear. The missing half: `_reader_loop` is `while not
    self._reader_stop.is_set()` *before* `src.read`. A leftover
    `while True` / `.clear()` still looks like a live hop
    (`retune_calls==1`, ring filled) while Sweep has already
    signalled stop — the next `_grab` races `src.read` on the
    same handle (SDR contract is one owner thread; `sync_config`
    mid-transfer hangs NIOS). A leftover USB call after `.set()`
    also writes a 5 ms chunk onto a ring `_stop_reader` is about
    to drop, so DecodeState `abs_t0` jumps. Direct `_reader_loop`
    after `_start_reader`; freeze the IQ thread. Occupancy/decode
    stay out. Did not copy #10 (short-join race).
    """
    monkeypatch.setattr("fpvscan.engine.threading.Thread", _CapturingThread)
    fs = 20e6
    src = MockSource(fs=10e6)
    eng = make_engine(src, capture_ms=80, ring_seconds=0.5)
    try:
        eng._start_reader(5800e6, fs, 0.5)
        dump = eng._ring
        ten_ms = max(2048, int(fs * 0.01))
        assert dump is not None
        assert dump.filled == ten_ms
        assert not eng._reader_stop.is_set()
        assert eng._reader_err is None
        eng._reader_stop.set()
        reads = 0

        def _read(n):
            nonlocal reads
            reads += 1
            raise AssertionError("USB read must not run after stop Event")

        src.read = _read
        eng._reader_loop(fs)
        assert reads == 0
        assert eng._reader_err is None
        assert eng._ring is dump
        assert dump.filled == ten_ms
        assert src.retune_calls == 1
        assert eng._reader_stop.is_set()
    finally:
        eng._stop_reader()
