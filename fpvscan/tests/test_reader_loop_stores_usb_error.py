"""LOCK USB reader must stash a sync_rx failure and stop, not write it."""
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


class _UsbDrop(RuntimeError):
    """Distinct type so a leftover string/None cannot pass isinstance."""


def test_reader_loop_stores_usb_error_and_does_not_write(monkeypatch):
    """#73 `_start_reader` clears leftover `_reader_err` after Sweep
    / peek-expiry `_stop_reader` left it set. #108 chunk *size*
    with a successful `read`; #110 writes that chunk onto the
    dump. The live USB failure path is `self._reader_err = e;
    return` inside `_reader_loop` — `sync_rx` ERR_TIMEOUT /
    ERR_IO after the retry. A leftover swallow (`except: pass`)
    still looks like a live hop (`retune_calls==1`) while the
    ring stays at the 10 ms prime; shipped `capture_ms=80` then
    hits `len(iq) < n` and sleeps `idle_ms` for the whole
    glance. A leftover re-raise kills the daemon thread without
    stashing `_reader_err`, so `_do_lock` never re-arms. A
    leftover write-zeros on the failed read throws away
    PLL-settled IQ — DecodeState `abs_t0` jumps, tracking never
    shrinks. Direct `_reader_loop` after `_start_reader`;
    occupancy/decode stay out. Freeze the IQ thread. Did not
    copy #84 (`_do_lock` still consults leftover err *before*
    start). File pacing (#102) is not pinned.
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
        assert eng._reader_err is None
        ns: list[int] = []

        def _read(n):
            ns.append(int(n))
            raise _UsbDrop("USB drop")

        src.read = _read
        eng._reader_loop(fs)
        assert ns == [five_ms]
        assert isinstance(eng._reader_err, _UsbDrop)
        assert str(eng._reader_err) == "USB drop"
        assert eng._ring is ring
        assert ring.filled == ten_ms
        assert src.retune_calls == 1
    finally:
        eng._stop_reader()
