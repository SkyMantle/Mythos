"""LOCK USB reader must stash a read error without closing the radio."""
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


class _Src(MockSource):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.close_calls = 0

    def close(self):
        self.close_calls += 1
        super().close()


def test_reader_loop_usb_exception_does_not_close_the_radio(monkeypatch):
    """#112 `_reader_loop` stashes a USB `read` exception into
    `_reader_err` and returns without writing. #73 start clears
    leftover err. #91 a later field keeps `src` open. The
    missing half: the except path must not `src.close()`.
    `_run` catches the re-raise, sleeps, and retries `_do_lock`
    on the same handle. A leftover `self.src.close()` (copy-paste
    from `_run` finally) zeros the USB handle: the retry's
    `_start_reader` `retune_and_read`s a freed board (NIOS hang),
    or FileSource `read` after close raises and LOCK looks dead
    with a primed dump. Direct `_reader_loop` after
    `_start_reader`; occupancy/decode stay out. Freeze the IQ
    thread. Did not copy #84 (`_do_lock` still consults leftover
    err *before* start).
    """
    monkeypatch.setattr("fpvscan.engine.threading.Thread", _CapturingThread)
    fs = 20e6
    src = _Src(fs=10e6)
    src.open()
    eng = make_engine(src, capture_ms=80, ring_seconds=0.5)
    try:
        eng._start_reader(5800e6, fs, 0.5)
        dump = eng._ring
        ten_ms = max(2048, int(fs * 0.01))
        assert dump is not None
        assert dump.filled == ten_ms
        assert eng._reader_err is None
        assert src.opened is True
        assert src.close_calls == 0

        def _read(_n):
            raise RuntimeError("usb timeout")

        src.read = _read
        eng._reader_loop(fs)
        assert isinstance(eng._reader_err, RuntimeError)
        assert "usb timeout" in str(eng._reader_err)
        assert dump.filled == ten_ms
        assert eng._ring is dump
        assert src.opened is True
        assert src.close_calls == 0
        assert src.retune_calls == 1
        assert not eng._reader_stop.is_set()
    finally:
        eng._stop_reader()
        assert src.close_calls == 0
        assert src.opened is True
