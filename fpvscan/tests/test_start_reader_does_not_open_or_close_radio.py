"""LOCK start-reader must retune without owning the USB handle."""
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
        self.open_calls = 0

    def open(self):
        self.open_calls += 1
        super().open()

    def close(self):
        self.close_calls += 1
        super().close()


def test_start_reader_does_not_open_or_close_the_radio(monkeypatch):
    """#19 / #50 `_run` *does* `src.open()` before the loop and
    `src.close()` in `finally` after `_stop_reader` — process
    lifetime. #115 `_stop_reader` does not close after a
    pre-opened start. #91 a later field keeps the handle open.
    The missing half: `_start_reader` itself must not
    `src.open()` / `src.close()`. The board is already opened
    by `_run`; start is `retune_and_read` + IQ thread. A leftover
    `self.src.open()` (copy-paste from `_run`) double-opens
    libbladeRF on every hop / AFC recenter / fs restart — second
    open fails or leaks an RX stream, LOCK looks dead with a
    primed dump. A leftover `self.src.close()` before retune
    zeros the handle while `_run` still owns it (NIOS hang if
    the orphan is still in `sync_rx`). Direct `_start_reader`
    on a closed board *and* after a simulated `_run` open;
    occupancy/decode stay out. Freeze the IQ thread.
    """
    monkeypatch.setattr("fpvscan.engine.threading.Thread", _CapturingThread)
    fs = 20e6
    src = _Src(fs=10e6)
    eng = make_engine(src, capture_ms=80, ring_seconds=0.5)
    try:
        assert src.opened is False
        assert src.open_calls == 0
        assert src.close_calls == 0
        eng._start_reader(5800e6, fs, 0.5)
        ten_ms = max(2048, int(fs * 0.01))
        dump = eng._ring
        assert dump is not None
        assert dump.filled == ten_ms
        assert src.retune_calls == 1
        assert src._fc == 5800e6
        assert src.open_calls == 0
        assert src.close_calls == 0
        assert src.opened is False

        eng._stop_reader()
        src.open()
        assert src.open_calls == 1
        assert src.opened is True
        eng._start_reader(5800e6, fs, 0.5)
        dump = eng._ring
        assert dump is not None
        assert dump.filled == ten_ms
        assert src.retune_calls == 2
        assert src.open_calls == 1
        assert src.close_calls == 0
        assert src.opened is True
    finally:
        eng._stop_reader()
        assert src.close_calls == 0
