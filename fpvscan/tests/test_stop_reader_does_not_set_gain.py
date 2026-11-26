"""LOCK stop-reader must drop the ring without rewriting VGA gain."""
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
        self.join_calls: list = []

    def start(self):
        self.started += 1

    def join(self, timeout=None):
        self.join_calls.append(timeout)


class _Src(MockSource):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.set_gain_calls = 0
        self.set_gain_db: list[float] = []

    def set_gain(self, db):
        self.set_gain_calls += 1
        self.set_gain_db.append(float(db))
        super().set_gain(db)


def test_stop_reader_does_not_set_gain(monkeypatch):
    """#50 `_run` *does* `set_gain(gain_db)` once after open,
    then `_apply_bias_tee_gain`. The `bias_tee` command *does*
    both on a click. #115 stop does not `src.close()`. #116
    stop does not null `_lock_tuned`. #120 start does not
    `set_gain`; stop does not `_lock_n = 0`. The missing
    half: `_stop_reader` itself must not `src.set_gain`.
    Teardown is Event + join + null ring, not the radio.
    Sweep command / peek-expiry / the >1 Hz fs restart (#2)
    call stop, then the next `_grab` / `_start_reader` on
    the same handle. A leftover
    `self.src.set_gain(self.cfg["sdr"]["gain_db"])` or
    `_apply_bias_tee_gain(False)` (copy-paste from `_run`)
    rewrites 15 → 30 while the LNA is still on — ADC clips
    on the next Sweep hop / LOCK restart, the teardown still
    looks live (`thread is None`, ring is None). Direct
    `_stop_reader` after planting leftover compensated VGA;
    occupancy / decode stay out. Freeze the IQ thread. Did
    not pin join timeout=2 (#10).
    """
    monkeypatch.setattr("fpvscan.engine.threading.Thread", _CapturingThread)
    fs = 20e6
    src = _Src(fs=10e6)
    src.gain = 15.0
    src.bias = True
    eng = make_engine(src, capture_ms=80, ring_seconds=0.5)
    try:
        eng._start_reader(5800e6, fs, 0.5)
        leftover = eng._reader_thread
        dump = eng._ring
        ev = eng._reader_stop
        ten_ms = max(2048, int(fs * 0.01))
        assert leftover is not None
        assert dump is not None
        assert dump.filled == ten_ms
        assert src.set_gain_calls == 0
        assert src.gain == 15.0
        assert eng._lock_tuned == 5800e6
        eng._stop_reader()
        assert src.set_gain_calls == 0
        assert src.set_gain_db == []
        assert src.gain == 15.0
        assert src.gain != 30.0
        assert src.bias is True
        assert eng._reader_thread is None
        assert eng._ring is None
        assert leftover.join_calls != []
        assert ev.is_set()
        assert src.retune_calls == 1
        assert src.set_sample_rate_calls == 0
    finally:
        eng._stop_reader()
        assert src.set_gain_calls == 0
        assert src.gain == 15.0
