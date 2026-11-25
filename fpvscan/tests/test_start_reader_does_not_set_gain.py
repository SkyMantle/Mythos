"""LOCK start-reader must not rewrite VGA gain on a hop."""
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
        self.set_gain_calls = 0
        self.set_gain_db: list[float] = []

    def set_gain(self, db):
        self.set_gain_calls += 1
        self.set_gain_db.append(float(db))
        super().set_gain(db)


def test_start_reader_does_not_set_gain(monkeypatch):
    """#50 `_run` *does* `set_gain(gain_db)` once after open,
    then `set_bias_tee` + `_apply_bias_tee_gain` (YAML 30 − 15
    LNA offset → 15 dB while the tee is on). #116 start does
    not `set_sample_rate` / open / close. #119 start does not
    `set_center_freq`. The missing half: `_start_reader`
    itself must not `src.set_gain`. VGA belongs to `_run` /
    the bias-tee command, not the hop. A leftover
    `self.src.set_gain(self.cfg["sdr"]["gain_db"])`
    (copy-paste from `_run`) rewrites 15 → 30 on every
    Sweep→LOCK / AFC recenter / >1 Hz fs restart while the
    LNA is still on — ADC clips, CVBS paints bars, the ring
    still looks primed (`retune_calls==1`). A leftover
    `_apply_bias_tee_gain(False)` is the same 30 dB. Direct
    `_start_reader` with leftover compensated VGA; occupancy
    / decode stay out. Freeze the IQ thread.
    """
    monkeypatch.setattr("fpvscan.engine.threading.Thread", _CapturingThread)
    fs = 20e6
    src = _Src(fs=10e6)
    src.gain = 15.0
    eng = make_engine(src, capture_ms=80, ring_seconds=0.5)
    try:
        assert src.set_gain_calls == 0
        assert src.gain == 15.0
        eng._start_reader(5800e6, fs, 0.5)
        ten_ms = max(2048, int(fs * 0.01))
        dump = eng._ring
        assert dump is not None
        assert dump.filled == ten_ms
        assert src.retune_calls == 1
        assert src._fc == 5800e6
        assert src.set_gain_calls == 0
        assert src.set_gain_db == []
        assert src.gain == 15.0
        assert src.gain != 30.0
        assert src.set_sample_rate_calls == 0
    finally:
        eng._stop_reader()
