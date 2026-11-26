"""LOCK stop-reader must drop the ring without toggling bias-tee."""
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
        self.set_bias_tee_calls = 0
        self.set_bias_tee_on: list[bool] = []

    def set_bias_tee(self, on):
        self.set_bias_tee_calls += 1
        self.set_bias_tee_on.append(bool(on))
        return super().set_bias_tee(on)


def test_stop_reader_does_not_set_bias_tee(monkeypatch):
    """#50 `_run` *does* `set_bias_tee` then `_apply_bias_tee_gain`
    after open. The `bias_tee` command *does* both on a click.
    #27 toggling the tee during a live `sync_rx` is the USB
    race. #115 stop does not `src.close()`. #120 start does
    not `set_bias_tee`. The missing half: `_stop_reader`
    itself must not `src.set_bias_tee`. Teardown is Event +
    join + null ring, not the 4.5 V rail. Sweep command /
    peek-expiry / the >1 Hz fs restart (#2) call stop, then
    the next `_grab` / `_start_reader` on the same handle. A
    leftover `self.src.set_bias_tee(self.cfg["sdr"].get(
    "bias_tee", False))` (copy-paste from `_run`; helpers
    omit the key so default False) drops the LNA on every
    Sweep while VGA is still the compensated 15 dB — SNR
    collapses, or the rail races the orphan still in
    `sync_rx` (ERR_IO / NIOS). A leftover `set_bias_tee(True)`
    raises 4.5 V on a teardown the operator did not ask for.
    Direct `_stop_reader` after planting leftover tee ON;
    occupancy / decode stay out. Freeze the IQ thread. Did
    not pin join timeout=2 (#10).
    """
    monkeypatch.setattr("fpvscan.engine.threading.Thread", _CapturingThread)
    fs = 20e6
    src = _Src(fs=10e6)
    src.bias = True
    src.gain = 15.0
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
        assert src.set_bias_tee_calls == 0
        assert src.bias is True
        assert eng._lock_tuned == 5800e6
        eng._stop_reader()
        assert src.set_bias_tee_calls == 0
        assert src.set_bias_tee_on == []
        assert src.bias is True
        assert src.bias is not False
        assert src.gain == 15.0
        assert eng._reader_thread is None
        assert eng._ring is None
        assert leftover.join_calls != []
        assert ev.is_set()
        assert src.retune_calls == 1
        assert src.set_sample_rate_calls == 0
    finally:
        eng._stop_reader()
        assert src.set_bias_tee_calls == 0
        assert src.bias is True
