"""LOCK later field must not toggle bias-tee on a live reader."""
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


def test_do_lock_later_field_does_not_set_bias_tee(monkeypatch):
    """#50 `_run` *does* `set_bias_tee` then `_apply_bias_tee_gain`
    after open. The `bias_tee` command *does* both on a click.
    #27 toggling the tee during a live `sync_rx` is the USB
    race. #120 `_start_reader` does not `set_bias_tee`. #91 /
    #115 a later field keeps the live ring and a short
    snapshot returns without stop. The missing half: that
    later-field miss must not `src.set_bias_tee`. The 4.5 V
    rail is process / operator lifetime, not the field. A
    leftover `self.src.set_bias_tee(self.cfg["sdr"].get(
    "bias_tee", False))` (copy-paste from `_run`; helpers
    omit the key so default False) would pass #120 (start
    never hits it) and still drop the LNA on every field
    while VGA is still the compensated 15 dB — SNR
    collapses, or the rail races the live IQ thread (ERR_IO
    / NIOS). A leftover `set_bias_tee(True)` raises 4.5 V on
    a field the operator did not ask for. Direct `_do_lock`
    after `_start_reader` with leftover tee ON; occupancy /
    decode stay out. Freeze the IQ thread. Src fs matches
    `video.sample_rate` so the 1 Hz restart (#2) does not
    fire. File pacing (#102) is not pinned.
    """
    monkeypatch.setattr("fpvscan.engine.threading.Thread", _CapturingThread)
    fs = 20e6
    src = _Src(fs=fs)
    src.bias = True
    src.gain = 15.0
    eng = make_engine(src, capture_ms=80, ring_seconds=0.5, idle_ms=0)
    try:
        eng.state.mode = "LOCK"
        eng.state.lock_target = 5800e6
        eng._start_reader(5800e6, fs, 0.5)
        leftover = eng._reader_thread
        dump = eng._ring
        ten_ms = max(2048, int(fs * 0.01))
        field = int(fs * 0.080)
        assert leftover is not None
        assert dump is not None
        assert dump.filled == ten_ms
        assert dump.filled < field
        assert src.set_bias_tee_calls == 0
        assert src.bias is True
        assert src.retune_calls == 1
        eng._do_lock()
        assert eng._reader_thread is leftover
        assert eng._ring is dump
        assert dump.filled == ten_ms
        assert leftover.started == 1
        assert leftover.join_calls == []
        assert src.retune_calls == 1
        assert src.set_bias_tee_calls == 0
        assert src.set_bias_tee_on == []
        assert src.bias is True
        assert src.bias is not False
        assert src.gain == 15.0
        assert src.set_sample_rate_calls == 0
        assert eng._lock_tuned == 5800e6
        assert eng._lock_n == 0
        assert eng._reader_err is None
    finally:
        eng._stop_reader()
