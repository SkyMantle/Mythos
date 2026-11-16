"""A channel-change hop must drop the previous VTx genlock, not reuse it."""
from __future__ import annotations

from fpvscan.dsp.cvbs import DecodeState
from tests.helpers import make_engine


class _FrozenThread:
    def __init__(self, target=None, args=(), kwargs=None, daemon=None):
        self.daemon = daemon

    def start(self):
        pass

    def join(self, timeout=None):
        pass


def test_start_reader_reseeds_leftover_decode_state_and_dec(monkeypatch):
    """#50 source-order: `_start_reader` constructs `DecodeState()`
    and `_lock_dec = None` *after* the hop. #85 a later field of
    the *same* hold keeps leftover `_lock_state` / `_lock_dec` so
    TBC phase survives. The runtime hole: a leftover
    `self._lock_state = leftover` / `self._lock_dec = 4` (or
    `leftover.period` kept, copy-paste from the later-field keep)
    still looks like a live hop (`retune_calls==1`) while the new
    VTx is tracked with the old line period. Tracking then asks
    for the wrong field length, `_lock_dec != dec` may not fire
    if measured BW happens to match, and every field grabs full
    `capture_ms`. Direct `_start_reader`; occupancy/decode stay
    out. Freeze the IQ thread so USB fill is not the path.
    """
    monkeypatch.setattr("fpvscan.engine.threading.Thread", _FrozenThread)
    eng = make_engine()
    leftover = DecodeState()
    leftover.period = 1278.0
    leftover.standard = "PAL"
    leftover.abs_t0 = 50_000.0
    leftover.lost = 0
    leftover.sign = -1.0
    eng._lock_state = leftover
    eng._lock_dec = 4
    try:
        eng._start_reader(5740e6, 20e6, 0.05)
        assert eng._lock_state is not leftover
        assert isinstance(eng._lock_state, DecodeState)
        assert eng._lock_state.period is None
        assert eng._lock_state.abs_t0 is None
        assert eng._lock_state.standard == "?"
        assert eng._lock_state.lost == 0
        assert eng._lock_state.sign == 1.0
        assert eng._lock_dec is None
        assert eng._lock_dec != 4
        assert eng._lock_tuned == 5740e6
        assert eng.src.retune_calls == 1
        assert eng._reader_thread is not None
        assert eng.src.center_freq == 5740e6
    finally:
        eng._stop_reader()
