"""A sweep hop must not cancel the glance `_maybe_peek` just armed."""
from __future__ import annotations

from types import SimpleNamespace

from fpvscan.engine import Detection
from tests.helpers import make_engine


def test_do_sweep_keeps_peek_hold_after_inspect(monkeypatch):
    """#70 `_maybe_peek` sets `mode=LOCK` / `lock_target` /
    `auto=True` / `auto_until=now+secs` on a confirm hit. #79
    `_start_reader` then keeps that identity across the LOCK
    hop. The sweep hop is the missing half: `_inspect` runs
    *inside* `_do_sweep`, after the grab, and the next while
    iteration is what sees LOCK and returns. A leftover
    `mode="SWEEP"` / `auto=False` / `lock_target=None` after
    the occupancy loop cancels the glance on the same step
    that armed it — the 400 MHz–6 GHz walk continues and the
    operator never sees the bird. Occupancy is injected so
    this does not depend on a live analog hit; `_inspect` is
    the peek so classify/USB stay out of the path.
    `fixed_freq` keeps this to one grab.
    """
    import fpvscan.engine as engine_mod

    eng = make_engine()
    eng.src.fixed_freq = True
    eng.src._fc = 5800e6
    eng.cfg["scan"]["fft_size"] = 64
    eng.cfg["scan"]["averages"] = 1

    monkeypatch.setattr(
        engine_mod.spectrum,
        "find_occupied",
        lambda *a, **k: [SimpleNamespace(
            center_hz=5800e6, bandwidth_hz=8e6, snr_db=18.0)],
    )

    def inspect_as_peek(_iq, _center_hz, _fs, occ):
        eng._maybe_peek(Detection(
            freq_hz=occ.center_hz,
            bandwidth_hz=occ.bandwidth_hz,
            snr_db=occ.snr_db,
            confidence=0.9,
        ))

    eng._inspect = inspect_as_peek
    try:
        eng._do_sweep()
        assert eng.src.retune_calls == 1
        assert eng._lock_tuned is None
        assert eng.state.mode == "LOCK"
        assert eng.state.auto is True
        assert eng.state.lock_target == 5800e6
        assert eng.state.auto_until > 0
        assert eng.snapshot()["mode"] == "LOCK"
        assert eng.snapshot()["auto"] is True
        assert eng.snapshot()["lock_target"] == 5800e6
        assert eng.state.mode != "SWEEP"
        assert eng.state.lock_target is not None
    finally:
        eng._stop_reader()
