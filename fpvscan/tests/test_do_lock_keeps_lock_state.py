"""A LOCK field must not reseed genlock while the IQ reader is already up."""
from __future__ import annotations

from fpvscan.dsp.cvbs import DecodeState
from tests.helpers import make_engine


def test_do_lock_keeps_leftover_genlock_when_reader_is_up():
    """#69 Clear keeps `_lock_state` / `_lock_dec`; #50 `_start_reader`
    *does* reseed `DecodeState()` on a channel change; #81 `_do_sweep`
    keeps genlock on an 8 MHz hop. `_do_lock` reseeds only when
    `_lock_dec` no longer matches `fs / ch_bw` (occupancy changed).
    The missing half: a later field on the same bird, same decimation,
    must not `_lock_state = DecodeState()` or `_lock_dec = None`.
    A leftover wipe on every decode throws away the line period /
    vsync phase the TBC already paid for — verticals jitter, the
    tracking window never shrinks, and every field grabs full
    `capture_ms`. First field starts the reader (hop *does* reseed);
    leftover is injected after that. `lost=1` so the tracking-window
    20 ms floor does not inflate `n` on zeros IQ.
    """
    eng = make_engine()
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    leftover = DecodeState(
        sign=-1.0, period=80.0, standard="PAL",
        abs_t0=12.0, lost=1,
    )
    try:
        eng._do_lock()
        eng._lock_state = leftover
        eng._lock_dec = 2
        eng._do_lock()
        assert eng._lock_state is leftover
        assert eng._lock_dec == 2
        assert leftover.sign == -1.0
        assert leftover.standard == "PAL"
        assert leftover.period == 80.0
        assert eng._lock_tuned == 5800e6
        assert eng._reader_thread is not None
    finally:
        eng._stop_reader()
