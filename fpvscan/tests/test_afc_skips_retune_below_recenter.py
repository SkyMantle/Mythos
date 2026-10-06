"""Digital AFC must not hop the LO on every correction."""
from __future__ import annotations

from tests.helpers import MockSource, make_engine


def test_small_afc_does_not_restart_the_reader():
    """`want` is `f + lo_offset`, not `f + off + _afc`. Putting the
    digital correction back into the physical hop (the pre-#4 path)
    calls `_start_reader` on every AFC tick, which resets
    DecodeState and is why LOCK video rolled when the free-running
    VTx drifted. #46 only checked that a recenter writes
    `_lock_tuned` back to `want`. This is the complementary
    contract: 400 kHz of leftover `_afc` must not retune.
    """
    src = MockSource()
    eng = make_engine(src)
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    try:
        eng._do_lock()
        assert src.retune_calls == 1
        tuned = eng._lock_tuned
        eng._afc = 400e3
        eng._do_lock()
        assert src.retune_calls == 1
        assert eng._lock_tuned == tuned == 5800e6
    finally:
        eng._stop_reader()
