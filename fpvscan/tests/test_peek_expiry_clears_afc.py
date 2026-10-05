"""Auto-peek expiry must drop leftover digital AFC before the next sweep."""
from __future__ import annotations

import time

import numpy as np

from tests.helpers import make_engine


def test_peek_expiry_zeros_afc_and_accumulator():
    """#7 checked mode/target/reader. It did not check `_afc` / `_acc`.
    The next `_maybe_peek` on current main does not clear those
    (that is draft #22). Expiry is the path that does. Leaving a
    1.5 MHz shift in `_afc` makes the next hold channelize off the
    VTx; leaving `_acc` paints the new bird with the old field.
    """
    eng = make_engine()
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6
    eng.state.auto = True
    eng.state.auto_until = time.time() - 1.0
    eng._afc = 1.5e6
    eng._acc = np.ones((8, 8), dtype=np.float32)
    try:
        eng._do_lock()
        assert eng.state.mode == "SWEEP"
        assert eng._afc == 0.0
        assert eng._acc is None
    finally:
        eng._stop_reader()
