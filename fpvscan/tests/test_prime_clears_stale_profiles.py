"""Re-priming quick-tune must drop profiles from the previous plan."""
from __future__ import annotations

import inspect

from fpvscan.sdr import bladerf as B


def test_prime_clears_quick_before_walking_freqs():
    """`_sweep_plan()` changes with YAML step / fs. A second
    `prime_quick_tune` that does not `_quick.clear()` first keeps
    ctypes buffers from the old device session (or leftover Hz
    that are no longer in the plan). `retune_and_read_fast` then
    hops with a 20 µs settle onto a stale LO — silent wrong-band
    occupancies, no Python exception.
    """
    src = inspect.getsource(B.BladeRF.prime_quick_tune)
    assert src.index("self._quick.clear()") < src.index("for f in freqs")
    assert src.index("self._quick.clear()") < src.index("set_center_freq")
