"""Omitted quick_tune must not walk the sweep plan at boot."""
from __future__ import annotations

from tests.helpers import MockSource, make_engine


class _PrimeSource(MockSource):
    def __init__(self):
        super().__init__()
        self.primed = None

    def prime_quick_tune(self, freqs):
        self.primed = list(freqs)
        return len(self.primed)


def test_omitted_quick_tune_does_not_prime():
    """#21 / #57 forwarded explicit `quick_tune` into BladeRF.use_meta.
    helpers.py omits the key. `_run` must not call `prime_quick_tune`
    when it is missing. A leftover True walks every 400 MHz–6 GHz hop
    before the first sweep (seconds of USB on a Pi 5) and a failed
    prime still prints as success-path noise. Shipped YAML is false.
    """
    src = _PrimeSource()
    eng = make_engine(src=src)
    assert "quick_tune" not in eng.cfg["sdr"]
    eng._stop.set()
    eng._run()
    assert src.primed is None
    assert src.opened is False  # finally close()
