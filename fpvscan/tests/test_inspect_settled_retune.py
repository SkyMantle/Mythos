"""INSPECT must settle the PLL; the sweep fast path is too short for line-rate."""
from __future__ import annotations

import numpy as np

from fpvscan.dsp.spectrum import Occupancy
from tests.helpers import MockSource, make_engine


class _BothPaths(MockSource):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.fast_calls = 0
        self.settled_hz: list[float] = []

    def retune_and_read(self, hz: float, n: int) -> np.ndarray:
        self.settled_hz.append(float(hz))
        return super().retune_and_read(hz, n)

    def retune_and_read_fast(self, hz: float, n: int) -> np.ndarray:
        self.fast_calls += 1
        self._fc = float(hz)
        return self._iq(n)


def test_inspect_uses_settled_retune_not_sweep_fast_path():
    """Sweep `_grab` prefers `retune_and_read_fast` (20 µs after a stored
    profile). INSPECT measures 15.6 kHz line-rate and needs the PLL
    dump in `retune_and_read`. Routing inspect through `_grab` leaves
    the discriminator on unsettled IQ and Wi-Fi-like blobs get confirmed.
    """
    src = _BothPaths(fs=20e6)
    eng = make_engine(src=src)
    # Short inspect: we only assert which retune path ran, not classify.
    eng.cfg["scan"]["inspect_ms"] = 1
    occ = Occupancy(center_hz=5800e6, bandwidth_hz=10e6,
                    peak_db=-18.0, snr_db=12.0)
    sweep_iq = np.zeros(4096, dtype=np.complex64)
    eng._inspect(sweep_iq, 5782.5e6, 20e6, occ)
    assert src.fast_calls == 0, (
        "INSPECT used the sweep fast path; line-rate needs a settled dump"
    )
    assert src.settled_hz, "INSPECT never called retune_and_read"
    assert src.settled_hz[0] == 5800e6
