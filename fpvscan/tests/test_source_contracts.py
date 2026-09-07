"""SDR contracts that gate LOCK SNR: PLL settle, Q11 scale, sim gain."""
from __future__ import annotations

import numpy as np

from fpvscan.sdr.base import SdrSource
from fpvscan.sdr.bladerf import CHANNEL_RX, CHANNEL_TX, SC16_SCALE
from fpvscan.sdr.sim import Emitter, SimSource


class _CountingSource(SdrSource):
    name = "count"

    def __init__(self):
        self.reads = []
        self._fc = 0.0
        self._fs = 4e6

    def open(self): pass
    def close(self): pass
    def set_gain(self, db): pass
    def set_center_freq(self, hz): self._fc = float(hz)
    def set_sample_rate(self, hz): self._fs = float(hz)

    @property
    def center_freq(self): return self._fc

    @property
    def sample_rate(self): return self._fs

    def read(self, n: int) -> np.ndarray:
        self.reads.append(int(n))
        return np.zeros(int(n), dtype=np.complex64)


def test_base_retune_dumps_pll_then_reads_requested_length():
    """Default retune_and_read throws away n//4 so inspect is not on unsettled IQ."""
    src = _CountingSource()
    n = 4000
    out = src.retune_and_read(1280e6, n)
    assert src.center_freq == 1280e6
    assert src.reads == [n // 4, n]
    assert len(out) == n


def test_sc16_q11_interleave_is_i_then_q_scaled_by_2048():
    """Wrong scale or IQ order makes FM deviation (and LOCK) garbage."""
    assert SC16_SCALE == 2048.0
    raw = np.array([2048, 0, 0, -2048], dtype=np.int16)
    iq = (raw.astype(np.float32) / SC16_SCALE).view(np.complex64)
    assert iq.shape == (2,)
    assert abs(iq[0] - (1 + 0j)) < 1e-5
    assert abs(iq[1] - (0 - 1j)) < 1e-5
    clip_frac = float(np.mean(np.abs(raw) > 2000))
    assert clip_frac == 0.5


def test_bladerf_channel_rx_is_even_tx_is_odd():
    """CHANNEL_RX(n) = (n<<1)|0 — swapping this enables the TX path."""
    assert CHANNEL_RX(0) == 0
    assert CHANNEL_RX(1) == 2
    assert CHANNEL_TX(0) == 1
    assert CHANNEL_TX(1) == 3


def test_sim_gain_scales_amplitude():
    """Default scene is calibrated at gain 40; a 20 dB cut must drop the tone."""
    em = [Emitter(5800e6, -12.0, "cw", 1e6, label="tone")]
    loud = SimSource(emitters=em, noise_db=-140, seed=1)
    quiet = SimSource(emitters=em, noise_db=-140, seed=1)
    for src, g in ((loud, 40.0), (quiet, 20.0)):
        src.open()
        src.set_sample_rate(2e6)
        src.set_center_freq(5800e6)
        src.set_gain(g)
    a = float(np.mean(np.abs(loud.read(4096))))
    b = float(np.mean(np.abs(quiet.read(4096))))
    assert a > 0
    assert b / a < 0.45
    assert b / a > 0.2
