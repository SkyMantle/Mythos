"""Sim retune is instant — dumping n//4 like a real PLL starves inspect."""
from __future__ import annotations

from fpvscan.sdr.sim import SimSource


class _SpySim(SimSource):
    def __init__(self):
        super().__init__(emitters=[], noise_db=-90.0, seed=1)
        self.reads: list[int] = []

    def read(self, n: int):
        self.reads.append(int(n))
        return super().read(n)


def test_sim_retune_reads_exactly_n_not_settle_plus_n():
    """SdrSource.retune_and_read dumps n//4 to let a bladeRF PLL settle.
    SimSource overrides that: there is no PLL, and inspect_ms is already
    a short capture. A dump would throw away a quarter of the line-rate
    window so PAL/NTSC classify fails on a scene that is actually video.
    """
    src = _SpySim()
    src.open()
    n = 8000
    iq = src.retune_and_read(1280e6, n)
    assert src.center_freq == 1280e6
    assert len(iq) == n
    assert src.reads == [n], (
        f"sim retune issued reads {src.reads}; "
        f"n//4 dump would look like {[n // 4, n]}"
    )
