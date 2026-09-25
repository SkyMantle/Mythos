"""bringup's decode path must match LOCK's FM scale, not inspect's."""
from __future__ import annotations

from pathlib import Path


SRC = (Path(__file__).resolve().parents[1]
       / "scripts" / "bringup.py").read_text(encoding="utf-8")


def test_bringup_decode_uses_lock_deviation_and_occupancy_pad():
    """Classify uses `occ.bw/5` (inspect). Decode used to inherit that
    and washed the sync tip, so the PNG disagreed with the LOCK
    picture the operator then opened on the same capture. Keep
    decode at `max(occ*1.6, 8e6)` and `deviation_hz=bw/4` — the
    same contracts as Engine._do_lock.
    """
    assert "max(top.bandwidth_hz * 1.6, 8e6)" in SRC
    assert "deviation_hz=bw / 4" in SRC
    assert "fm_demod(ch_iq, fs2, top.bandwidth_hz / 5)" in SRC
    assert "demod.deemphasis(base, fs_ch)" in SRC
