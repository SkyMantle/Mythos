"""Quick-tune must fire immediately; missing profiles must fall back."""
from __future__ import annotations

from pathlib import Path

from fpvscan.sdr.bladerf import META_FLAG_RX_NOW, RETUNE_NOW


SRC = (Path(__file__).resolve().parents[1]
       / "fpvscan" / "sdr" / "bladerf.py").read_text(encoding="utf-8")


def test_quick_retune_schedules_now_not_a_timestamp():
    """RETUNE_NOW is 0 in libbladeRF. Passing the current timestamp (or
    the RX metadata timestamp) schedules the hop in the past/future and
    the sweep sits on the previous LO until the schedule fires.
    """
    assert RETUNE_NOW == 0
    assert "self._dev, self.ch, RETUNE_NOW, int(hz), buf" in SRC


def test_meta_reads_request_samples_immediately():
    """Without META_FLAG_RX_NOW the first sync_rx after a retune waits
    for a timestamped slot. Sweep then measures PLL settle as if it were
    occupancy, and inspect confirms a neighbor spur.
    """
    assert META_FLAG_RX_NOW == (1 << 31)
    assert "meta.flags = META_FLAG_RX_NOW" in SRC
    assert "C.byref(meta) if self.use_meta else None" in SRC


def test_fast_retune_falls_back_when_the_profile_is_missing():
    """xA4 / old libbladeRF often refuse get_quick_tune for part of the
    400 MHz–6 GHz plan. Treating that as a hard error (or skipping the
    hop) punches holes in the sweep. The fast path must settle via the
    ordinary retune_and_read.
    """
    body = SRC[SRC.index("def retune_and_read_fast"):]
    assert "if self.quick_retune(hz):" in body
    assert "return self.retune_and_read(hz, n)" in body
    assert "self.read(max(1024, int(self._fs * 20e-6)))" in body
