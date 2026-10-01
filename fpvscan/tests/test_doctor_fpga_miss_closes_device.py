"""doctor.py must close the bladeRF even when the FPGA is empty."""
from __future__ import annotations

from pathlib import Path


SRC = (Path(__file__).resolve().parents[1]
       / "scripts" / "doctor.py").read_text(encoding="utf-8")


def test_fpga_miss_still_closes_the_open_handle():
    """Step 6 reports an unconfigured FPGA but must still
    `bladerf_close`. Returning from the else-branch (or only closing
    on `conf > 0`) leaves the exclusive USB handle held by a 'failed'
    doctor. The next bringup/run.py then dies with 'device busy' /
    bladeRF-cli 'already in use' — the operator thinks the board is
    wedged. Close sits after both FPGA branches, not inside success.
    """
    fpga_at = SRC.index("6. FPGA")
    close_at = SRC.index("bladerf_close(dev)")
    assert close_at > fpga_at
    assert "bladerf_is_fpga_configured" in SRC
    miss = SRC[SRC.index('bad("не залита")'):close_at]
    assert "return" not in miss
