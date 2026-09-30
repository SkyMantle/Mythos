"""record.py --agc must reach BladeRF, not stay a dead flag."""
from __future__ import annotations

from pathlib import Path


SRC = (Path(__file__).resolve().parents[1]
       / "scripts" / "record.py").read_text(encoding="utf-8")


def test_record_forwards_agc_into_bladerf():
    """`--agc` is how an operator hunts a weak VTx without a
    bringup gain sweep. If the flag is parsed but not passed to
    `BladeRF(..., agc=)`, the capture stays at the numeric
    `gain_db` and a later file replay still looks empty. The
    constructor is the only place AGC is applied before open().
    """
    assert 'ap.add_argument("--agc", action="store_true")' in SRC
    assert ("BladeRF(device=a.device, lib_path=a.lib, "
            "gain_db=a.gain, agc=a.agc)") in SRC
