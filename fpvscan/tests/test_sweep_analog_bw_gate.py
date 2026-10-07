"""Sweep must not INSPECT occupancies that cannot be analog video."""
from __future__ import annotations

from unittest.mock import patch

from fpvscan.dsp.spectrum import Occupancy
from tests.helpers import MockSource, make_engine


def _occ(bw_hz: float) -> Occupancy:
    return Occupancy(
        center_hz=5800e6, bandwidth_hz=bw_hz, peak_db=-18.0, snr_db=14.0,
    )


def test_inspect_only_runs_for_analog_occupancy_bandwidth():
    """#31 only grepped `2.5e6` / `35e6` in `_do_sweep`. find_occupied's
    YAML min_bw is 2 MHz, so a 2.2 MHz Wi-Fi-like blob still reaches
    this gate. Inspecting it burns a 25 ms retune per hop and can
    auto-peek onto noise. A 40 MHz blob is two VTx; classifying the
    pair as one hit parks LOCK between them. Inclusive 2.5–35 MHz is
    the bench 4.3 MHz VTx contract.
    """
    src = MockSource()
    eng = make_engine(src)
    eng._sweep_plan = lambda: [5800e6]
    inspected: list[float] = []
    eng._inspect = lambda _iq, _c, _fs, occ: inspected.append(occ.bandwidth_hz)

    cases = (
        (1.0e6, False),
        (2.2e6, False),
        (2.5e6, True),
        (8.0e6, True),
        (35.0e6, True),
        (40.0e6, False),
    )
    for bw, want in cases:
        inspected.clear()
        eng._sweep_i = 0
        with patch(
            "fpvscan.engine.spectrum.find_occupied", return_value=[_occ(bw)],
        ):
            eng._do_sweep()
        if want:
            assert inspected == [bw], f"bw={bw / 1e6:.1f} MHz should inspect"
        else:
            assert inspected == [], f"bw={bw / 1e6:.1f} MHz should skip"
