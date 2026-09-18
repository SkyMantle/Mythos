"""Directionality checks for the shared-rate offline simulation corpus."""
from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import resample_poly

from fpvscan import hardware_rate, sweep_inspect


CAPS = Path(__file__).resolve().parents[1] / "caps"
# Fixture provenance labels only; none of these identifiers reaches DSP.
POSITIVE_STAMPS = (
    "20260912T075948",
    "20260912T090201",
    "20260912T092833",
    "20260912T094557",
    "20260912T095610",
    "20260912T100123",
    "20260912T112200",
    "20260912T112309",
    "20260912T112612",
    "20260912T113110",
)
PICTURE_STAMPS = (
    "20260912T112200",
    "20260912T112309",
    "20260912T112612",
)
SCAN = {
    "start_hz": 1.1e9,
    "stop_hz": 5.1e9,
    "sample_rate": 35e6,
    "channel_bw_hz": 12e6,
    "step_hz": 12e6,
    "fft_size": 8192,
    "averages": 16,
    "threshold_mode": "auto",
    "threshold_offset_db": 1.2,
    "threshold_min_db": 1.0,
    "threshold_max_db": 2.5,
    "threshold_k": 0.75,
    "noise_percentile": 25,
    "edge_guard": 0.95,
    "dc_notch_hz": 200e3,
    "fft_min_bw_hz": 0.6e6,
    "line_tol_hz": 200,
    "inspect_min_row_corr": 0.02,
    "inspect_min_lines": 32,
}
VIDEO = {
    "sample_rate": 20e6,
    "channel_bw_hz": 12e6,
    "deviation_hz": 4.8e6,
    "width": 160,
    "auto_levels": True,
}
PLAN = hardware_rate.derive_hardware_rate({
    "scan": SCAN, "video": VIDEO, "sdr": {},
})


def _fixture(stamp: str) -> tuple[np.ndarray, float]:
    matches = list(CAPS.glob(f"iq_{stamp}*.cf32"))
    assert len(matches) == 1, f"expected one IQ fixture for {stamp}"
    sidecar = json.loads(
        matches[0].with_suffix(".json").read_text(encoding="utf-8"),
    )
    source_fs = float(sidecar["sample_rate"])
    source = np.fromfile(matches[0], dtype=np.complex64)
    source = source[-int(source_fs * 0.080):]
    ratio = Fraction(
        int(PLAN.hardware_sample_rate_hz), int(source_fs),
    ).limit_denominator(2000)
    iq = resample_poly(source, ratio.numerator, ratio.denominator)
    return np.asarray(iq, dtype=np.complex64), PLAN.hardware_sample_rate_hz


@pytest.mark.parametrize("stamp", POSITIVE_STAMPS)
def test_labeled_positive_survives_derived_rate_pipeline(stamp: str) -> None:
    iq, fs = _fixture(stamp)
    metadata_center = 2.2e9  # arbitrary and deliberately unrelated to fixture RF
    proposals, _ = sweep_inspect.propose_iq(
        iq[:SCAN["fft_size"] * SCAN["averages"]],
        fs,
        metadata_center_hz=metadata_center,
        scan=SCAN,
    )
    assert proposals
    proposal = proposals[0]
    evidence = sweep_inspect.inspect_iq(
        iq,
        fs,
        signal_offset_hz=proposal.center_hz - metadata_center,
        bandwidth_hz=proposal.bandwidth_hz,
        scan=SCAN,
        video=VIDEO,
    )
    assert evidence.analog_evidence, evidence.diagnostics()


@pytest.mark.parametrize("stamp", PICTURE_STAMPS)
def test_known_user_picture_still_decodes_at_derived_rate(stamp: str) -> None:
    iq, fs = _fixture(stamp)
    proposals, _ = sweep_inspect.propose_iq(
        iq[:SCAN["fft_size"] * SCAN["averages"]],
        fs,
        metadata_center_hz=4.4e9,
        scan=SCAN,
    )
    assert proposals
    proposal = proposals[0]
    evidence = sweep_inspect.inspect_iq(
        iq,
        fs,
        signal_offset_hz=proposal.center_hz - 4.4e9,
        bandwidth_hz=proposal.bandwidth_hz,
        scan=SCAN,
        video=VIDEO,
    )
    assert evidence.video_confirmed, evidence.diagnostics()
