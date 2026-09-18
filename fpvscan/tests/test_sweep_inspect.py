from __future__ import annotations

from pathlib import Path
from queue import Queue

import numpy as np
import pytest

from fpvscan import scan_gate, scan_view, sweep_inspect
from fpvscan.dsp import adaptive_if, cvbs, spectrum
from fpvscan.dsp.demod import LINE_PAL
from fpvscan.engine import Engine


CAPS = Path(__file__).resolve().parents[1] / "caps"
SCAN = {
    "fft_size": 8192,
    "averages": 16,
    "threshold_mode": "auto",
    "threshold_db": 1.0,
    "threshold_offset_db": 1.2,
    "threshold_min_db": 1.0,
    "threshold_max_db": 2.5,
    "threshold_k": 0.75,
    "noise_percentile": 25,
    "edge_guard": 0.95,
    "dc_notch_hz": 200e3,
    "fft_min_bw_hz": 0.6e6,
    "min_bw_hz": 1.5e6,
    "max_bw_hz": 30e6,
    "inspect_bw_hz": 18e6,
    "line_tol_hz": 200,
    "inspect_min_row_corr": 0.02,
    "inspect_min_lines": 32,
}
VIDEO = {
    "channel_bw_hz": 16e6,
    "deviation_hz": 4.8e6,
    "width": 160,
    "auto_levels": True,
    "sample_rate": 20e6,
}


def _capture(name: str, fs: float) -> np.ndarray:
    path = CAPS / name
    if not path.exists():
        pytest.skip(f"optional IQ fixture is missing: {name}")
    iq = np.fromfile(path, dtype=np.complex64)
    return iq[-int(fs * 0.080):]


def _evaluate(name: str, fs: float):
    iq = _capture(name, fs)
    proposals, rf = sweep_inspect.propose_iq(
        iq[:SCAN["fft_size"] * SCAN["averages"]],
        fs,
        metadata_center_hz=1.1e9,
        scan=SCAN,
    )
    assert proposals
    proposal = proposals[0]
    evidence = sweep_inspect.inspect_iq(
        iq,
        fs,
        signal_offset_hz=proposal.center_hz - 1.1e9,
        bandwidth_hz=proposal.bandwidth_hz,
        scan={**SCAN, "inspect_ms": 56},
        video=VIDEO,
    )
    return rf, proposal, evidence


def test_same_baseband_arbitrary_rf_metadata_has_same_decision() -> None:
    iq = _capture("iq_20260912T112200.005630Z_5110.000M.cf32", 20e6)
    dwell = iq[:SCAN["fft_size"] * SCAN["averages"]]
    low, _ = sweep_inspect.propose_iq(
        dwell, 20e6, metadata_center_hz=1.1e9, scan=SCAN,
    )
    high, _ = sweep_inspect.propose_iq(
        dwell, 20e6, metadata_center_hz=5.1e9, scan=SCAN,
    )
    assert bool(low) is bool(high)
    assert len(low) == len(high)
    assert [
        round(item.center_hz - 1.1e9) for item in low
    ] == [
        round(item.center_hz - 5.1e9) for item in high
    ]


def test_dense_followup_is_not_gated_by_named_frequency_bands() -> None:
    scan = {
        **SCAN,
        "start_hz": 0.4e9,
        "stop_hz": 6.0e9,
        "sample_rate": 35e6,
        "channel_bw_hz": 16e6,
        "step_hz": 12e6,
        "cluster_step_mhz": "4",
    }
    first = scan_view.extras_for_hit(4.1e9, 4.107e9, scan)
    second = scan_view.extras_for_hit(1.1e9, 1.107e9, scan)
    assert first and second
    assert len(first) == len(second)


@pytest.mark.parametrize(
    "name",
    [
        "iq_20260912T112200.005630Z_5110.000M.cf32",
        "iq_20260912T112309.614473Z_5493.000M.cf32",
        "iq_20260912T112612.044012Z_3331.000M.cf32",
    ],
)
def test_user_picture_fixtures_are_proposed_and_confirmed(name: str) -> None:
    _, _, evidence = _evaluate(name, 20e6)
    assert evidence.analog_evidence is True
    assert evidence.video_confirmed is True
    assert evidence.stage == "video_confirmed"


def test_decoder_miss_hands_off_stable_analog_without_picture_claim() -> None:
    _, _, evidence = _evaluate(
        "iq_20260912T090201.781776Z_1321.000M.cf32", 20e6,
    )
    assert evidence.analog_evidence is True
    assert evidence.video_confirmed is False
    assert evidence.stage == "analog_evidence"
    assert evidence.votes >= 2
    assert evidence.selection.confirmed is True


def test_weak_fixture_is_not_handed_to_lock() -> None:
    _, _, evidence = _evaluate(
        "iq_20260911T134729.138839Z_1320.000M.cf32", 20e6,
    )
    assert evidence.analog_evidence is False
    assert evidence.video_confirmed is False
    assert evidence.rejected_reason


def test_broad_flat_fm_region_is_proposed_without_center_peak() -> None:
    nfft = 4096
    fs = 20e6
    psd = np.full(nfft, -55.0)
    lo, hi = 900, 3200
    psd[lo:hi] = -48.0
    psd[lo + 120] = -34.0
    found = spectrum.find_occupied(
        psd, 0.0, fs, threshold_db=2.0, min_bw_hz=0.6e6,
    )
    assert found
    assert found[0].bandwidth_hz > 8e6
    assert abs(found[0].center_hz) < abs(found[0].peak_hz)


def test_dc_and_short_band_edge_spurs_are_rejected() -> None:
    nfft = 4096
    fs = 20e6
    psd = np.full(nfft, -60.0)
    bin_hz = fs / nfft
    dc = nfft // 2
    psd[dc - int(80e3 / bin_hz):dc + int(80e3 / bin_hz)] = -20.0
    guard = int(nfft * (1 - 0.95) / 2)
    psd[guard:guard + int(0.7e6 / bin_hz)] = -30.0
    found = spectrum.find_occupied(
        psd, 0.0, fs, threshold_db=3.0, min_bw_hz=0.6e6,
        dc_notch_hz=200e3,
    )
    assert found == []


def _selection(**kwargs) -> adaptive_if.Selection:
    values = dict(
        decimation=1,
        effective_bw_hz=8e6,
        cutoff_hz=4e6,
        offset_hz=0.0,
        score=0.4,
        line_rate_hz=LINE_PAL,
        line_prominence_db=4.0,
        row_corr=0.05,
        analog=False,
        reason="test",
        generation=1,
        candidates_evaluated=2,
        confirmed=False,
        stable=True,
        windows_evaluated=2,
        winner_votes=2,
        spectral_retention=0.5,
        line_confidence=0.4,
        sync_score=0.0,
        vertical_detail=2.0,
    )
    values.update(kwargs)
    return adaptive_if.Selection(**values)


class _FakeEval:
    def __init__(self, selection: adaptive_if.Selection) -> None:
        self.selection = selection
        self.base = np.zeros(256, dtype=np.float32)
        self.fm_base = self.base
        self.fs_ch = 8e6
        self.line_hint = cvbs.LineRateHint(
            line_hz=selection.line_rate_hz, attempted=True,
        )


def test_weak_pal_comb_is_analog_evidence_without_video(monkeypatch) -> None:
    monkeypatch.setattr(
        sweep_inspect.adaptive_if,
        "select",
        lambda *args, **kwargs: _FakeEval(_selection(line_prominence_db=3.5)),
    )
    evidence = sweep_inspect.inspect_iq(
        np.zeros(2048, dtype=np.complex64),
        20e6,
        bandwidth_hz=8e6,
        scan={
            **SCAN,
            "line_prominence_db": 2.0,
            "inspect_stable_prominence_db": scan_gate.STABLE_ANALOG_PROMINENCE_DB,
        },
        video=VIDEO,
        decode=False,
    )
    assert evidence.analog_evidence is True
    assert evidence.video_confirmed is False
    assert evidence.stage == "analog_evidence"
    assert evidence.rejected_reason == "decoder preview not confirmed"
    assert evidence.standard == "PAL"
    assert evidence.selection.confirmed is True
    assert evidence.selection.analog is True


def test_junk_occupancy_is_not_analog_evidence(monkeypatch) -> None:
    monkeypatch.setattr(
        sweep_inspect.adaptive_if,
        "select",
        lambda *args, **kwargs: _FakeEval(_selection(
            line_rate_hz=0.0,
            line_prominence_db=20.0,
            analog=False,
            confirmed=False,
        )),
    )
    evidence = sweep_inspect.inspect_iq(
        np.zeros(2048, dtype=np.complex64),
        20e6,
        bandwidth_hz=8e6,
        scan=SCAN,
        video=VIDEO,
        decode=False,
    )
    assert evidence.analog_evidence is False
    assert evidence.video_confirmed is False
    assert evidence.rejected_reason == "no PAL/NTSC line rate"


def test_inspect_capture_honors_yaml_window() -> None:
    assert sweep_inspect.MAX_INSPECT_SECONDS == pytest.approx(0.12)
    assert sweep_inspect.inspect_capture_seconds({"inspect_ms": 90}) == pytest.approx(0.09)
    assert sweep_inspect.inspect_capture_seconds({"inspect_ms": 110}) == pytest.approx(0.11)
    assert sweep_inspect.inspect_capture_seconds({"inspect_ms": 200}) == pytest.approx(0.12)
    assert sweep_inspect.analog_evidence_prominence_db({}) == scan_gate.LINE_PROMINENCE_DB
    assert sweep_inspect.analog_evidence_prominence_db({
        "line_prominence_db": 2.0,
        "inspect_stable_prominence_db": scan_gate.STABLE_ANALOG_PROMINENCE_DB,
    }) == pytest.approx(2.0)


def test_inspect_work_and_preview_are_bounded() -> None:
    rf, _, evidence = _evaluate(
        "iq_20260912T075948.808230Z_4010.000M.cf32", 20e6,
    )
    assert rf.proposals_kept <= sweep_inspect.MAX_RF_PROPOSALS
    assert evidence.evaluations <= (
        sweep_inspect.MAX_INSPECT_CANDIDATES
        * sweep_inspect.MAX_INSPECT_WINDOWS
    )
    assert evidence.preview_samples <= int(
        20e6 * sweep_inspect.MAX_INSPECT_SECONDS,
    )


def test_stable_analog_can_auto_peek_without_picture_score() -> None:
    det = type("D", (), {
        "confidence": 0.4,
        "pic_score": 0.0,
        "pic_locked": False,
        "analog_evidence": True,
        "inspect_votes": 2,
        "prominence_db": 12.0,
    })()
    assert scan_gate.auto_peek_allowed(det, {
        "auto_peek": True,
        "auto_peek_min_conf": 0.6,
        "auto_peek_min_pic": 0.12,
    })
    weak = type("D", (), {
        "confidence": 0.4,
        "pic_score": 0.0,
        "pic_locked": False,
        "analog_evidence": True,
        "inspect_votes": 2,
        "prominence_db": 3.0,
    })()
    peek_scan = {
        "auto_peek": True,
        "auto_peek_min_conf": 0.6,
        "auto_peek_min_pic": 0.12,
    }
    assert scan_gate.auto_peek_allowed(weak, peek_scan) is False
    weak.prominence_db = scan_gate.STABLE_ANALOG_PROMINENCE_DB
    assert scan_gate.auto_peek_allowed(weak, peek_scan) is True


class _Source:
    name = "stub"
    sample_rate = 20e6
    overflows = 0
    clip_frac = 0.0
    adc_rms = 0.0
    bias_tee = False


def test_valid_sweep_selection_is_adopted_without_reader_restart(
        monkeypatch) -> None:
    cfg = {
        "scan": {"start_hz": 1e9, "stop_hz": 1.2e9, "sample_rate": 20e6},
        "video": VIDEO,
        "rotator": {"enable": False},
        "sdr": {},
    }
    engine = Engine(_Source(), cfg, Queue())
    selection = adaptive_if.Selection(
        decimation=2,
        effective_bw_hz=9e6,
        cutoff_hz=4.5e6,
        offset_hz=0.0,
        score=0.5,
        line_rate_hz=15_625.0,
        line_prominence_db=12.0,
        row_corr=0.03,
        analog=True,
        reason="stable sweep analogue",
        generation=1,
        candidates_evaluated=4,
        confirmed=True,
        stable=True,
        windows_evaluated=2,
        winner_votes=2,
    )
    target = 1.1e9
    shared_fs = engine._rate_plan.hardware_sample_rate_hz
    engine._sweep_handoffs[round(target / 50e3)] = (
        selection, shared_fs, engine._sweep_generation,
    )
    monkeypatch.setattr(
        engine,
        "_stop_reader",
        lambda: pytest.fail("DSP handoff must not restart hardware"),
    )
    assert engine._adopt_sweep_handoff(target) is True
    assert engine._adaptive_if.selection is not None
    assert engine._adaptive_if.selection.decimation == 2
