from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from queue import Queue

import numpy as np
import pytest

from fpvscan.dsp import adaptive_if
from fpvscan.dsp import cvbs, demod
from fpvscan.engine import Engine
from fpvscan import scan_view


CAPTURE = (
    Path(__file__).resolve().parents[1]
    / "caps"
    / "iq_20260912T075948.808230Z_4010.000M.cf32"
)
UNCONFIRMED_CAPTURE = (
    Path(__file__).resolve().parents[1]
    / "caps"
    / "iq_20260912T090201.781776Z_1321.000M.cf32"
)
NEW_CAPTURE_CASES = [
    pytest.param(
        "iq_20260912T092833.936010Z_4869.000M.cf32",
        "positive-decoder-miss",
        1.1e9,
        True,
        id="positive-miss-offset-content",
    ),
    pytest.param(
        "iq_20260912T094557.043870Z_3429.000M.cf32",
        "positive-decoder-miss",
        5.1e9,
        True,
        id="positive-miss-noisy-content",
    ),
    pytest.param(
        "iq_20260912T095610.131565Z_5473.200M.cf32",
        "decoded-positive-torn-geometry",
        2.2e9,
        False,
        id="decoded-positive-torn-content",
    ),
    pytest.param(
        "iq_20260912T100123.084728Z_3429.800M.cf32",
        "positive-decoder-miss",
        4.4e9,
        True,
        id="positive-miss-slanted-content",
    ),
]


def _fixture_iq(path: Path = CAPTURE) -> np.ndarray:
    assert path.exists(), f"adaptive-IF fixture is missing: {path.name}"
    iq = np.fromfile(path, dtype=np.complex64)
    return (iq - np.mean(iq)).astype(np.complex64)


def _post_deploy_replay(timestamp_prefix: str):
    matches = list(CAPTURE.parent.glob(f"iq_{timestamp_prefix}*.cf32"))
    assert len(matches) == 1, f"expected one fixture for {timestamp_prefix}"
    fs = 20e6
    iq = np.fromfile(matches[0], dtype=np.complex64)[-int(fs * 0.056):]
    iq = (iq - np.mean(iq)).astype(np.complex64)
    nudge = scan_view.prelock_mix_hz(
        demod.blob_offset_hz(iq, fs),
        tracking=False,
        fs=fs,
        ch_bw=16e6,
        off_hz=0.0,
        max_hz=2.5e6,
    )
    result = adaptive_if.select(
        iq,
        fs,
        channel_cap_hz=16e6,
        deviation_hz=4.8e6,
        width=160,
        proposed_nudge_hz=nudge,
    )
    frame = cvbs.decode(
        result.base,
        result.fs_ch,
        width=160,
        state=cvbs.DecodeState(),
        line_hint=result.line_hint,
    )
    return iq, result, frame, cvbs.score_picture(frame)


def _periodic_base(fs: float, line_hz: float, seconds: float = 0.056):
    period = int(round(fs / line_hz))
    line = np.full(period, 0.70, dtype=np.float32)
    line[:max(3, period // 16)] = 0.0
    line[period // 3:period // 2] = 0.45
    count = int(np.ceil(fs * seconds / period))
    return np.tile(line, count)[:int(fs * seconds)]


def _score(value: float, *, prom: float = 20.0, analog: bool = True,
           retention: float = 0.8
           ) -> adaptive_if.CandidateScore:
    return adaptive_if.CandidateScore(
        score=value,
        line_rate_hz=15625.0,
        line_prominence_db=prom,
        line_confidence=0.8,
        row_corr=0.2,
        sync_score=0.4,
        spectral_retention=retention,
        clip_penalty=0.0,
        noise_penalty=0.0,
        analog=analog,
    )


def test_candidates_depend_on_sample_rate_not_rf_center() -> None:
    a = adaptive_if.generate_candidates(20e6, channel_cap_hz=16e6)
    b = adaptive_if.generate_candidates(35e6, channel_cap_hz=16e6)
    assert [c.decimation for c in a] == [1, 2, 3]
    assert [c.output_rate_hz for c in a] == [20e6, 10e6, 20e6 / 3]
    assert [c.output_rate_hz for c in b] != [c.output_rate_hz for c in a]
    assert all(
        c.effective_bw_hz == c.output_rate_hz * adaptive_if.FILTER_USABLE_FRAC
        for c in a + b
    )


def test_same_baseband_at_arbitrary_rf_centers_selects_same_path() -> None:
    iq = _fixture_iq()

    def select_at_metadata_center(_center_hz: float):
        # RF center is metadata only and intentionally is not an API argument.
        return adaptive_if.select(
            iq, 20e6, channel_cap_hz=16e6, deviation_hz=4.8e6,
        ).selection

    low = select_at_metadata_center(1.1e9)
    high = select_at_metadata_center(5.1e9)
    assert (low.decimation, low.effective_bw_hz, low.offset_hz) == (
        high.decimation, high.effective_bw_hz, high.offset_hz,
    )


def test_confirmed_pal_hint_anchors_decode_and_published_rate() -> None:
    fs = 2e6
    base = _periodic_base(fs, 16_400.0)
    hint = cvbs.LineRateHint(
        line_hz=15_625.0, attempted=True, confirmed=True,
    )
    state = cvbs.DecodeState(
        period=fs / 16_400.0, standard="NTSC", abs_t0=0.0, sticky=True,
    )

    frame = cvbs.decode(
        base, fs, width=80, state=state, line_hint=hint,
    )

    assert frame is not None
    assert frame.standard == "PAL"
    assert frame.line_rate == pytest.approx(15_625.0)
    assert hint.line_hz == pytest.approx(15_625.0)
    assert state.standard == "PAL"
    assert fs / float(state.period) == pytest.approx(15_625.0)


def test_confirmed_ntsc_hint_keeps_ntsc_fallback_geometry() -> None:
    fs = 2e6
    base = _periodic_base(fs, demod.LINE_NTSC)
    hint = cvbs.LineRateHint(
        line_hz=demod.LINE_NTSC, attempted=True, confirmed=True,
    )

    frame = cvbs.free_run(base, fs, width=80, line_hint=hint)

    assert frame is not None
    assert frame.standard == "NTSC"
    assert frame.line_rate == pytest.approx(demod.LINE_NTSC)


def test_unconfirmed_hint_runs_one_bounded_blind_search(monkeypatch) -> None:
    calls = {"n": 0}

    def estimate(_base, _fs, *, stats=None):
        calls["n"] += 1
        if stats is not None:
            stats.update(input_samples=2048, nfft=4096)
        return demod.LINE_NTSC

    monkeypatch.setattr(cvbs, "estimate_line_hz", estimate)
    hint = cvbs.LineRateHint()
    base = _periodic_base(2e6, demod.LINE_NTSC)

    assert hint.resolve(base, 2e6) == pytest.approx(demod.LINE_NTSC)
    assert hint.resolve(base, 2e6) == pytest.approx(demod.LINE_NTSC)
    assert calls["n"] == 1
    assert hint.input_samples <= cvbs.LINE_HUNT_MAX_INPUT_SAMPLES
    assert hint.nfft <= cvbs.LINE_HUNT_MAX_NFFT


def test_positive_decoder_miss_selects_decimated_path_and_improves() -> None:
    iq = _fixture_iq()
    result = adaptive_if.select(
        iq, 20e6,
        channel_cap_hz=16e6,
        deviation_hz=4.8e6,
        proposed_nudge_hz=189e3,
    )
    baseline_candidate = adaptive_if.generate_candidates(
        20e6, channel_cap_hz=16e6,
    )[0]
    preview = iq[-int(20e6 * adaptive_if.PREVIEW_SECONDS):]
    base, _, fs_ch = adaptive_if._demodulate(
        preview, 20e6, 0.0, baseline_candidate, 4.8e6,
    )
    baseline, _ = adaptive_if._candidate_score(
        base, fs_ch, input_clip_frac=0.0,
        spectral_retention=adaptive_if._spectral_retention(
            adaptive_if._occupied_spectrum(preview, 20e6),
            mix_hz=0.0,
            cutoff_hz=baseline_candidate.cutoff_hz,
        ),
        width=160,
    )

    selected = result.selection
    assert selected.decimation == 2
    assert selected.effective_bw_hz < baseline_candidate.effective_bw_hz
    assert selected.score >= baseline.score + 0.15
    assert selected.line_prominence_db >= baseline.line_prominence_db + 5.0
    assert selected.row_corr >= baseline.row_corr + 0.05
    assert selected.offset_hz == 0.0
    assert selected.confirmed is True
    assert selected.stable is True
    assert selected.windows_evaluated == 3
    assert selected.winner_votes >= 2
    assert result.line_hint.attempted is True
    assert result.line_hint.confirmed is True
    assert 14_000.0 < float(result.line_hint.line_hz or 0.0) < 17_500.0
    assert result.input_offset_samples == 0
    assert len(result.base) > int(
        result.fs_ch * adaptive_if.PREVIEW_SECONDS,
    )

    full_rows = []
    for candidate in (
        baseline_candidate,
        adaptive_if.generate_candidates(20e6, channel_cap_hz=16e6)[1],
    ):
        full, _, full_fs = adaptive_if._demodulate(
            iq, 20e6, 0.0, candidate, 4.8e6,
        )
        comb = demod.classify_video(full, full_fs, min_prominence_db=0.0)
        shared = cvbs.LineRateHint(line_hz=comb.line_rate, attempted=True)
        raster = cvbs.free_run(full, full_fs, width=160, line_hint=shared)
        full_rows.append(cvbs.score_picture(raster).row_corr)
    assert full_rows[1] >= full_rows[0] + 0.02


def test_second_fixture_is_conservative_and_not_analog() -> None:
    iq = _fixture_iq(UNCONFIRMED_CAPTURE)
    result = adaptive_if.select(
        iq, 20e6, channel_cap_hz=16e6, deviation_hz=4.8e6,
    )
    selected = result.selection
    assert selected.decimation == 1
    assert selected.analog is False
    assert selected.confirmed is False
    assert "unconfirmed" in selected.reason or "unstable" in selected.reason
    assert selected.windows_evaluated == 3

    dec3 = adaptive_if.generate_candidates(
        20e6, channel_cap_hz=16e6,
    )[2]
    retention = [
        adaptive_if._spectral_retention(
            adaptive_if._occupied_spectrum(window, 20e6),
            mix_hz=0.0,
            cutoff_hz=dec3.cutoff_hz,
        )
        for _, window in adaptive_if._preview_windows(iq, 20e6)
    ]
    assert float(np.median(retention)) < adaptive_if.MIN_SPECTRAL_RETENTION


@pytest.mark.parametrize(
    ("filename", "content_category", "metadata_center_hz", "expect_decimated"),
    NEW_CAPTURE_CASES,
)
def test_new_positive_fixtures_choose_stable_runtime_path(
        filename: str, content_category: str,
        metadata_center_hz: float, expect_decimated: bool) -> None:
    path = CAPTURE.parent / filename
    if not path.exists():
        pytest.skip(f"optional adaptive-IF fixture is missing: {path.name}")
    iq = _fixture_iq(path)
    # Match the configured first-lock capture.  RF center is deliberately
    # arbitrary metadata and is not passed to the signal-only selector.
    iq = iq[-int(40e6 * 0.056):]
    assert metadata_center_hz in (1.1e9, 2.2e9, 4.4e9, 5.1e9)
    assert content_category in (
        "positive-decoder-miss", "decoded-positive-torn-geometry",
    )

    result = adaptive_if.select(
        iq, 40e6, channel_cap_hz=20.465e6, deviation_hz=4.8e6,
        proposed_nudge_hz=demod.blob_offset_hz(iq, 40e6),
    )
    selected = result.selection
    if expect_decimated:
        assert selected.decimation > 1
    assert selected.confirmed is True
    assert selected.stable is True
    assert selected.winner_votes == selected.windows_evaluated
    assert selected.spectral_retention >= adaptive_if.MIN_SPECTRAL_RETENTION
    assert selected.offset_hz == 0.0
    # Strong centered evidence avoids two redundant offset evaluations.
    assert selected.candidates_evaluated == (
        len(adaptive_if.generate_candidates(40e6, channel_cap_hz=20.465e6))
        * selected.windows_evaluated
    )


@pytest.mark.parametrize(
    "timestamp_prefix",
    [
        "20260912T112200",
        "20260912T112309",
        "20260912T112612",
    ],
)
def test_post_deploy_operator_picture_fixtures_decode_directionally(
        timestamp_prefix: str) -> None:
    iq, result, frame, picture = _post_deploy_replay(timestamp_prefix)

    assert result.input_offset_samples == 0
    assert len(result.base) > int(
        result.fs_ch * adaptive_if.PREVIEW_SECONDS,
    )
    assert len(iq) == int(20e6 * 0.056)
    assert frame is not None
    assert frame.standard == "PAL"
    assert 15_400.0 < frame.line_rate < 15_850.0
    assert frame.lines >= 160
    assert picture.row_corr >= 0.30
    assert picture.value >= 0.60


def test_post_deploy_rabbit_only_fixture_remains_decoder_miss() -> None:
    _, result, frame, picture = _post_deploy_replay("20260912T113110")

    assert result.selection.confirmed is False
    assert frame is not None
    assert picture.row_corr < 0.10
    assert picture.value < 0.30
    assert not cvbs.analog_usable(frame, pic=picture)


def test_strong_line_comb_without_vertical_picture_is_not_confirmed() -> None:
    fs = 2e6
    deviation = 0.48e6
    t = np.arange(int(fs * 0.08), dtype=np.float64) / fs
    # A periodic two-harmonic comb has excellent line prominence and
    # near-perfect repeated rows, but no vertical picture content.
    inst = (
        0.35 * np.sin(2 * np.pi * 15_625.0 * t)
        + 0.12 * np.sin(2 * np.pi * 31_250.0 * t)
    )
    phase = np.cumsum(inst) * (2 * np.pi * deviation / fs)
    iq = np.exp(1j * phase).astype(np.complex64)

    selected = adaptive_if.select(
        iq, fs, channel_cap_hz=1.6e6, deviation_hz=deviation,
    ).selection
    assert selected.line_prominence_db > 40.0
    assert selected.row_corr > 0.9
    assert selected.confirmed is False
    assert selected.analog is False
    assert selected.decimation == 1


def test_lock_analog_confirm_floor_stays_twelve_db() -> None:
    assert adaptive_if.ANALOG_MEDIAN_PROMINENCE_DB == 12.0


def test_noise_does_not_claim_analog_and_work_is_bounded(monkeypatch) -> None:
    from fpvscan.dsp import cvbs

    calls = {"estimator": 0}

    def counted(*_args, **_kwargs):
        calls["estimator"] += 1
        return None

    monkeypatch.setattr(cvbs, "estimate_line_hz", counted)
    rng = np.random.default_rng(20260912)
    noise = (
        rng.normal(size=400_000) + 1j * rng.normal(size=400_000)
    ).astype(np.complex64)
    result = adaptive_if.select(noise, 20e6, proposed_nudge_hz=125e3)
    assert result.selection.analog is False
    assert result.selection.candidates_evaluated <= (
        adaptive_if.MAX_CANDIDATES * adaptive_if.MAX_WINDOWS
        + adaptive_if.MIN_CONFIRM_WINDOWS
    )
    # The comb result is shared with preview raster scoring.
    assert calls["estimator"] == 0


def test_lifecycle_reuses_cache_and_loss_rearms_once() -> None:
    lifecycle = adaptive_if.Lifecycle(loss_frames=3)
    selection = adaptive_if.Selection(
        decimation=2,
        effective_bw_hz=9e6,
        cutoff_hz=4.5e6,
        offset_hz=0.0,
        score=0.7,
        line_rate_hz=15625.0,
        line_prominence_db=24.0,
        row_corr=0.2,
        analog=True,
        reason="test",
        generation=1,
        candidates_evaluated=3,
    )
    lifecycle.adopt(selection, 20e6)
    assert lifecycle.needs_evaluation(20e6) is False
    assert lifecycle.observe(True) is False
    assert lifecycle.needs_evaluation(20e6) is False
    assert lifecycle.observe(False) is False
    assert lifecycle.observe(False) is False
    assert lifecycle.observe(False) is True
    assert lifecycle.needs_evaluation(20e6) is True
    assert lifecycle.reason == "3 consecutive unusable frames"


def test_keep_cached_selection_on_free_run_line_rate() -> None:
    lifecycle = adaptive_if.Lifecycle(loss_frames=8)
    selection = adaptive_if.Selection(
        decimation=2,
        effective_bw_hz=9e6,
        cutoff_hz=4.5e6,
        offset_hz=0.0,
        score=0.4,
        line_rate_hz=15625.0,
        line_prominence_db=8.0,
        row_corr=0.04,
        analog=True,
        reason="test",
        generation=1,
        candidates_evaluated=2,
    )
    lifecycle.adopt(selection, 20e6)
    assert adaptive_if.keep_cached_selection(
        line_rate_hz=15625.0, assembler_period=None,
    ) is True
    assert adaptive_if.keep_cached_selection(
        assembler_period=None, line_rate_hz=None,
    ) is False
    for _ in range(16):
        keep = adaptive_if.keep_cached_selection(
            line_rate_hz=15734.0, assembler_period=None,
        )
        assert keep is True
        assert lifecycle.observe(True) is False
    assert lifecycle.needs_evaluation(20e6) is False


def test_offset_requires_material_hysteresis_improvement() -> None:
    baseline = _score(0.50)
    assert not adaptive_if.offset_improves(baseline, _score(0.55))
    assert not adaptive_if.offset_improves(
        baseline, _score(0.70, prom=25.0, analog=False),
    )
    assert adaptive_if.offset_improves(baseline, _score(0.63, prom=20.0))
    assert not adaptive_if.offset_improves(
        baseline, replace(_score(0.70), line_prominence_db=18.9),
    )


def test_low_retention_penalty_requires_robust_analog_evidence() -> None:
    weak = _score(0.50, analog=False, retention=0.21)
    strong = _score(0.50, analog=True, retention=0.21)
    assert adaptive_if._guarded_score(weak) == 0.25
    assert adaptive_if._guarded_score(strong) == 0.50


class _Source:
    name = "stub"
    sample_rate = 20e6
    overflows = 0
    clip_frac = 0.0
    adc_rms = 0.0
    bias_tee = False


def test_cached_dsp_selection_does_not_restart_reader(monkeypatch) -> None:
    cfg = {
        "scan": {
            "start_hz": 400e6, "stop_hz": 6000e6, "sample_rate": 20e6,
            "channel_bw_hz": 16e6, "step_hz": 12e6, "fft_size": 2048,
            "priority_bands": False,
        },
        "video": {
            "sample_rate": 20e6, "channel_bw_hz": 16e6,
            "deviation_hz": 4.8e6, "lo_offset_hz": 0.0, "width": 160,
        },
        "rotator": {"enable": False},
        "sdr": {},
    }
    engine = Engine(_Source(), cfg, Queue())
    iq = np.ones(4096, dtype=np.complex64)
    selection = adaptive_if.Selection(
        decimation=2, effective_bw_hz=9e6, cutoff_hz=4.5e6,
        offset_hz=0.0, score=0.7, line_rate_hz=15625.0,
        line_prominence_db=24.0, row_corr=0.2, analog=True,
        reason="cached", generation=1, candidates_evaluated=3,
        confirmed=True, stable=True, windows_evaluated=2, winner_votes=2,
    )
    engine._adaptive_if.adopt(selection, 20e6)
    stopped = {"n": 0}

    def forbidden_stop():
        stopped["n"] += 1

    monkeypatch.setattr(engine, "_stop_reader", forbidden_stop)
    monkeypatch.setattr(
        adaptive_if, "select",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("cached sticky path must not rescore candidates")
        ),
    )
    _, _, _, _, _, hint, _ = engine._adaptive_demod(
        iq, 20e6, 0.0, cfg["video"], 4.8e6,
    )
    assert stopped["n"] == 0
    assert engine._adaptive_if.evaluations == 1
    assert hint is not None
    assert hint.confirmed is True
    assert hint.line_hz == pytest.approx(15625.0)
    diag = engine.snapshot()["adaptive_if"]
    assert diag["decimation"] == 2
    assert diag["effective_bw_hz"] == 9e6
    assert diag["generation"] == 1
