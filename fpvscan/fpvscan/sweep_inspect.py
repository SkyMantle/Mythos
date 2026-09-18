"""Frequency-agnostic, bounded SWEEP proposal and analogue inspection.

The functions in this module are shared by the live engine and offline IQ
replay.  RF metadata is used only to translate baseband offsets into display
coordinates; all acceptance decisions are made from captured signal content.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import time
from typing import Any

import numpy as np

from . import scan_gate
from .dsp import adaptive_if, cvbs, demod, spectrum

MAX_RF_PROPOSALS = 3
MAX_INSPECT_CANDIDATES = 2
MAX_INSPECT_WINDOWS = 2
# Honor scan.inspect_ms 90/110; do not clamp a 90 ms grab to 56 ms.
MAX_INSPECT_SECONDS = 0.12
# Auto-peek still uses scan_gate.STABLE_ANALOG_PROMINENCE_DB (8 dB).
# analog_evidence uses scan.line_prominence_db (2 dB) for distant FM.
STABLE_LINE_CONFIDENCE = 0.20
STABLE_VERTICAL_DETAIL = 1.0
VIDEO_CONFIRM_ROW_CORR = 0.28
VIDEO_CONFIRM_SCORE = 0.20


@dataclass(frozen=True)
class RFDiagnostics:
    noise_floor_db: float
    threshold_offset_db: float
    proposals_seen: int
    proposals_kept: int
    elapsed_ms: float


@dataclass(frozen=True)
class InspectResult:
    rf_candidate: bool
    analog_evidence: bool
    video_confirmed: bool
    stage: str
    rejected_reason: str
    standard: str
    line_rate_hz: float
    prominence_db: float
    line_confidence: float
    row_corr: float
    picture_score: float
    picture_lines: int
    picture_locked: bool
    votes: int
    windows: int
    retention: float
    vertical_detail: float
    elapsed_ms: float
    preview_samples: int
    evaluations: int
    center_offset_hz: float
    selection: adaptive_if.Selection

    def diagnostics(self) -> dict[str, Any]:
        body = asdict(self)
        body["selection"] = self.selection.diagnostics()
        return body


def _proposal_rank(item: spectrum.Occupancy, fs: float) -> tuple[float, float]:
    """Prefer broad occupied energy while retaining a deterministic order."""
    broad = min(1.0, float(item.bandwidth_hz) / max(float(fs) * 0.35, 1.0))
    centered = 1.0 - min(1.0, abs(float(item.center_hz)) / max(float(fs) / 2.0, 1.0))
    score = float(item.snr_db) + 2.0 * broad + 0.25 * centered
    return score, float(item.bandwidth_hz)


def limit_proposals(
    proposals: list[spectrum.Occupancy],
    sample_rate_hz: float,
    *,
    metadata_center_hz: float = 0.0,
    scan: dict[str, Any] | None = None,
) -> list[spectrum.Occupancy]:
    """Rank and cap expensive inspect work, independent of absolute RF."""
    sc = scan or {}
    fs = float(sample_rate_hz)
    baseband = [
        replace(item, center_hz=item.center_hz - float(metadata_center_hz))
        for item in proposals
    ]
    limit = max(1, min(
        int(sc.get("sweep_max_candidates", MAX_RF_PROPOSALS)),
        MAX_RF_PROPOSALS,
    ))
    order = sorted(
        range(len(proposals)),
        key=lambda index: _proposal_rank(baseband[index], fs),
        reverse=True,
    )
    return [proposals[index] for index in order[:limit]]


def analog_evidence_prominence_db(scan: dict[str, Any] | None = None) -> float:
    """Distant analogue comb floor from YAML, not the 8 dB auto-peek gate."""
    sc = scan or {}
    return float(sc.get("line_prominence_db", scan_gate.LINE_PROMINENCE_DB))


def inspect_capture_seconds(scan: dict[str, Any] | None = None) -> float:
    """IQ window for inspect; honors scan.inspect_ms up to MAX_INSPECT_SECONDS."""
    sc = scan or {}
    raw_ms = sc.get("inspect_ms")
    if raw_ms is not None:
        try:
            seconds = float(raw_ms) / 1000.0
        except (TypeError, ValueError):
            seconds = MAX_INSPECT_SECONDS
    else:
        seconds = MAX_INSPECT_SECONDS
    return max(0.020, min(MAX_INSPECT_SECONDS, seconds))


def propose_iq(
    iq: np.ndarray,
    sample_rate_hz: float,
    *,
    metadata_center_hz: float = 0.0,
    scan: dict[str, Any] | None = None,
) -> tuple[list[spectrum.Occupancy], RFDiagnostics]:
    """Run the live RF proposal stage on one dwell capture."""
    started = time.perf_counter()
    sc = scan or {}
    fs = float(sample_rate_hz)
    nfft = max(256, int(sc.get("fft_size", 8192)))
    averages = max(1, int(sc.get("averages", 16)))
    need = min(len(iq), nfft * averages)
    if need < nfft:
        nfft = 1 << int(np.floor(np.log2(max(256, need))))
        averages = 1
    data = np.asarray(iq[:need], dtype=np.complex64)
    psd = spectrum.psd_db(data, nfft=nfft, averages=averages)
    edge = float(sc.get("edge_guard", 0.95))
    notch = float(sc.get("dc_notch_hz", 200e3))
    percentile = float(sc.get("noise_percentile", 25.0))
    view = spectrum.usable_view(
        psd, fs, edge_guard=edge, dc_notch_hz=notch,
        noise_percentile=percentile,
    )
    floor = spectrum.noise_floor_db(view, percentile=percentile)
    threshold = spectrum.occupancy_offset_db(view, sc)
    found = spectrum.find_occupied(
        psd, float(metadata_center_hz), fs,
        threshold_db=threshold,
        edge_guard=edge,
        min_bw_hz=scan_gate.fft_min_bw_hz(sc),
        dc_notch_hz=notch,
        noise_percentile=percentile,
    )
    # Rank in baseband coordinates so adding an arbitrary metadata centre
    # cannot change the decision.
    kept = limit_proposals(
        found,
        fs,
        metadata_center_hz=metadata_center_hz,
        scan=sc,
    )
    diag = RFDiagnostics(
        noise_floor_db=float(floor),
        threshold_offset_db=float(threshold),
        proposals_seen=len(found),
        proposals_kept=len(kept),
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
    )
    return kept, diag


def inspect_iq(
    iq: np.ndarray,
    sample_rate_hz: float,
    *,
    signal_offset_hz: float = 0.0,
    bandwidth_hz: float,
    scan: dict[str, Any] | None = None,
    video: dict[str, Any] | None = None,
    decode: bool = True,
) -> InspectResult:
    """Inspect one RF proposal with a fixed candidate/window budget.

    Stable repeated line-comb plus non-periodic vertical detail is sufficient
    for ``analog_evidence`` even when a short decoder preview cannot establish
    a picture.  ``video_confirmed`` remains a separate, stricter decoder fact.
    """
    started = time.perf_counter()
    sc = scan or {}
    vc = video or {}
    fs = float(sample_rate_hz)
    max_samples = max(256, int(fs * inspect_capture_seconds(sc)))
    data = np.asarray(iq[-max_samples:], dtype=np.complex64)
    data = (data - np.mean(data)).astype(np.complex64, copy=False)
    cap = adaptive_if.numeric_channel_cap(vc.get("channel_bw_hz"))
    if cap is None:
        cap = min(
            max(float(bandwidth_hz), 8e6),
            float(sc.get("inspect_bw_hz") or fs * 0.9),
            fs * 0.9,
        )
    deviation = float(vc.get("deviation_hz") or 0.0)
    if deviation <= 1e5:
        deviation = max(1e5, float(bandwidth_hz) / 4.0)
    result = adaptive_if.select(
        data,
        fs,
        base_mix_hz=float(signal_offset_hz),
        channel_cap_hz=cap,
        deviation_hz=deviation,
        width=min(320, int(vc.get("width", 640))),
        max_candidates=MAX_INSPECT_CANDIDATES,
        max_windows=MAX_INSPECT_WINDOWS,
    )
    selected = result.selection
    standard = demod.standard_from_line_rate(
        selected.line_rate_hz,
        max_err_hz=float(sc.get("line_tol_hz", 200.0)),
    )
    needed_votes = min(adaptive_if.MIN_CONFIRM_WINDOWS, selected.windows_evaluated)
    min_prom_db = analog_evidence_prominence_db(sc)
    stable_line = bool(
        selected.stable
        and selected.winner_votes >= needed_votes
        and standard in ("PAL", "NTSC")
        and selected.line_prominence_db >= min_prom_db
        and selected.line_confidence >= float(sc.get(
            "inspect_stable_line_confidence", STABLE_LINE_CONFIDENCE,
        ))
        and selected.vertical_detail >= float(sc.get(
            "inspect_stable_vertical_detail", STABLE_VERTICAL_DETAIL,
        ))
    )
    analog = bool(selected.confirmed or stable_line)

    picture = cvbs.score_picture(None)
    if analog and decode:
        hint = cvbs.LineRateHint(
            line_hz=selected.line_rate_hz,
            attempted=True,
            confirmed=True,
        )
        frame = cvbs.decode(
            result.base,
            result.fs_ch,
            width=min(320, int(vc.get("width", 640))),
            state=None,
            line_hint=hint,
            auto_levels=bool(vc.get("auto_levels", True)),
            sharpen=0.0,
        )
        picture = cvbs.score_picture(frame)
    else:
        frame = None
    confirmed = bool(
        frame is not None
        and analog
        and standard in ("PAL", "NTSC")
        and picture.value >= float(sc.get(
            "inspect_video_score", VIDEO_CONFIRM_SCORE,
        ))
        and picture.is_analog(
            min_corr=max(
                float(sc.get("inspect_min_row_corr", 0.02)),
                float(sc.get("inspect_video_row_corr", VIDEO_CONFIRM_ROW_CORR)),
            ),
            require_lock=bool(sc.get("inspect_require_lock", False)),
            min_lines=int(sc.get("inspect_min_lines", 32)),
        )
    )
    if confirmed:
        stage = "video_confirmed"
        rejected = ""
    elif analog:
        stage = "analog_evidence"
        rejected = "decoder preview not confirmed"
    else:
        stage = "rejected"
        if standard not in ("PAL", "NTSC"):
            rejected = "no PAL/NTSC line rate"
        elif not selected.stable or selected.winner_votes < needed_votes:
            rejected = "line evidence not temporally stable"
        elif selected.line_prominence_db < min_prom_db:
            rejected = "line prominence below stable threshold"
        elif selected.line_confidence < float(sc.get(
                "inspect_stable_line_confidence", STABLE_LINE_CONFIDENCE)):
            rejected = "line confidence below stable threshold"
        else:
            rejected = "comb lacks vertical picture detail"

    handoff = selected
    if analog and not selected.confirmed:
        # Safe to cache only with the same sample rate.  The engine enforces
        # that condition before adoption; this merely records the trusted,
        # repeated line evidence and its selected path.
        handoff = replace(selected, analog=True, confirmed=True)
    return InspectResult(
        rf_candidate=True,
        analog_evidence=analog,
        video_confirmed=confirmed,
        stage=stage,
        rejected_reason=rejected,
        standard=standard,
        line_rate_hz=float(selected.line_rate_hz),
        prominence_db=float(selected.line_prominence_db),
        line_confidence=float(selected.line_confidence),
        row_corr=float(picture.row_corr),
        picture_score=float(picture.value),
        picture_lines=int(picture.lines),
        picture_locked=bool(picture.locked),
        votes=int(selected.winner_votes),
        windows=int(selected.windows_evaluated),
        retention=float(selected.spectral_retention),
        vertical_detail=float(selected.vertical_detail),
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
        preview_samples=len(data),
        evaluations=int(selected.candidates_evaluated),
        center_offset_hz=float(selected.offset_hz),
        selection=handoff,
    )
