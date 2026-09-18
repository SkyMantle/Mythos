"""Replay every paired IQ capture through production SWEEP functions.

Usage:
    py scripts/evaluate_sweep_iq.py
    py scripts/evaluate_sweep_iq.py --json report.json
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy.signal import resample_poly
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fpvscan import hardware_rate, scan_gate, sweep_inspect  # noqa: E402
from fpvscan.dsp import cvbs, demod, spectrum  # noqa: E402


# Labels are fixture provenance only.  They are never passed to detection.
LABELS = {
    "20260911T134729": "weak-negative",
    "20260912T075948": "decoder-miss-positive",
    "20260912T090201": "decoder-miss-positive",
    "20260912T092833": "decoder-miss-positive",
    "20260912T094557": "decoder-miss-positive",
    "20260912T095610": "decoded-positive",
    "20260912T100123": "decoder-miss-positive",
    "20260912T112200": "picture-positive",
    "20260912T112309": "picture-positive",
    "20260912T112612": "picture-positive",
    "20260912T113110": "decoder-miss-positive",
}


def _label(path: Path) -> str:
    return next(
        (label for stamp, label in LABELS.items() if stamp in path.name),
        "ambiguous",
    )


def _legacy_proposals(
    iq: np.ndarray,
    fs: float,
    center: float,
    scan: dict[str, Any],
) -> list[spectrum.Occupancy]:
    """Reproduce the former double-dilation/peak-centre RF stage."""
    nfft = int(scan.get("fft_size", 8192))
    averages = int(scan.get("averages", 16))
    psd = spectrum.psd_db(iq[:nfft * averages], nfft, averages)
    edge = float(scan.get("edge_guard", 0.95))
    notch = float(scan.get("dc_notch_hz", 200e3))
    percentile = float(scan.get("noise_percentile", 25.0))
    view = spectrum.usable_view(
        psd, fs, edge_guard=edge, dc_notch_hz=notch,
        noise_percentile=percentile,
    )
    floor = spectrum.noise_floor_db(view, percentile=percentile)
    threshold = spectrum.occupancy_offset_db(view, scan)
    mask = view > floor + threshold
    bin_hz = fs / nfft
    gap = max(1, int(1e6 / bin_hz))
    if gap > 1:
        kernel = np.ones(2 * gap + 1, dtype=bool)
        dilated = np.convolve(mask, kernel, mode="same") > 0
        mask = np.convolve(~dilated, kernel, mode="same") == 0
    guard = int(nfft * (1 - edge) / 2)
    out: list[spectrum.Occupancy] = []
    index = 0
    while index < len(mask):
        if not mask[index]:
            index += 1
            continue
        end = index
        while end < len(mask) and mask[end]:
            end += 1
        width = (end - index) * bin_hz
        if width >= scan_gate.fft_min_bw_hz(scan):
            segment = view[index:end]
            peak = index + int(np.argmax(segment))
            peak_hz = center + (guard + peak - nfft / 2) * bin_hz
            out.append(spectrum.Occupancy(
                center_hz=peak_hz,
                bandwidth_hz=width,
                peak_db=float(np.max(segment)),
                snr_db=float(np.max(segment) - floor),
                peak_hz=peak_hz,
            ))
        index = end
    return out


def _legacy_inspect(
    iq: np.ndarray,
    fs: float,
    center: float,
    proposal: spectrum.Occupancy,
    scan: dict[str, Any],
) -> dict[str, Any]:
    """Offline equivalent of the former single-comb + offset-decode gate."""
    offset = proposal.center_hz - center
    out_bw = min(
        max(proposal.bandwidth_hz, 8e6),
        scan_gate.inspect_bw_hz(scan, center),
        fs * 0.9,
    )
    channel, fs_ch = demod.channelize(iq, fs, offset, out_bw_hz=out_bw)
    base = demod.fm_demod(
        channel, fs_ch, deviation_hz=proposal.bandwidth_hz / 5,
    )
    score = demod.classify_video(
        base,
        fs_ch,
        tol_hz=float(scan.get("line_tol_hz", 200)),
        min_prominence_db=float(scan.get("line_prominence_db", 2)),
        min_conf=float(scan.get("min_confidence", 0.0)),
        min_harmonics=int(scan.get("min_harmonics", 0)),
        harm_db=float(scan.get("line_harm_db", 3)),
    )
    best = cvbs.score_picture(None)
    accepted = bool(score.is_video)
    if accepted and scan.get("inspect_decode", True):
        decoded = False
        for relative in (0.0, 1e6, -1e6, 2e6, -2e6, 4e6, -4e6):
            channel, fs_ch = demod.channelize(
                iq, fs, offset + relative, out_bw_hz=out_bw,
            )
            base = demod.fm_demod(
                channel,
                fs_ch,
                deviation_hz=max(proposal.bandwidth_hz, 8e6) / 5,
            )
            frame = cvbs.decode(
                demod.deemphasis(base, fs_ch),
                fs_ch,
                width=160,
                state=None,
                sharpen=0.0,
            )
            picture = cvbs.score_picture(frame)
            if picture.value > best.value:
                best = picture
            if frame is not None and picture.is_analog(
                min_corr=float(scan.get("inspect_min_row_corr", 0.02)),
                require_lock=bool(scan.get("inspect_require_lock", False)),
                min_lines=int(scan.get("inspect_min_lines", 32)),
            ):
                decoded = True
                best = picture
                break
        accepted = decoded or scan_gate.inspect_soft_ok(
            standard=score.standard,
            bandwidth_hz=proposal.bandwidth_hz,
            prominence_db=score.prominence_db,
            confidence=score.confidence,
            harmonics=score.harmonics,
            row_corr=best.row_corr,
            pic_lines=best.lines,
            scan=scan,
            center_hz=center,
        )
    auto_handoff = bool(
        accepted
        and score.confidence >= float(scan.get("auto_peek_min_conf", 0.6))
        and (
            best.value >= float(scan.get("auto_peek_min_pic", 0.12))
            or best.locked
        )
    )
    return {
        "accepted": accepted,
        "auto_handoff": auto_handoff,
        "standard": score.standard,
        "prominence_db": score.prominence_db,
        "row_corr": best.row_corr,
    }


def _resample_iq(
    iq: np.ndarray, source_rate_hz: float, target_rate_hz: float,
) -> np.ndarray:
    """Polyphase anti-aliased source-rate simulation."""
    if abs(float(source_rate_hz) - float(target_rate_hz)) <= 1.0:
        return np.asarray(iq, dtype=np.complex64)
    ratio = Fraction(
        int(round(target_rate_hz)), int(round(source_rate_hz)),
    ).limit_denominator(2000)
    return np.asarray(
        resample_poly(iq, ratio.numerator, ratio.denominator),
        dtype=np.complex64,
    )


def _modern_summary(
    iq: np.ndarray,
    fs: float,
    center: float,
    scan: dict[str, Any],
    video: dict[str, Any],
) -> dict[str, Any]:
    dwell_n = min(
        len(iq),
        int(scan.get("fft_size", 8192)) * int(scan.get("averages", 16)),
    )
    proposals, rf = sweep_inspect.propose_iq(
        iq[:dwell_n],
        fs,
        metadata_center_hz=center,
        scan=scan,
    )
    result: dict[str, Any] = {
        "rf_candidate": bool(proposals),
        "rf_proposals_seen": rf.proposals_seen,
        "rf_proposals_kept": rf.proposals_kept,
        "rf_threshold_db": round(rf.threshold_offset_db, 3),
        "rf_elapsed_ms": round(rf.elapsed_ms, 2),
    }
    if not proposals:
        return {
            **result,
            "stage": "rejected",
            "rejected_reason": "no occupied RF region",
            "analog_evidence": False,
            "video_confirmed": False,
            "lock_result": "not handed off",
        }
    proposal = proposals[0]
    offset = float(proposal.center_hz - center)
    evidence = sweep_inspect.inspect_iq(
        iq,
        fs,
        signal_offset_hz=offset,
        bandwidth_hz=proposal.bandwidth_hz,
        scan=scan,
        video=video,
    )
    return {
        **result,
        "proposal_offset_mhz": round(offset / 1e6, 4),
        "proposal_bw_mhz": round(proposal.bandwidth_hz / 1e6, 3),
        "stage": evidence.stage,
        "rejected_reason": evidence.rejected_reason,
        "analog_evidence": evidence.analog_evidence,
        "video_confirmed": evidence.video_confirmed,
        "standard": evidence.standard,
        "line_rate_hz": round(evidence.line_rate_hz, 2),
        "prominence_db": round(evidence.prominence_db, 2),
        "line_confidence": round(evidence.line_confidence, 3),
        "votes": f"{evidence.votes}/{evidence.windows}",
        "retention": round(evidence.retention, 3),
        "vertical_detail": round(evidence.vertical_detail, 3),
        "row_corr": round(evidence.row_corr, 3),
        "picture_score": round(evidence.picture_score, 3),
        "inspect_evaluations": evidence.evaluations,
        "inspect_samples": evidence.preview_samples,
        "inspect_elapsed_ms": round(evidence.elapsed_ms, 2),
        "lock_result": (
            "handoff with cached signal path"
            if evidence.analog_evidence else "not handed off"
        ),
    }


def evaluate(
    caps: Path,
    config: Path,
    *,
    requested_step_hz: float | None = None,
    channel_cap_hz: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    scan = dict(cfg.get("scan") or {})
    video = dict(cfg.get("video") or {})
    if requested_step_hz is not None:
        scan["step_hz"] = float(requested_step_hz)
    if channel_cap_hz is not None:
        scan["channel_bw_hz"] = float(channel_cap_hz)
        video["channel_bw_hz"] = float(channel_cap_hz)
    planned_cfg = {**cfg, "scan": scan, "video": video}
    plan = hardware_rate.derive_hardware_rate(planned_cfg)
    target_fs = plan.hardware_sample_rate_hz
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    peak_array_bytes = 0
    for sidecar in sorted(caps.glob("iq_*.json")):
        data_path = sidecar.with_suffix(".cf32")
        if not data_path.exists():
            continue
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        fs = float(meta["sample_rate"])
        center = float(meta.get("center_hz", 0.0))
        iq = np.fromfile(data_path, dtype=np.complex64)
        derived_iq = _resample_iq(iq, fs, target_fs)
        peak_array_bytes = max(
            peak_array_bytes, int(iq.nbytes + derived_iq.nbytes),
        )
        derived = _modern_summary(
            derived_iq, target_fs, center, scan, video,
        )
        dwell_n = min(len(iq), int(scan.get("fft_size", 8192))
                      * int(scan.get("averages", 16)))
        legacy = _legacy_proposals(iq[:dwell_n], fs, center, scan)
        legacy_result = (
            _legacy_inspect(iq, fs, center, legacy[0], scan)
            if legacy else {
                "accepted": False,
                "auto_handoff": False,
                "standard": "?",
                "prominence_db": 0.0,
                "row_corr": 0.0,
            }
        )
        proposals, rf = sweep_inspect.propose_iq(
            iq[:dwell_n],
            fs,
            metadata_center_hz=center,
            scan=scan,
        )
        row: dict[str, Any] = {
            "fixture": data_path.name,
            "label": _label(data_path),
            "sample_rate_mhz": fs / 1e6,
            "legacy_rf_candidate": bool(legacy),
            "legacy_inspect_accepted": legacy_result["accepted"],
            "legacy_auto_handoff": legacy_result["auto_handoff"],
            "legacy_standard": legacy_result["standard"],
            "legacy_prominence_db": legacy_result["prominence_db"],
            "legacy_row_corr": legacy_result["row_corr"],
            "rf_candidate": bool(proposals),
            "rf_proposals_seen": rf.proposals_seen,
            "rf_proposals_kept": rf.proposals_kept,
            "rf_threshold_db": round(rf.threshold_offset_db, 3),
            "rf_elapsed_ms": round(rf.elapsed_ms, 2),
            "derived_sample_rate_mhz": target_fs / 1e6,
            "derived_samples": int(derived_iq.size),
            **{f"derived_{key}": value for key, value in derived.items()},
        }
        if not proposals:
            row.update({
                "stage": "rejected",
                "rejected_reason": "no occupied RF region",
                "analog_evidence": False,
                "video_confirmed": False,
                "lock_result": "not handed off",
            })
            rows.append(row)
            continue
        proposal = proposals[0]
        offset = float(proposal.center_hz - center)
        evidence = sweep_inspect.inspect_iq(
            iq,
            fs,
            signal_offset_hz=offset,
            bandwidth_hz=proposal.bandwidth_hz,
            scan=scan,
            video=video,
        )
        row.update({
            "proposal_offset_mhz": round(offset / 1e6, 4),
            "proposal_peak_offset_mhz": round(
                (proposal.peak_hz - center) / 1e6, 4,
            ),
            "proposal_bw_mhz": round(proposal.bandwidth_hz / 1e6, 3),
            "proposal_snr_db": round(proposal.snr_db, 2),
            "stage": evidence.stage,
            "rejected_reason": evidence.rejected_reason,
            "analog_evidence": evidence.analog_evidence,
            "video_confirmed": evidence.video_confirmed,
            "standard": evidence.standard,
            "line_rate_hz": round(evidence.line_rate_hz, 2),
            "prominence_db": round(evidence.prominence_db, 2),
            "line_confidence": round(evidence.line_confidence, 3),
            "votes": f"{evidence.votes}/{evidence.windows}",
            "retention": round(evidence.retention, 3),
            "vertical_detail": round(evidence.vertical_detail, 3),
            "row_corr": round(evidence.row_corr, 3),
            "picture_score": round(evidence.picture_score, 3),
            "inspect_evaluations": evidence.evaluations,
            "inspect_samples": evidence.preview_samples,
            "inspect_elapsed_ms": round(evidence.elapsed_ms, 2),
            "lock_result": (
                "handoff with cached signal path"
                if evidence.analog_evidence else "not handed off"
            ),
        })
        rows.append(row)
    elapsed = time.perf_counter() - started
    summary = {
        "fixtures": len(rows),
        "target_rate_hz": target_fs,
        "rate_reason": plan.reason,
        "requested_step_hz": plan.requested_sweep_step_hz,
        "requested_channel_cap_hz": plan.requested_channel_cap_hz,
        "usable_span_hz": plan.usable_span_hz,
        "effective_sweep_step_hz": plan.effective_sweep_step_hz,
        "estimated_sc16_mb_s": plan.estimated_sc16_mb_s,
        "baseline_35msps_sc16_mb_s": hardware_rate.sc16_throughput_mb_s(35e6),
        "usb_payload_reduction_pct": round(
            100.0 * (
                1.0
                - plan.estimated_sc16_mb_s
                / hardware_rate.sc16_throughput_mb_s(35e6)
            ),
            1,
        ),
        "corpus_wall_s": round(elapsed, 2),
        "peak_input_plus_resampled_mb": round(
            peak_array_bytes / 1_000_000.0, 2,
        ),
    }
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--caps", type=Path, default=ROOT / "caps")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--step-mhz", type=float)
    parser.add_argument("--channel-cap-mhz", type=float)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    rows, resource_summary = evaluate(
        args.caps,
        args.config,
        requested_step_hz=(
            None if args.step_mhz is None else args.step_mhz * 1e6
        ),
        channel_cap_hz=(
            None
            if args.channel_cap_mhz is None
            else args.channel_cap_mhz * 1e6
        ),
    )
    if args.json:
        args.json.write_text(
            json.dumps(
                {"rows": rows, "resources": resource_summary},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    if not args.summary_only:
        for row in rows:
            print(json.dumps(row, ensure_ascii=False))
    positives = [row for row in rows if row["label"].endswith("positive")]
    result_summary = {
        "fixtures": len(rows),
        "legacy_rf_candidates": sum(
            bool(row["legacy_rf_candidate"]) for row in rows
        ),
        "legacy_inspect_accepted": sum(
            bool(row["legacy_inspect_accepted"]) for row in rows
        ),
        "legacy_auto_handoffs": sum(
            bool(row["legacy_auto_handoff"]) for row in rows
        ),
        "rf_candidates": sum(bool(row["rf_candidate"]) for row in rows),
        "analog_handoffs": sum(bool(row["analog_evidence"]) for row in rows),
        "video_confirmed": sum(bool(row["video_confirmed"]) for row in rows),
        "labeled_positive_handoffs": sum(
            bool(row["analog_evidence"]) for row in positives
        ),
        "labeled_positives": len(positives),
        "derived_rf_candidates": sum(
            bool(row["derived_rf_candidate"]) for row in rows
        ),
        "derived_analog_handoffs": sum(
            bool(row["derived_analog_evidence"]) for row in rows
        ),
        "derived_video_confirmed": sum(
            bool(row["derived_video_confirmed"]) for row in rows
        ),
        "derived_labeled_positive_handoffs": sum(
            bool(row["derived_analog_evidence"]) for row in positives
        ),
        "lost_labeled_positive_handoffs": [
            row["fixture"] for row in positives
            if row["analog_evidence"] and not row["derived_analog_evidence"]
        ],
        "derived_fixture_changes": [
            {
                "fixture": row["fixture"],
                "label": row["label"],
                "before": row["stage"],
                "after": row["derived_stage"],
            }
            for row in rows
            if (
                row["analog_evidence"] != row["derived_analog_evidence"]
                or row["video_confirmed"] != row["derived_video_confirmed"]
            )
        ],
    }
    print(json.dumps(result_summary, ensure_ascii=False))
    print(json.dumps(resource_summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
