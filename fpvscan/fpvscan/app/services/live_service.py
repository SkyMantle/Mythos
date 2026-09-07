from __future__ import annotations

from typing import Any

from fpvscan.app.services.catalog import catalog_values
from fpvscan.app.services.ports import EnginePort
from fpvscan.scan_view import grid_snapshot, spectrum_snapshot


class LiveService:
    def __init__(self, engine: EnginePort) -> None:
        self._engine = engine

    async def live_context(self) -> dict[str, Any]:
        snap = self._engine.snapshot()
        video = dict(snap.get("video") or {})
        spec = snap.get("spectrum") or {}
        freq = (
            spec.get("cursor_hz")
            or snap.get("lock_target")
            or snap.get("tuned_hz")
            or video.get("freq_hz")
        )
        detections = [
            {
                "freq_hz": d.get("freq_hz"),
                "snr_db": d.get("snr_db"),
                "bandwidth_hz": d.get("bandwidth_hz"),
                "standard": d.get("standard"),
                "confidence": d.get("confidence"),
                "channel": d.get("channel"),
                "band": d.get("band"),
                "pic_locked": d.get("pic_locked"),
            }
            for d in snap.get("detections") or []
        ]
        snr = None
        if freq is not None:
            for det in detections:
                if det.get("freq_hz") is None:
                    continue
                if abs(float(det["freq_hz"]) - float(freq)) <= 8e6:
                    snr = det.get("snr_db")
                    break
        cfg = self._engine.current_cfg()
        spectrum = snap.get("spectrum") or spectrum_snapshot(
            cfg,
            mode=str(snap.get("mode") or "SWEEP"),
            lock_target=snap.get("lock_target"),
            tuned_hz=float(snap.get("tuned_hz") or 0),
            last=None,
            afc_hz=float(snap.get("afc_hz") or 0),
        )
        grid = snap.get("grid") or grid_snapshot(
            cfg,
            mode=str(snap.get("mode") or "SWEEP"),
            sweep_i=int(snap.get("sweep_i") or 0),
            sweeps_done=int(snap.get("sweeps_done") or 0),
            sweep_pos_hz=float(snap.get("sweep_pos_hz") or 0),
            tuned_hz=float(snap.get("tuned_hz") or 0),
            lock_target=snap.get("lock_target"),
            visiting_hz=list(snap.get("visiting_hz") or []),
            afc_hz=float(snap.get("afc_hz") or 0),
        )
        return {
            "mode": str(snap.get("mode") or "SWEEP"),
            "frequency_hz": float(freq) if freq else None,
            "lock_state": bool(video.get("locked")) if video else False,
            "lock_target_hz": snap.get("lock_target"),
            "auto": bool(snap.get("auto")),
            "source": str(snap.get("source") or ""),
            "recording": bool(snap.get("recording")),
            "video_metrics": {
                "fps": snap.get("fps"),
                "locked": video.get("locked"),
                "standard": video.get("standard"),
                "line_rate": video.get("line_rate"),
                "lines": video.get("lines"),
                "afc_hz": snap.get("afc_hz"),
                "freq_err_hz": snap.get("freq_err_hz"),
                "afc_pegged": snap.get("afc_pegged"),
                "hunt_span_hz": snap.get("hunt_span_hz"),
                "timings_ms": dict(snap.get("timings_ms") or {}),
                "clip_frac": snap.get("clip_frac"),
                "overflows": snap.get("overflows"),
            },
            "parameters": catalog_values(cfg),
            "detections": detections,
            "last_frame_ref": snap.get("last_frame_ref"),
            "rssi": None,
            "snr": snr,
            "spectrum": spectrum,
            "grid": grid,
            "afc_pegged": snap.get("afc_pegged"),
            "freq_err_hz": snap.get("freq_err_hz"),
            "afc_hz": snap.get("afc_hz"),
            "hunt_span_hz": snap.get("hunt_span_hz"),
        }
