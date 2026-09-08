from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from fpvscan.app.repositories.sqlite_store import SqliteTestStore


class FakeEngine:
    def __init__(self) -> None:
        self.cfg = {
            "scan": {
                "threshold_db": 5.0,
                "confirm_hits": 2,
                "auto_peek": True,
                "start_hz": 400e6,
                "stop_hz": 6000e6,
                "sample_rate": 35e6,
                "channel_bw_hz": 10e6,
                "step_hz": 0,
                "cluster_step_mhz": "8",
                "hit_filter": "hide_weak",
                "fft_size": 8192,
                "averages": 8,
                "priority_bands": False,
            },
            "video": {
                "afc": True,
                "afc_gain": 0.5,
                "afc_deadband_hz": 80e3,
                "afc_max_step_hz": 0.25e6,
                "afc_digital_max_hz": 1.5e6,
                "capture_ms": 56,
                "rec_preset": "veryfast",
                "sharpen": 0.5,
                "sample_rate": 20e6,
                "channel_bw_hz": 10e6,
                "lo_offset_hz": 0.0,
                "hunt": True,
                "hunt_every": 20,
                "hunt_drop": 0.12,
                "average": 5,
                "motion_thresh": 24.0,
                "track_window_margin": 1.45,
                "h_phase_frac": 0.0,
                "h_pll": False,
                "spectrum_every": 1,
                "spectrum_every_4": False,
                "spectrum_pin_center": True,
                "spectrum_ema": True,
                "spectrum_smooth3": True,
            },
            "sdr": {"gain_db": 35, "auto_gain": True, "bias_tee": True, "settle_us": 500},
        }
        self._last_spectrum = {
            "bins": [-40.0, -20.0, -38.0],
            "center_hz": 4988e6,
            "span_hz": 20e6,
            "floor_db": -40.0,
            "peak_db": -20.0,
            "nfft": 2048,
        }
        self._defaults = copy.deepcopy(self.cfg)
        self.src = type("Src", (), {"set_gain": lambda self, _g: None})()
        self.commands: list[tuple] = []
        self._snap: dict = {
            "mode": "LOCK",
            "tuned_hz": 4988e6,
            "lock_target": 4988e6,
            "auto": False,
            "source": "sim",
            "recording": False,
            "fps": 6.2,
            "timings_ms": {"decode": 50.0},
            "clip_frac": 0.0,
            "overflows": 0,
            "afc_hz": 120.0,
            "freq_err_hz": 80.0,
            "afc_pegged": False,
            "hunt_span_hz": 0.25e6,
            "video": {
                "freq_hz": 4988e6,
                "standard": "PAL",
                "line_rate": 15625.0,
                "lines": 288,
                "locked": True,
            },
            "last_frame_ref": "out/photos/shot.webp",
            "sweep_pos_hz": 4988e6,
            "sweep_i": 12,
            "sweeps_done": 3,
            "detections": [{
                "freq_hz": 4988e6,
                "snr_db": 22.0,
                "bandwidth_hz": 6e6,
                "standard": "PAL",
                "confidence": 0.8,
                "channel": None,
                "band": "5G8",
                "pic_locked": True,
            }],
        }

    def snapshot(self) -> dict:
        from fpvscan.scan_view import grid_snapshot, spectrum_snapshot
        data = copy.deepcopy(self._snap)
        data["lock_target"] = self._snap.get("lock_target")
        data["spectrum"] = spectrum_snapshot(
            self.cfg,
            mode=str(data.get("mode") or "SWEEP"),
            lock_target=data.get("lock_target"),
            tuned_hz=float(data.get("tuned_hz") or 0),
            last=self._last_spectrum,
        )
        data["spectrum"]["afc_pegged"] = bool(data.get("afc_pegged"))
        data["spectrum"]["freq_err_hz"] = data.get("freq_err_hz")
        data["spectrum"]["afc_hz"] = data.get("afc_hz")
        data["spectrum"]["hunt_span_hz"] = data.get("hunt_span_hz")
        data["grid"] = grid_snapshot(
            self.cfg,
            mode=str(data.get("mode") or "SWEEP"),
            sweep_i=int(data.get("sweep_i") or 0),
            sweeps_done=int(data.get("sweeps_done") or 0),
            sweep_pos_hz=float(data.get("sweep_pos_hz") or 0),
            tuned_hz=float(data.get("tuned_hz") or 0),
            lock_target=data.get("lock_target"),
            visiting_hz=[4988e6],
        )
        return data

    def current_cfg(self) -> dict:
        return copy.deepcopy(self.cfg)

    def default_cfg(self) -> dict:
        return copy.deepcopy(self._defaults)

    def yaml_text(self) -> str:
        return (
            "scan:\n"
            "  threshold_db: 5.0 # над шумовою підлогою\n"
            "video:\n"
            "  afc: true # автопідстроювання\n"
        )

    def apply_values(self, values: dict) -> None:
        for key, value in values.items():
            section, name = key.split(".", 1)
            self.cfg[section][name] = value

    def current_param_values(self) -> dict:
        from fpvscan.app.services.catalog import catalog_values
        return catalog_values(self.cfg)

    def command(self, name: str, **kw) -> None:
        self.commands.append((name, kw))

    def refresh_lock(self) -> None:
        self.commands.append(("refresh_lock", {}))


@pytest.fixture
def engine() -> FakeEngine:
    return FakeEngine()


@pytest.fixture
def store() -> SqliteTestStore:
    return SqliteTestStore()
