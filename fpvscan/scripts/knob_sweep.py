#!/usr/bin/env python3
"""LOCK analog knob sweep against a running fpvscan HTTP API (Pi 5).

Run this ON the Raspberry Pi against local systemd ``fpvscan``. It does
not start a service, does not write config.yaml, and does not open the
web console. Default origin is loopback (config.yaml ``web.port``, else
8080) because the service binds ``0.0.0.0:<port>``. This is not a
Windows-localhost tool. From a PC, pass ``--base http://<pi-ip>:<port>``.

    /opt/fpvscan/.venv/bin/python ~/knob_sweep.py --freq <MHz from Rabbit> --out ~/knob-sweep/ --save-frames --base http://10.252.65.47:8080

Never wget -O ~/logging.py (shadows stdlib when cwd is $HOME). Use
~/knob_sweep.py only. Keep --save-frames as one token (--save-frame alias).

One factor at a time around live values. Default ``--group lock`` is analog
IF/AFC plus a small decode/CVBS slice (~30 trials when live values are
skipped). ``--group decode`` / ``--group cvbs`` add the full geometry grids.
Scan FFT/threshold knobs are skipped: they do not lock PAL.
Rolling/sheared picture is line period.

Default hunt is short (~90–120 s for ~30 trials) and **stops on the first
analog picture**. Pass ``--no-stop-on-picture`` to finish the full OFAT grid.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import socket
import statistics
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# Confirmed in fpvscan/web/server.py + app/routers. Do not invent paths.
API_STATE = "/api/state"
API_LOCK = "/api/lock/{freq_hz}"
API_SNAPSHOT = "/api/snapshot"
API_HEALTH = "/api/health"
API_PARAMS = "/api/test/parameters"
API_PARAMS_CURRENT = "/api/test/parameters/current"
API_LIVE = "/api/test/live"

DECODE_GRID: dict[str, list[Any]] = {
    "video.h_pll": [False, True],
    "video.h_phase_frac": [0.0, 0.03, -0.03],
    "video.track_window_margin": [1.25, 1.45, 1.75],
    "video.width": [480, 640],
}

CVBS_GRID: dict[str, list[Any]] = {
    "video.average": [0, 5, 12],
    "video.motion_thresh": [12.0, 24.0, 48.0],
    "video.sharpen": [0.0, 0.5],
    "video.auto_levels": [True, False],
    "video.crop_left_frac": [0.0, 0.07],
    "video.crop_bottom_lines": [0, 6],
}

# Analog IF + AFC, plus a decode/CVBS slice that can change the LOCK pane.
# 5.8-centric defaults. When --freq < BAND_SPLIT_HZ, apply_band_extras
# unions narrower BW / 13e6 sample_rate / capture 40+80. Not a MHz list.
# Live duplicates are skipped by OFAT; typical live leftovers
# (12e6 BW, 4.8e6 dev, 56ms, 20e6 fs, afc on, yaml picture defaults)
# → baseline + ~29 = ~30 trials on high band, not a full factorial.
DEFAULT_LOCK_GRID: dict[str, list[Any]] = {
    "video.channel_bw_hz": [8e6, 10e6, 12e6, 16e6, 20e6],
    "video.deviation_hz": [3.5e6, 4.8e6, 6e6, 8e6, 10e6],
    "video.capture_ms": [40, 56, 80, 100],
    "video.sample_rate": [13e6, 20e6, 35e6],
    "video.afc": [True, False],
    "video.afc_deadband_hz": [40e3, 80e3, 160e3],
    "video.afc_digital_max_hz": [0.5e6, 1.5e6, 2.5e6],
    "video.h_pll": [False, True],
    "video.h_phase_frac": [0.0, 0.03, -0.03],
    "video.average": [0, 5, 12],
    "video.sharpen": [0.0, 0.5],
    "video.width": [480, 640],
    "video.crop_left_frac": [0.0, 0.07],
    "video.crop_bottom_lines": [0, 6],
    "video.track_window_margin": [1.25, 1.45, 1.75],
}

GROUPS: dict[str, dict[str, list[Any]]] = {
    "lock": DEFAULT_LOCK_GRID,
    "decode": DECODE_GRID,
    "cvbs": CVBS_GRID,
}

# Generic split — not a channel list. Low-band analog IF is usually
# narrower than leftover 5.8 live knobs (20 Msps / 12e6 BW / 56 ms).
BAND_SPLIT_HZ = 2e9
LOW_BAND_EXTRA: dict[str, list[Any]] = {
    "video.channel_bw_hz": [4e6, 6e6],
    "video.sample_rate": [13e6],
    "video.capture_ms": [40, 80],
}
PAIRED_LOW_IF: dict[str, Any] = {
    "video.channel_bw_hz": 10e6,
    "video.sample_rate": 13e6,
}
ANALOG_CORR_FLOOR = 0.28

# Hunt timing: ~30 trials × (900 + 2200) ms ≈ 93 s, not 5+ minutes.
DEFAULT_DWELL_MS = 900.0
DEFAULT_FRESH_MS = 2200.0
ANALOG_SAMPLE_FLOOR_S = 0.4

# Sweep/inspect only — do not put these on an analog-picture grid.
SKIP_PREFIXES: tuple[str, ...] = (
    "scan.fft_size",
    "scan.averages",
    "scan.threshold",
    "scan.edge_guard",
    "scan.dc_notch",
    "scan.line_tol",
    "scan.inspect_",
)

EMPTY_VIDEO_GET_RETRIES = 8
EMPTY_VIDEO_NOTES = (
    "no video after dwell",
    "decode never published",
    "no video key",
)
LOW_BAND_KEY_ORDER = (
    "video.sample_rate",
    "video.capture_ms",
    "video.channel_bw_hz",
    "video.deviation_hz",
)

IF_KEYS = frozenset({
    "video.channel_bw_hz",
    "video.deviation_hz",
    "video.sample_rate",
    "video.capture_ms",
})
COSMETIC_KEYS = frozenset({
    "video.sharpen",
    "video.average",
    "video.width",
    "video.h_phase_frac",
    "video.track_window_margin",
    "video.crop_left_frac",
    "video.crop_bottom_lines",
})
MID_SWEEP_MSG = "signal may have appeared mid-sweep; knobs not attributed"

CSV_FIELDS = (
    "timestamp", "trial", "freq_hz", "applied_keys", "applied_values",
    "lock_target", "tuned_hz", "pic_score", "locked", "free_run",
    "line_rate", "row_corr", "luma_mean", "luma_std", "fps",
    "mode", "standard", "lines", "afc_hz", "http_status", "error",
    "frame_path", "video_freq_hz", "stale", "note",
)

# run.py uvicorn: cfg["web"]["host"], cfg["web"]["port"]. Service has no --port.
# Repo config.yaml binds 0.0.0.0 (loopback + ZeroTier). A Pi that was pointed
# at a single ZT IP (install_pi.sh documents this) refuses 127.0.0.1:8080.
DEFAULT_PORT = 8080
DEFAULT_LOOPBACK = "127.0.0.1"
WILDCARD_HOSTS = frozenset({"0.0.0.0", "::", "*", ""})
COMMON_PORTS = (8080,)
# Operator shaman ZeroTier — hint only, never the hardcoded default --base.
OPERATOR_BASE_HINT = "http://10.252.65.47:8080"

# AFC / RF snap can walk lock_target by ~1 MHz; kHz interp is not this.
LOCK_DRIFT_HZ = 0.5e6


def parse_freq_hz(raw: str) -> float:
    """Accept Hz (scientific or integer) or operator MHz (< 1e7)."""
    text = str(raw).strip().lower().replace("mhz", "").replace("hz", "").strip()
    value = float(text)
    if not (value > 0):
        raise ValueError(f"frequency must be positive, got {raw!r}")
    if value < 1e7:
        return value * 1e6
    return value


def is_skipped_key(key: str) -> bool:
    return any(key == p or key.startswith(p) for p in SKIP_PREFIXES)


def values_close(a: Any, b: Any, rel: float = 1e-3) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) is bool(b)
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return a == b
    if fa == fb:
        return True
    scale = max(abs(fa), abs(fb), 1.0)
    return abs(fa - fb) / scale <= rel


def is_low_band(freq_hz: float) -> bool:
    """True below 2 GHz. Not a named-channel list."""
    try:
        return float(freq_hz) < BAND_SPLIT_HZ
    except (TypeError, ValueError):
        return False


def apply_band_extras(
    grid: dict[str, list[Any]], freq_hz: float,
) -> dict[str, list[Any]]:
    """Union + front-load low-band IF when freq < 2e9. No-op at/above 2e9.

    Extends keys already in ``grid`` (lock group). Does not inject IF
    knobs into a decode-only grid. Threshold is generic, not a MHz list.
    13e6 is prepended even if live is 20e6 (OFAT still skips a live
    duplicate). sample_rate / capture run before leftover 5.8 BW OFAT.
    """
    out = {key: list(vals) for key, vals in grid.items()}
    if not is_low_band(freq_hz):
        return out
    for key, extra in LOW_BAND_EXTRA.items():
        if key not in out:
            continue
        prioritized: list[Any] = []
        for val in extra:
            if not any(values_close(val, existing) for existing in prioritized):
                prioritized.append(val)
        for val in out[key]:
            if not any(values_close(val, existing) for existing in prioritized):
                prioritized.append(val)
        out[key] = prioritized
    ordered: dict[str, list[Any]] = {}
    for key in LOW_BAND_KEY_ORDER:
        if key in out:
            ordered[key] = out[key]
    for key, vals in out.items():
        if key not in ordered:
            ordered[key] = vals
    return ordered


def is_analog_lock(
    metrics: dict[str, Any] | None, min_corr: float = ANALOG_CORR_FLOOR,
) -> bool:
    m = metrics or {}
    if not m.get("locked"):
        return False
    try:
        return float(m.get("row_corr") or 0.0) >= float(min_corr)
    except (TypeError, ValueError):
        return False


def paired_snow_trial(
    metrics: dict[str, Any] | None,
    freq_hz: float,
    current: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """One two-key IF patch after baseline when low-band analog is snow.

    Correlation ignores multi-key patches. High band / already-locked:
    no extra trial. Not a full factorial.
    """
    if not is_low_band(freq_hz):
        return None
    if is_analog_lock(metrics):
        return None
    live = current or {}
    if all(values_close(live.get(k), v) for k, v in PAIRED_LOW_IF.items()):
        return None
    return {"label": "paired-low-if", "values": dict(PAIRED_LOW_IF)}


def ofat_trials(
    current: dict[str, Any],
    grid: dict[str, list[Any]],
) -> list[dict[str, Any]]:
    """Baseline (empty patch) then one-key patches. Skip live duplicates."""
    trials: list[dict[str, Any]] = [{"label": "baseline", "values": {}}]
    for key, candidates in grid.items():
        if is_skipped_key(key):
            continue
        live = current.get(key)
        for cand in candidates:
            if live is not None and values_close(live, cand):
                continue
            trials.append({"label": f"{key}={cand}", "values": {key: cand}})
    return trials


def trial_score(row: dict[str, Any]) -> tuple[float, float]:
    """Keep-best / correlation key: row_corr primary, pic_score tie-break."""
    return (
        float(row.get("row_corr") or 0.0),
        float(row.get("pic_score") or 0.0),
    )


def applied_patch(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("applied_values") or ""
    if not raw:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    try:
        parsed = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def row_scorable(row: dict[str, Any]) -> bool:
    """False when video never published — must not rank or win keep-best."""
    if row.get("error"):
        return False
    note = str(row.get("note") or "").lower()
    if any(marker in note for marker in EMPTY_VIDEO_NOTES):
        return False
    if row.get("row_corr") is None:
        return False
    if row.get("pic_score") is None and row.get("line_rate") is None:
        return False
    return True


def correlate_knobs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank OFAT knobs by how much they move row_corr (pic_score tie-break).

    Ignores baseline (empty patch), multi-key patches, and empty-video
    rows. Groups remaining rows by the single applied key. ``delta`` is
    best_corr − worst_corr for that key (always ≥ 0 when n≥1). Ranking is
    by delta, then best_corr. Does not PUT winners — print-only.
    """
    groups: dict[str, list[tuple[Any, dict[str, Any]]]] = {}
    for row in rows:
        if not row_scorable(row):
            continue
        patch = applied_patch(row)
        if not patch:
            continue
        if len(patch) != 1:
            continue
        key, value = next(iter(patch.items()))
        groups.setdefault(key, []).append((value, row))
    ranked: list[dict[str, Any]] = []
    for key, items in groups.items():
        best_val, best_row = max(items, key=lambda item: trial_score(item[1]))
        worst_val, worst_row = min(items, key=lambda item: trial_score(item[1]))
        best_corr = float(best_row.get("row_corr") or 0.0)
        worst_corr = float(worst_row.get("row_corr") or 0.0)
        ranked.append({
            "key": key,
            "n": len(items),
            "best_value": best_val,
            "best_corr": best_corr,
            "worst_value": worst_val,
            "worst_corr": worst_corr,
            "delta": best_corr - worst_corr,
        })
    ranked.sort(key=lambda item: (item["delta"], item["best_corr"]), reverse=True)
    return ranked


def format_correlation(ranked: list[dict[str, Any]]) -> str:
    """Always return a named block so Ctrl+C / luma failure still shows one."""
    if not ranked:
        return "knob correlation (none; not applied)"
    lines = [
        "knob correlation (row_corr range per OFAT key; not applied):",
    ]
    for item in ranked:
        lines.append(
            f"  {str(item['key']):<32} n={item['n']:<2}"
            f"  best={item['best_value']} corr={item['best_corr']:.3f}"
            f"  worst={item['worst_value']} corr={item['worst_corr']:.3f}"
            f"  delta={item['delta']:.3f}"
        )
    return "\n".join(lines)


def infer_free_run(video: dict[str, Any] | None, locked: bool | None) -> bool | None:
    """Prefer engine free_run; else unlocked video present = snow path."""
    if not video:
        return None
    if video.get("free_run") is not None:
        return bool(video.get("free_run"))
    if locked is None:
        locked = video.get("locked")
    if locked is None:
        return None
    return not bool(locked)


def still_wait_s(dwell_s: float, fps: float | None) -> float:
    """POST /api/snapshot waits for the next analog frame (0.3 fps → >3 s)."""
    extra = 0.0
    try:
        if fps is not None and float(fps) > 0.05:
            extra = 2.5 / float(fps)
    except (TypeError, ValueError):
        extra = 0.0
    return max(6.0, float(dwell_s), extra)


def still_skip_note(metrics: dict[str, Any] | None, got_frame: bool) -> str | None:
    """Truthful NOTE when --save-frames got no new still."""
    if got_frame:
        return None
    m = metrics or {}
    locked = m.get("locked")
    free = m.get("free_run")
    corr = m.get("row_corr")
    analog = False
    try:
        analog = corr is not None and float(corr) >= ANALOG_CORR_FLOOR
    except (TypeError, ValueError):
        analog = False
    if locked or analog:
        return "no new still (timeout or copy; lock was analog)"
    if free:
        return "no new still (snow/free_run never writes)"
    return "no new still"


def video_missing(state: dict[str, Any] | None) -> bool:
    if not isinstance(state, dict):
        return True
    video = state.get("video")
    if not isinstance(video, dict) or not video:
        return True
    return video.get("pic_score") is None and video.get("line_rate") is None


def diagnose_state(state: dict[str, Any] | None, freq_hz: float) -> str | None:
    """Why pic_score/line_rate are None — empty video used to be silent."""
    if not state:
        return "empty /api/state"
    notes: list[str] = []
    mode = str(state.get("mode") or "").upper()
    if mode != "LOCK":
        notes.append(f"mode={mode or 'missing'} (not LOCK)")
    target = state.get("lock_target")
    if target is None:
        notes.append("lock_target=None")
    else:
        try:
            if abs(float(target) - freq_hz) >= 1e6:
                notes.append(
                    f"lock_target={float(target)/1e6:.3f}MHz "
                    f"!= {freq_hz/1e6:.3f}MHz"
                )
        except (TypeError, ValueError):
            notes.append(f"lock_target={target!r}")
    if video_missing(state):
        notes.append("no video key (decode never ran, or HTTP snapshot stale)")
    return "; ".join(notes) or None


def lock_target_drifted(
    lock_target: Any, freq_hz: float, max_hz: float = LOCK_DRIFT_HZ,
) -> bool:
    """True when AFC moved lock_target off the requested sweep frequency."""
    try:
        return abs(float(lock_target) - float(freq_hz)) >= float(max_hz)
    except (TypeError, ValueError):
        return False


def relock_if_drifted(
    http: HttpClient, freq_hz: float, lock_target: Any,
) -> str | None:
    """POST /api/lock/--freq when AFC walked the target. None if already on freq."""
    if not lock_target_drifted(lock_target, freq_hz):
        return None
    post_lock(http, freq_hz, force=True)
    try:
        was = f"{float(lock_target) / 1e6:.3f}"
    except (TypeError, ValueError):
        was = repr(lock_target)
    return f"re-locked {freq_hz / 1e6:.3f} MHz after AFC walk (was {was})"


def metrics_fingerprint(metrics: dict[str, Any] | None) -> tuple[Any, ...]:
    if not metrics:
        return (None,)

    def _r(val: Any, nd: int) -> Any:
        if val is None:
            return None
        try:
            return round(float(val), nd)
        except (TypeError, ValueError):
            return val

    return (
        _r(metrics.get("pic_score"), 3),
        _r(metrics.get("line_rate"), 1),
        metrics.get("locked"),
        _r(metrics.get("row_corr"), 3),
        _r(metrics.get("video_freq_hz"), 0),
    )


def extract_metrics(state: dict[str, Any], live: dict[str, Any] | None = None) -> dict[str, Any]:
    """Metrics from GET /api/state, filling gaps from engine ``_last_video``.

    ``snapshot()`` already copies ``_last_video`` into ``video`` so HTTP
    clients usually see a live blob. Overlay keeps tests and raw dumps
    consistent when ``video`` is stale/empty and ``_last_video`` is present.
    """
    video = dict(state.get("video") or {})
    last = state.get("_last_video")
    if isinstance(last, dict) and last:
        filled = dict(last)
        filled.update({k: v for k, v in video.items() if v is not None})
        video = filled
    live = live or {}
    vm = dict(live.get("video_metrics") or {})
    locked = video.get("locked")
    if locked is None:
        locked = vm.get("locked")
    if locked is None:
        locked = live.get("lock_state")
    pic = video.get("pic_score")
    row = video.get("row_corr")
    line_rate = video.get("line_rate")
    if line_rate is None:
        line_rate = vm.get("line_rate")
    fps = state.get("fps")
    if fps is None:
        fps = vm.get("fps")
    free = infer_free_run(video, locked if isinstance(locked, bool) else None)
    video_freq = video.get("freq_hz")
    return {
        "mode": state.get("mode") or live.get("mode"),
        "lock_target": state.get("lock_target") if state.get("lock_target") is not None
        else live.get("lock_target_hz"),
        "tuned_hz": state.get("tuned_hz"),
        "pic_score": pic,
        "locked": locked,
        "free_run": free,
        "line_rate": line_rate,
        "row_corr": row,
        "fps": fps,
        "standard": video.get("standard") or vm.get("standard"),
        "lines": video.get("lines") if video.get("lines") is not None else vm.get("lines"),
        "afc_hz": state.get("afc_hz") if state.get("afc_hz") is not None else vm.get("afc_hz"),
        "last_frame_ref": state.get("last_frame_ref") or live.get("last_frame_ref"),
        "clip_frac": state.get("clip_frac"),
        "overflows": state.get("overflows"),
        "video_freq_hz": video_freq,
    }


def sanitize_import_path() -> None:
    """Stop cwd/home ``logging.py`` from shadowing stdlib (wget -O ~/logging.py)."""
    drop_dirs: set[str] = set()
    for folder in (Path.home(), Path(__file__).resolve().parent):
        try:
            resolved = folder.resolve()
        except OSError:
            continue
        if (resolved / "logging.py").is_file():
            drop_dirs.add(str(resolved))
    cleaned: list[str] = []
    for entry in sys.path:
        if entry == "":
            continue
        try:
            resolved = str(Path(entry).resolve())
        except OSError:
            resolved = entry
        if resolved in drop_dirs:
            continue
        cleaned.append(entry)
    sys.path[:] = cleaned


def luma_stats(path: Path) -> tuple[float | None, float | None]:
    """Best-effort luma from a still. Never raise — skip if PIL/stdlib is broken.

    Use ``tobytes()`` / numpy — Pillow 12+ deprecates ``Image.getdata``.
    """
    try:
        sanitize_import_path()
        from PIL import Image  # type: ignore
        im = Image.open(path).convert("L")
        try:
            import numpy as np  # type: ignore
            arr = np.asarray(im, dtype=np.float64)
            if arr.size == 0:
                return None, None
            return float(arr.mean()), float(arr.std())
        except Exception:
            raw = im.tobytes()
    except Exception:
        return None, None
    if not raw:
        return None, None
    try:
        n = len(raw)
        mean = float(sum(raw) / n)
        std = float(statistics.pstdev(raw)) if n > 1 else 0.0
        return mean, std
    except Exception:
        return None, None


def _json_body(payload: dict[str, Any] | None) -> bytes | None:
    if payload is None:
        return None
    return json.dumps(payload).encode("utf-8")


class HttpClient:
    def __init__(self, base: str, timeout_s: float = 15.0) -> None:
        self.base = base.rstrip("/")
        self.timeout_s = timeout_s

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> tuple[int, Any]:
        url = self.base + path
        body = _json_body(payload)
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        headers["X-Request-Id"] = uuid.uuid4().hex
        req = Request(url, data=body, method=method.upper(), headers=headers)
        try:
            with urlopen(req, timeout=timeout_s or self.timeout_s) as resp:
                raw = resp.read()
                status = int(getattr(resp, "status", 200) or 200)
        except HTTPError as exc:
            raw = exc.read() if exc.fp is not None else b""
            status = int(exc.code)
            data = _decode_json(raw)
            # 2xx is success even if journald logged a formatter KeyError.
            if 200 <= status < 300:
                return status, data
            raise SweepHttpError(status, path, data) from exc
        except URLError as exc:
            raise SweepHttpError(0, path, {"error": str(exc.reason)}) from exc
        return status, _decode_json(raw)


def _decode_json(raw: bytes) -> Any:
    if not raw:
        return None
    text = raw.decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}


class SweepHttpError(RuntimeError):
    def __init__(self, status: int, path: str, data: Any) -> None:
        self.status = status
        self.path = path
        self.data = data
        super().__init__(f"HTTP {status} {path}: {data}")


def loopback_origin(port: int) -> str:
    return f"http://{DEFAULT_LOOPBACK}:{int(port)}"


def _yaml_scalar(raw: str) -> str:
    return raw.split("#", 1)[0].strip().strip("'\"")


def read_web_bind(path: Path) -> tuple[str | None, int | None]:
    """Parse web.host / web.port from config.yaml (no PyYAML required)."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None, None
    host: str | None = None
    port: int | None = None
    in_web = False
    for line in text.splitlines():
        stripped = line.split("#", 1)[0].rstrip()
        if not stripped.strip():
            continue
        indent = len(stripped) - len(stripped.lstrip(" \t"))
        key_line = stripped.strip()
        if indent == 0 and key_line.endswith(":"):
            in_web = key_line == "web:"
            continue
        if not in_web:
            continue
        if key_line.startswith("host:"):
            val = _yaml_scalar(key_line.split(":", 1)[1])
            host = val or None
        elif key_line.startswith("port:"):
            val = _yaml_scalar(key_line.split(":", 1)[1])
            try:
                port = int(float(val))
            except (TypeError, ValueError):
                pass
    return host, port


def default_config_paths(install_root: str | Path) -> list[Path]:
    return [
        Path(install_root).expanduser() / "config.yaml",
        Path.cwd() / "config.yaml",
        Path(__file__).resolve().parent.parent / "config.yaml",
    ]


def local_ipv4_hosts() -> list[str]:
    """Non-loopback IPv4s on this machine (ZeroTier, LAN). 0.0.0.0 is not a client URL."""
    found: list[str] = []

    def add(ip: str | None) -> None:
        if not ip:
            return
        ip = str(ip).strip()
        if not ip or ip in found:
            return
        if ip.startswith("127.") or ip in WILDCARD_HOSTS or ip in ("localhost", "::1"):
            return
        found.append(ip)

    try:
        add(socket.gethostbyname(socket.gethostname()))
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            add(info[4][0])
    except OSError:
        pass
    # UDP connect picks the source IP for that route (ZT 10.252/16, else default).
    for peer in (("10.252.65.1", 1), ("8.8.8.8", 80)):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                sock.connect(peer)
                add(sock.getsockname()[0])
            finally:
                sock.close()
        except OSError:
            pass
    return found


def list_base_candidates(
    *,
    explicit: str | None = None,
    env_base: str | None = None,
    config_paths: list[Path] | None = None,
    default_port: int = DEFAULT_PORT,
    local_hosts: list[str] | None = None,
    discover_local: bool = True,
) -> list[str]:
    """Origins: --base, $FPVSCAN_BASE, loopback, config web.host, local IPv4s.

    ``0.0.0.0`` is a bind address, never a client URL — use 127.0.0.1 unless
    the service is bound to a single non-loopback IP (then try that IP too).
    """
    out: list[str] = []

    def add(url: str | None) -> None:
        if not url:
            return
        cleaned = url.strip().rstrip("/")
        if cleaned and cleaned not in out:
            out.append(cleaned)

    add(explicit)
    add((env_base or "").strip() or None)

    cfg_host: str | None = None
    cfg_port: int | None = None
    for path in config_paths or []:
        host, port = read_web_bind(path)
        if host is not None or port is not None:
            cfg_host, cfg_port = host, port
            break

    port = int(cfg_port) if cfg_port else int(default_port)
    add(loopback_origin(port))
    if cfg_host and cfg_host not in WILDCARD_HOSTS and cfg_host not in (
        DEFAULT_LOOPBACK, "localhost", "::1",
    ):
        add(f"http://{cfg_host}:{port}")
    extras = local_hosts if local_hosts is not None else (
        local_ipv4_hosts() if discover_local else []
    )
    for ip in extras:
        add(f"http://{ip}:{port}")
    for extra in COMMON_PORTS:
        add(loopback_origin(extra))
    return out


def is_unreachable(exc: SweepHttpError) -> bool:
    """HTTP 0 / errno 111: nothing listening. Not a 404 on /api/health."""
    return int(exc.status) == 0


def format_unreachable(
    tried: list[str],
    last_err: SweepHttpError | None,
    *,
    default_port: int = DEFAULT_PORT,
) -> str:
    shown = tried[0] if tried else loopback_origin(default_port)
    tried_txt = ", ".join(tried) if tried else shown
    last = str(last_err) if last_err else "connection refused"
    return (
        f"cannot reach {shown}: {last}\n"
        f"Tried: {tried_txt}\n"
        f"fpvscan is not listening on {shown}; this CLI talks to the Pi "
        f"service locally, not to your PC. Use --base {OPERATOR_BASE_HINT} "
        f"if the console is on ZeroTier (uvicorn web.host is that IP, not loopback).\n"
        f"Connection refused (HTTP 0 / errno 111) means nothing is bound on that "
        f"host:port — not a missing /api/health route, and not a Windows-localhost script.\n"
        f"On the Pi: sudo systemctl status fpvscan ; "
        f"ss -lntp | grep -E '{default_port}|python'\n"
        f"Last error: {last}"
    )


def apply_parameters(http: HttpClient, values: dict[str, Any]) -> dict[str, Any]:
    body = {"idempotency_key": str(uuid.uuid4()), "values": values}
    status, data = http.request("PUT", API_PARAMS, body)
    if not (200 <= status < 300):
        raise SweepHttpError(status, API_PARAMS, data)
    if not isinstance(data, dict):
        data = {}
    data["http_status"] = status
    return data


def post_lock(http: HttpClient, freq_hz: float, force: bool = True) -> dict[str, Any]:
    path = API_LOCK.format(freq_hz=int(round(freq_hz)))
    if force:
        path += "?force=1"
    status, data = http.request("POST", path)
    if not (200 <= status < 300):
        raise SweepHttpError(status, path, data)
    return {"http_status": status, "data": data}


def _pull_state_live(http: HttpClient) -> tuple[dict[str, Any], dict[str, Any] | None]:
    _, state = http.request("GET", API_STATE)
    if not isinstance(state, dict):
        state = {}
    live: dict[str, Any] | None = None
    try:
        _, live_raw = http.request("GET", API_LIVE)
        live = live_raw if isinstance(live_raw, dict) else None
    except SweepHttpError:
        live = None
    return state, live


def wait_settle(
    http: HttpClient,
    *,
    freq_hz: float,
    dwell_s: float,
    extra_s: float = 0.0,
    fresh_s: float = 0.0,
    before: dict[str, Any] | None = None,
    need_video: bool = True,
    poll_s: float = 0.25,
    sleep_fn=None,
    clock_fn=None,
) -> tuple[dict[str, Any], dict[str, Any] | None, str | None]:
    """Poll until LOCK on freq, dwell, then wait for a *new* decode blob.

    ``before`` is the previous trial's metrics. After a PUT, HTTP used to
    return the same cached video forever; we now wait until pic_score /
    line_rate / video.freq_hz change (or timeout, then mark stale).
    Analog lock (locked + corr ≥ floor) after ``ANALOG_SAMPLE_FLOOR_S``
    ends the wait early so a hunt does not burn the rest of dwell/fresh.
    """
    if sleep_fn is None:
        sleep_fn = time.sleep
    if clock_fn is None:
        clock_fn = time.monotonic
    notes: list[str] = []
    t0 = clock_fn()
    ready_deadline = t0 + max(dwell_s, 1.0) + extra_s
    state: dict[str, Any] = {}
    last_err: str | None = None
    locked_ok = False
    while clock_fn() < ready_deadline:
        try:
            _, state = http.request("GET", API_STATE)
            if not isinstance(state, dict):
                state = {}
        except SweepHttpError as exc:
            last_err = str(exc)
            sleep_fn(poll_s)
            continue
        mode = str(state.get("mode") or "").upper()
        target = state.get("lock_target")
        on_freq = target is not None and abs(float(target) - freq_hz) < 1e6
        if mode == "LOCK" and on_freq:
            locked_ok = True
            break
        sleep_fn(poll_s)
    else:
        if last_err and not state:
            raise SweepHttpError(0, API_STATE, {"error": last_err})
    if not locked_ok:
        notes.append(
            f"not LOCK on {freq_hz/1e6:.3f} MHz "
            f"(mode={state.get('mode')!s}, lock_target={state.get('lock_target')})"
        )

    sleep_fn(dwell_s + extra_s)
    state, live = _pull_state_live(http)
    for _ in range(EMPTY_VIDEO_GET_RETRIES):
        if not need_video or not video_missing(state):
            break
        sleep_fn(poll_s)
        try:
            state, live = _pull_state_live(http)
        except SweepHttpError as exc:
            notes.append(str(exc))
            break

    before_fp = metrics_fingerprint(before) if before else None
    wait_s = max(fresh_s, 0.0)
    timed_out_fresh = False
    if wait_s > 0 and (need_video or before_fp is not None):
        fresh_deadline = clock_fn() + wait_s
        while clock_fn() < fresh_deadline:
            metrics = extract_metrics(state, live)
            video_ok = (not need_video) or (not video_missing(state))
            fresh_ok = True
            if before_fp is not None:
                fresh_ok = metrics_fingerprint(metrics) != before_fp
            analog_ok = is_analog_lock(metrics)
            elapsed = clock_fn() - t0
            if analog_ok and elapsed >= ANALOG_SAMPLE_FLOOR_S and video_ok:
                break
            if video_ok and fresh_ok:
                break
            sleep_fn(poll_s)
            try:
                state, live = _pull_state_live(http)
            except SweepHttpError as exc:
                notes.append(str(exc))
                break
        else:
            timed_out_fresh = True
            metrics = extract_metrics(state, live)
            if need_video and video_missing(state):
                notes.append("no video after dwell (decode never published)")
            elif before_fp is not None and metrics_fingerprint(metrics) == before_fp:
                notes.append("metrics unchanged after PUT (stale snapshot or IF no-op)")

    if need_video and video_missing(state) and "no video after dwell" not in " ".join(notes):
        notes.append("no video after dwell (decode never published)")

    diag = diagnose_state(state, freq_hz)
    if diag and (video_missing(state) or not locked_ok):
        notes.append(diag)
    note = "; ".join(dict.fromkeys(n for n in notes if n)) or None
    _ = timed_out_fresh
    return state, live, note


def maybe_save_frame(
    http: HttpClient,
    *,
    out_dir: Path,
    install_root: Path,
    prev_ref: str | None,
    timeout_s: float,
) -> Path | None:
    try:
        http.request("POST", API_SNAPSHOT)
    except SweepHttpError:
        return None
    deadline = time.monotonic() + timeout_s
    ref = prev_ref
    while time.monotonic() < deadline:
        try:
            _, state = http.request("GET", API_STATE)
        except SweepHttpError:
            time.sleep(0.2)
            continue
        if isinstance(state, dict):
            ref = state.get("last_frame_ref") or ref
            if ref and ref != prev_ref:
                break
        time.sleep(0.2)
    if not ref or ref == prev_ref:
        return None
    src = Path(ref)
    if not src.is_absolute():
        src = install_root / ref
    if not src.is_file():
        return None
    dest = out_dir / "frames" / src.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return dest


def current_values(http: HttpClient) -> dict[str, Any]:
    status, data = http.request("GET", API_PARAMS_CURRENT)
    if not (200 <= status < 300) or not isinstance(data, dict):
        raise SweepHttpError(status, API_PARAMS_CURRENT, data)
    values = data.get("values")
    if not isinstance(values, dict):
        raise SweepHttpError(status, API_PARAMS_CURRENT, data)
    return values


def merge_grid(group_names: list[str], extra_keys: list[str]) -> dict[str, list[Any]]:
    grid: dict[str, list[Any]] = {}
    for name in group_names:
        block = GROUPS.get(name)
        if block is None:
            raise SystemExit(f"unknown group {name!r}; choose {sorted(GROUPS)}")
        grid.update(block)
    for key in extra_keys:
        if key not in grid:
            grid[key] = []
    return grid


def row_from_trial(
    *,
    trial_i: int,
    freq_hz: float,
    patch: dict[str, Any],
    metrics: dict[str, Any],
    error: str | None,
    http_status: int | None,
    frame_path: str | None,
    luma: tuple[float | None, float | None],
    stale: bool = False,
    note: str | None = None,
) -> dict[str, Any]:
    keys = sorted(patch)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trial": trial_i,
        "freq_hz": freq_hz,
        "applied_keys": ",".join(keys),
        "applied_values": json.dumps(patch, default=str) if patch else "",
        "lock_target": metrics.get("lock_target"),
        "tuned_hz": metrics.get("tuned_hz"),
        "pic_score": metrics.get("pic_score"),
        "locked": metrics.get("locked"),
        "free_run": metrics.get("free_run"),
        "line_rate": metrics.get("line_rate"),
        "row_corr": metrics.get("row_corr"),
        "luma_mean": luma[0],
        "luma_std": luma[1],
        "fps": metrics.get("fps"),
        "mode": metrics.get("mode"),
        "standard": metrics.get("standard"),
        "lines": metrics.get("lines"),
        "afc_hz": metrics.get("afc_hz"),
        "http_status": http_status,
        "error": error,
        "frame_path": frame_path,
        "video_freq_hz": metrics.get("video_freq_hz"),
        "stale": stale,
        "note": note,
    }


def _row_trial_i(row: dict[str, Any]) -> int:
    try:
        return int(row.get("trial") or 0)
    except (TypeError, ValueError):
        return 0


def row_lock_drifted(row: dict[str, Any]) -> bool:
    freq = row.get("freq_hz")
    if freq is None:
        return False
    try:
        hz = float(freq)
    except (TypeError, ValueError):
        return False
    return (
        lock_target_drifted(row.get("lock_target"), hz)
        or lock_target_drifted(row.get("tuned_hz"), hz)
    )


def patch_is_cosmetic(patch: dict[str, Any]) -> bool:
    if not patch:
        return False
    for key in patch:
        if str(key).startswith("video.crop_"):
            continue
        if key in COSMETIC_KEYS:
            continue
        return False
    return True


def patch_is_if(patch: dict[str, Any]) -> bool:
    return bool(patch) and all(key in IF_KEYS for key in patch)


def row_analog(row: dict[str, Any]) -> bool:
    if not row_scorable(row) or row_lock_drifted(row):
        return False
    return is_analog_lock(row)


def early_trials_were_snow(rows: list[dict[str, Any]]) -> bool:
    analog = [row for row in rows if row_analog(row)]
    if not analog:
        return False
    first_i = min(_row_trial_i(row) for row in analog)
    if first_i <= 0:
        return False
    earlier = [row for row in rows if _row_trial_i(row) < first_i]
    if not earlier:
        return False
    return not any(row_analog(row) for row in earlier)


def mid_sweep_signal_note(rows: list[dict[str, Any]]) -> str | None:
    if not early_trials_were_snow(rows):
        return None
    analog = [row for row in rows if row_analog(row)]
    if not analog:
        return None
    if any(patch_is_if(applied_patch(row)) for row in analog):
        return None
    if any(patch_is_cosmetic(applied_patch(row)) for row in analog):
        return MID_SWEEP_MSG
    return None


def _eligible_winner_rows(
    rows: list[dict[str, Any]], *, analog_only: bool,
) -> list[dict[str, Any]]:
    eligible = [
        row for row in rows
        if row_scorable(row) and not row_lock_drifted(row)
        and row.get("row_corr") is not None
    ]
    if analog_only:
        eligible = [row for row in eligible if row_analog(row)]
    if not eligible:
        return []
    if early_trials_were_snow(rows):
        analog_if = [
            row for row in eligible
            if patch_is_if(applied_patch(row)) and row_analog(row)
        ]
        if analog_if:
            return analog_if
        # Late cosmetics after snow are the TX turning on, not a knob win.
        # Snow IF rows must not become the printed winner.
        return []
    return eligible


def best_trial_patch(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Winning OFAT patch: highest row_corr, then pic_score. None if no metrics."""
    scored = _eligible_winner_rows(rows, analog_only=False)
    if not scored:
        return None
    best = max(scored, key=trial_score)
    return applied_patch(best)


def analog_winner_patch(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Best locked + row_corr ≥ ANALOG_CORR_FLOOR patch. None if only snow."""
    scored = _eligible_winner_rows(rows, analog_only=True)
    if not scored:
        return None
    patch = applied_patch(max(scored, key=trial_score))
    return patch or None


def format_winning_put(patch: dict[str, Any], base: str | None = None) -> str:
    origin = (base or OPERATOR_BASE_HINT).rstrip("/")
    body = {
        "idempotency_key": str(uuid.uuid4()),
        "values": patch,
    }
    payload = json.dumps(body, default=str)
    return (
        "winning PUT (not applied; pass --keep-best next time to keep it):\n"
        f"  PUT {API_PARAMS} values={json.dumps(patch, default=str)}\n"
        f"  curl -sS -X PUT {origin}{API_PARAMS} "
        f"-H \"Content-Type: application/json\" -d '{payload}'"
    )


def write_outputs(out_dir: Path, rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "knob_sweep.jsonl"
    csv_path = out_dir / "knob_sweep.csv"
    with jsonl.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str) + "\n")
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return jsonl, csv_path


def _mhz(val: Any) -> str:
    try:
        return f"{float(val) / 1e6:.3f}"
    except (TypeError, ValueError):
        return str(val)


def _print_trial_line(i: int, label: str, row: dict[str, Any]) -> None:
    extra = ""
    if row.get("stale"):
        extra += "  STALE"
    if row.get("error"):
        extra += f"  ERR {row['error']}"
    if row.get("note"):
        extra += f"  NOTE {row['note']}"
    print(
        f"  [{i:03d}] {label:<40} "
        f"score={row.get('pic_score')} locked={row.get('locked')} "
        f"free_run={row.get('free_run')} line={row.get('line_rate')} "
        f"corr={row.get('row_corr')} tuned={_mhz(row.get('tuned_hz'))} "
        f"fps={row.get('fps')} vfreq={_mhz(row.get('video_freq_hz'))}"
        + extra
    )


def _picture_ok(metrics: dict[str, Any], freq_hz: float) -> bool:
    if not is_analog_lock(metrics):
        return False
    if lock_target_drifted(metrics.get("lock_target"), freq_hz):
        return False
    if lock_target_drifted(metrics.get("tuned_hz"), freq_hz):
        return False
    return True


def run(args: argparse.Namespace) -> int:
    sanitize_import_path()
    freq_hz = parse_freq_hz(args.freq)
    out_dir = Path(args.out).expanduser()
    install_root = Path(args.install_root).expanduser()
    if not args.group:
        args.group = ["lock"]
    extra_keys = [k.strip() for k in (args.keys or "").split(",") if k.strip()]
    grid = apply_band_extras(merge_grid(args.group, extra_keys), freq_hz)
    for bad in list(grid):
        if is_skipped_key(bad) and bad not in extra_keys:
            del grid[bad]

    candidates = list_base_candidates(
        explicit=args.base,
        env_base=os.environ.get("FPVSCAN_BASE"),
        config_paths=default_config_paths(install_root),
    )
    http: HttpClient | None = None
    last_err: SweepHttpError | None = None
    chosen = candidates[0] if candidates else loopback_origin(DEFAULT_PORT)
    for i, base in enumerate(candidates):
        client = HttpClient(base, timeout_s=args.http_timeout)
        try:
            client.request("GET", API_HEALTH)
            http = client
            chosen = base
            if i > 0:
                print(f"using {base} (auto-detected)", file=sys.stderr)
            break
        except SweepHttpError as exc:
            last_err = exc
            if not is_unreachable(exc):
                print(f"cannot reach {base}: {exc}", file=sys.stderr)
                return 2
    if http is None:
        print(format_unreachable(candidates, last_err), file=sys.stderr)
        return 2
    args.base = chosen
    assert http is not None

    try:
        live_values = current_values(http)
    except SweepHttpError as exc:
        if is_unreachable(exc):
            print(format_unreachable([chosen], exc), file=sys.stderr)
        else:
            print(f"cannot read current parameters from {chosen}: {exc}", file=sys.stderr)
        return 2

    for key in extra_keys:
        if key not in live_values:
            print(f"unknown parameter {key!r} (not in GET {API_PARAMS_CURRENT})",
                  file=sys.stderr)
            return 2
        if not grid.get(key):
            live = live_values[key]
            if isinstance(live, bool):
                grid[key] = [not live]
            elif isinstance(live, (int, float)) and not isinstance(live, bool):
                grid[key] = [live * 0.75, live * 1.25] if live else [live]

    trials = ofat_trials(live_values, grid)
    extra_trial = None
    if args.dry_run:
        payload = {
            "freq_hz": freq_hz,
            "n": len(trials),
            "trials": trials,
            "low_band": is_low_band(freq_hz),
            "paired_if_snow": True,
        }
        print(json.dumps(payload, default=str, indent=2))
        return 0

    print(f"LOCK {freq_hz/1e6:.3f} MHz  {len(trials)} trials  {args.base}")
    if is_low_band(freq_hz):
        print(
            "low-band IF slice (<2 GHz): sample_rate/capture first, "
            "13e6 kept even if live is 20e6, paired BW×fs if baseline snow"
        )
    try:
        post_lock(http, freq_hz, force=True)
    except SweepHttpError as exc:
        print(f"POST lock failed: {exc}", file=sys.stderr)
        return 2

    dwell_s = args.dwell_ms / 1000.0
    fresh_s = max(0.0, float(args.fresh_ms) / 1000.0)
    rows: list[dict[str, Any]] = []
    prev_ref: str | None = None
    restore_keys = sorted(k for k in grid if k in live_values)
    engine_vals = dict(live_values)
    prev_metrics: dict[str, Any] | None = None
    stop_on_picture = bool(getattr(args, "stop_on_picture", True))

    try:
        try:
            state0, live0, lock_note = wait_settle(
                http, freq_hz=freq_hz, dwell_s=dwell_s, extra_s=0.0,
                fresh_s=fresh_s, before=None, need_video=True,
            )
        except SweepHttpError as exc:
            print(f"POST /api/lock/{int(round(freq_hz))}?force=1 did not enter LOCK: {exc}",
                  file=sys.stderr)
            return 2
        m0 = extract_metrics(state0 if isinstance(state0, dict) else {}, live0)
        mode0 = str(m0.get("mode") or "").upper()
        tgt0 = m0.get("lock_target")
        on_freq = tgt0 is not None and abs(float(tgt0) - freq_hz) < 1e6
        if mode0 != "LOCK" or not on_freq:
            msg = lock_note or diagnose_state(state0, freq_hz) or (
                f"mode={mode0!s} lock_target={tgt0}"
            )
            print(
                f"POST /api/lock/{int(round(freq_hz))}?force=1 did not enter LOCK: {msg}",
                file=sys.stderr,
            )
            row = row_from_trial(
                trial_i=-1, freq_hz=freq_hz, patch={}, metrics=m0,
                error=msg, http_status=200, frame_path=None, luma=(None, None),
                note=msg,
            )
            rows.append(row)
            return 2
        prev_ref = m0.get("last_frame_ref")
        prev_metrics = m0
        if video_missing(state0):
            print(
                f"LOCK ok but no video yet: {lock_note or diagnose_state(state0, freq_hz)}",
                file=sys.stderr,
            )

        extra_trial = paired_snow_trial(m0, freq_hz, live_values)
        if extra_trial:
            trials = [trials[0], extra_trial, *trials[1:]]
            print("  [paired] low-band BW×sample_rate (baseline snow)")

        for i, trial in enumerate(trials):
            patch: dict[str, Any] = dict(trial.get("values") or {})
            label = str(trial.get("label") or f"trial-{i}")
            err: str | None = None
            status: int | None = 200
            metrics: dict[str, Any] = {}
            frame_path: Path | None = None
            luma: tuple[float | None, float | None] = (None, None)
            note: str | None = None
            stale = False
            try:
                wanted = {k: live_values[k] for k in restore_keys}
                wanted.update(patch)
                diff = {
                    k: v for k, v in wanted.items()
                    if not values_close(engine_vals.get(k), v)
                }
                extra = 8.0 if "video.sample_rate" in diff else 0.0
                if diff:
                    try:
                        applied = apply_parameters(http, diff)
                        status = int(applied.get("http_status") or 200)
                        engine_vals.update(diff)
                        pending = applied.get("pending_keys") or []
                        if pending:
                            note = f"pending={pending}"
                    except SweepHttpError as exc:
                        err = str(exc)
                        status = exc.status
                        row = row_from_trial(
                            trial_i=i, freq_hz=freq_hz, patch=patch, metrics={},
                            error=err, http_status=status, frame_path=None,
                            luma=(None, None), note=note,
                        )
                        rows.append(row)
                        print(f"  [{i:03d}] FAIL {label}: {err}")
                        continue

                tgt_prev = None if prev_metrics is None else prev_metrics.get("lock_target")
                try:
                    relock_note = relock_if_drifted(http, freq_hz, tgt_prev)
                except SweepHttpError as exc:
                    relock_note = f"re-lock failed: {exc}"
                if relock_note:
                    bits_pre = [n for n in (note, relock_note) if n]
                    note = "; ".join(bits_pre) or None

                try:
                    state, live, settle_note = wait_settle(
                        http, freq_hz=freq_hz, dwell_s=dwell_s, extra_s=extra,
                        fresh_s=fresh_s,
                        before=prev_metrics if diff else None,
                        need_video=True,
                    )
                except SweepHttpError as exc:
                    state, live, settle_note = {}, None, None
                    err = (err + "; " if err else "") + str(exc)
                    status = status or exc.status

                metrics = extract_metrics(
                    state if isinstance(state, dict) else {}, live,
                )
                if lock_target_drifted(metrics.get("lock_target"), freq_hz):
                    try:
                        walked = relock_if_drifted(
                            http, freq_hz, metrics.get("lock_target"),
                        )
                    except SweepHttpError as exc:
                        walked = f"re-lock failed: {exc}"
                    if walked:
                        note = "; ".join(n for n in (note, walked) if n) or None
                        try:
                            state, live, settle_note = wait_settle(
                                http, freq_hz=freq_hz, dwell_s=dwell_s,
                                extra_s=0.0, fresh_s=fresh_s,
                                before=None, need_video=True,
                            )
                            metrics = extract_metrics(
                                state if isinstance(state, dict) else {}, live,
                            )
                        except SweepHttpError as exc:
                            err = (err + "; " if err else "") + str(exc)

                bits = [n for n in (note, settle_note) if n]
                if video_missing(state if isinstance(state, dict) else {}):
                    bits.append(diagnose_state(
                        state if isinstance(state, dict) else {}, freq_hz,
                    ) or "no video key")
                note = "; ".join(bits) or None
                if diff and prev_metrics and (
                    metrics_fingerprint(metrics) == metrics_fingerprint(prev_metrics)
                ):
                    stale = True
                    if not note or "unchanged" not in note:
                        bits.append("metrics unchanged after PUT")
                        note = "; ".join(b for b in bits if b)
                if args.save_frames and not err:
                    try:
                        frame_path = maybe_save_frame(
                            http,
                            out_dir=out_dir,
                            install_root=install_root,
                            prev_ref=prev_ref,
                            timeout_s=still_wait_s(dwell_s, metrics.get("fps")),
                        )
                    except Exception as exc:
                        err = (err + "; " if err else "") + f"save-frame: {exc}"
                    if frame_path is not None:
                        prev_ref = metrics.get("last_frame_ref") or prev_ref
                        try:
                            luma = luma_stats(frame_path)
                        except Exception:
                            luma = (None, None)
                    else:
                        prev_ref = metrics.get("last_frame_ref") or prev_ref
                        if args.save_frames and not err and i == 0:
                            skip = still_skip_note(metrics, False)
                            if skip:
                                bits.append(skip)
                                note = "; ".join(b for b in bits if b)
                if metrics:
                    prev_metrics = metrics
            except Exception as exc:
                err = (err + "; " if err else "") + str(exc)
                if status is None:
                    status = 0
            row = row_from_trial(
                trial_i=i, freq_hz=freq_hz, patch=patch, metrics=metrics,
                error=err, http_status=status,
                frame_path=None if frame_path is None else str(frame_path),
                luma=luma, stale=stale, note=note,
            )
            rows.append(row)
            if err and not metrics:
                print(f"  [{i:03d}] FAIL {label}: {err}")
            else:
                _print_trial_line(i, label, row)

            if stop_on_picture and _picture_ok(metrics, freq_hz):
                if not patch:
                    print("baseline already analog, stopping.")
                else:
                    print(f"stop-on-picture at trial {i} {label}")
                break
    finally:
        # Flush records before restore PUT — a restore hang/crash must not
        # leave "N trials / 0 rows".
        try:
            jsonl, csv_path = write_outputs(out_dir, rows)
            print(f"wrote {jsonl}  ({len(rows)} rows)")
            print(f"wrote {csv_path}")
        except OSError as exc:
            print(f"write failed: {exc}", file=sys.stderr)
        finally:
            if rows:
                print(format_correlation(correlate_knobs(rows)))
        if restore_keys and not args.dry_run:
            baseline = {k: live_values[k] for k in restore_keys if k in live_values}
            keep = getattr(args, "keep_best", False)
            patch = best_trial_patch(rows) if keep else None
            if keep and patch is not None:
                wanted = dict(baseline)
                wanted.update(patch)
                try:
                    apply_parameters(http, wanted)
                    print(
                        "kept best trial knobs (live PUT, yaml untouched): "
                        + (json.dumps(patch) if patch else "baseline")
                    )
                except SweepHttpError as exc:
                    print(f"keep-best failed: {exc}", file=sys.stderr)
            elif not args.no_restore and baseline:
                try:
                    apply_parameters(http, baseline)
                    print("restored baseline knobs (live PUT, yaml untouched)")
                except SweepHttpError as exc:
                    print(f"restore failed: {exc}", file=sys.stderr)
            if not keep:
                late = mid_sweep_signal_note(rows)
                if late:
                    print(late)
                else:
                    winner = analog_winner_patch(rows)
                    if winner:
                        print(format_winning_put(winner, args.base))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="OFAT analog LOCK knob sweep via fpvscan HTTP (no web UI).",
    )
    ap.add_argument(
        "--freq", required=True,
        help="LOCK frequency this session (Hz or MHz). "
             "Use Rabbit OSD now — frequencies do not repeat.",
    )
    ap.add_argument("--out", default=str(Path.home() / "knob-sweep"),
                    help="directory for JSONL/CSV (and frames/)")
    ap.add_argument(
        "--base", default=None,
        help=(
            "fpvscan HTTP origin. Default: $FPVSCAN_BASE, then "
            "http://127.0.0.1:<web.port>, then config web.host if it is a "
            "specific IP (not 0.0.0.0), then local IPv4s. "
            f"If loopback is refused, pass --base {OPERATOR_BASE_HINT}."
        ),
    )
    ap.add_argument(
        "--group", action="append", default=None,
        choices=sorted(GROUPS),
        help="knob group (repeatable). Default: lock (~30 analog-picture trials). "
             "decode = H-PLL/phase/width; cvbs = average/sharpen/crop. "
             "Do not pass scan FFT/threshold keys.",
    )
    ap.add_argument("--keys", default="",
                    help="extra catalog keys, comma-separated")
    ap.add_argument(
        "--dwell-ms", type=float, default=DEFAULT_DWELL_MS,
        help="settle time after lock/PUT before sampling "
             f"(default {int(DEFAULT_DWELL_MS)}; old default was 2500)",
    )
    ap.add_argument(
        "--fresh-ms", type=float, default=DEFAULT_FRESH_MS,
        help="after dwell, wait this long for pic_score/line_rate to change "
             f"(default {int(DEFAULT_FRESH_MS)}; 0 = sample immediately; "
             "sample_rate still adds extra dwell). Analog lock after 400 ms "
             "ends the wait early.",
    )
    ap.add_argument(
        "--stop-on-picture",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="stop remaining OFAT after the first locked row with "
             f"corr≥{ANALOG_CORR_FLOOR} (default on). "
             "--no-stop-on-picture runs the full grid.",
    )
    ap.add_argument(
        "--save-frames", "--save-frame", action="store_true", dest="save_frames",
        help="POST /api/snapshot and copy last_frame_ref into --out "
             "(keep as one token; --save-frame is an alias)",
    )
    ap.add_argument("--install-root", default="/opt/fpvscan",
                    help="Pi install dir for resolving last_frame_ref")
    ap.add_argument("--http-timeout", type=float, default=15.0)
    ap.add_argument("--dry-run", action="store_true",
                    help="print trial list after reading current parameters")
    ap.add_argument("--no-restore", action="store_true",
                    help="leave last trial knobs in live engine.cfg")
    ap.add_argument(
        "--keep-best", action="store_true",
        help="after the sweep, PUT the winning trial (highest row_corr) "
             "instead of restoring the pre-sweep baseline",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    sanitize_import_path()
    ap = build_parser()
    args = ap.parse_args(argv)
    if not args.group:
        args.group = ["lock"]
    try:
        return run(args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
