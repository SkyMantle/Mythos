"""Build the sweep/lock parameter catalog from live config + YAML comments."""
from __future__ import annotations

import json
import re
from typing import Any

from fpvscan.app.domain.exceptions import ValidationError
from fpvscan.app.domain.models import ParamSpec
from fpvscan.app.services.catalog_tasks import catalog_affects, catalog_modes, catalog_task

_COMMENT = re.compile(
    r"^(?P<indent>\s*)(?P<key>[A-Za-z_][\w]*)\s*:\s*(?P<rest>.*?)(?:\s+#\s*(?P<cmt>.*))?$"
)
_SKIP = frozenset({
    "sdr.driver", "sdr.device", "sdr.lib_path", "sdr.path",
    "sdr.rx_channel", "sdr.quick_tune", "sdr.num_buffers",
    "sdr.buffer_size", "sdr.num_transfers", "video.ffmpeg_path",
    "web.host", "web.port",
})
_GROUP = {"scan": "sweep", "video": "lock", "sdr": "shared"}
_SECTIONS = ("scan", "video", "sdr")
_NAMED_BOUNDS: dict[str, tuple[float, float, float]] = {
    "fft_size": (256, 65536, 1),
    "averages": (1, 64, 1),
    "confirm_hits": (1, 10, 1),
    "min_harmonics": (0, 8, 1),
    "inspect_min_lines": (10, 400, 1),
    "stream_quality": (1, 100, 1),
    "stream_method": (0, 6, 1),
    "rec_fps": (1, 60, 1),
    "rec_height": (120, 1080, 1),
    "rec_crf": (0, 51, 1),
    "width": (160, 1920, 1),
    "spectrum_every": (1, 256, 1),
    "hunt_every": (1, 200, 1),
    "hunt_after_lock": (0, 200, 1),
    "average": (0, 40, 0.5),
    "motion_thresh": (1, 128, 1),
    "sharpen": (0, 2.0, 0.05),
    "ring_seconds": (0.1, 5.0, 0.1),
    "track_window_margin": (1.0, 3.0, 0.05),
    "auto_peek_secs": (0, 120, 1),
    "auto_peek_cooldown_s": (0, 600, 1),
    "gain_db": (0, 60, 1),
    "bias_tee_gain_offset_db": (0, 40, 1),
    "settle_us": (0, 10_000, 50),
    "afc_gain": (0, 1, 0.05),
    "afc_deadband_hz": (0, 500_000, 1_000),
    "afc_digital_max_hz": (50_000, 3_000_000, 50_000),
    "afc_max_step_hz": (1_000, 500_000, 5_000),
    "afc_digital_headroom_hz": (0, 3_000_000, 50_000),
    "h_phase_frac": (-0.5, 0.5, 0.01),
    "crop_left_frac": (0.0, 0.25, 0.01),
    "crop_bottom_lines": (0, 24, 1),
    "hunt_min_gain": (0, 1, 0.01),
    "hunt_skip_if_score": (0, 1, 0.01),
    "hunt_drop": (0, 1, 0.01),
    "min_confidence": (0, 1, 0.01),
    "inspect_min_row_corr": (0, 1, 0.01),
    "inspect_conf_bypass": (0, 1, 0.01),
}
_NAMED_ENUMS: dict[str, list[str]] = {
    "rec_preset": [
        "ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow",
    ],
    "cluster_step_mhz": ["off", "12", "8", "4"],
    "hit_filter": ["all", "hide_weak", "hide_no_video", "hide_near_dup"],
}
_NAMED_LABELS: dict[str, str] = {
    "cluster_step_mhz": "Крок кластера",
    "hit_filter": "Фільтр знахідок",
}
_ENUM_LABELS: dict[str, dict[str, str]] = {
    "cluster_step_mhz": {
        "off": "вимк. (грубо)",
        "12": "12 МГц",
        "8": "8 МГц",
        "4": "4 МГц",
    },
    "hit_filter": {
        "all": "усі",
        "hide_weak": "ховати слабкі",
        "hide_no_video": "ховати без картинки",
        "hide_near_dup": "ховати сусідів",
    },
}


def extract_yaml_comments(text: str) -> dict[str, str]:
    comments: dict[str, str] = {}
    section = ""
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        match = _COMMENT.match(raw.rstrip())
        if match is None:
            continue
        key, cmt, indent = match.group("key"), match.group("cmt"), match.group("indent")
        if not indent:
            section = key
            continue
        if section and cmt:
            comments[f"{section}.{key}"] = cmt.strip()
    return comments


def _leaf(key: str) -> str:
    return key.rsplit(".", 1)[-1]


def _humanize(name: str) -> str:
    for suffix in ("_hz", "_mhz", "_db", "_ms", "_us", "_secs"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name.replace("_", " ").strip().title()


def _unit(name: str) -> str | None:
    if name.endswith("_hz"):
        return "Hz"
    if name.endswith("_mhz"):
        return "MHz"
    if name.endswith("_db") or name == "gain_db":
        return "dB"
    if name.endswith("_ms"):
        return "ms"
    if name.endswith("_us"):
        return "µs"
    if name.endswith("_secs") or name.endswith("_cooldown_s"):
        return "s"
    if name in {"width", "rec_height"}:
        return "px"
    if name == "rec_fps":
        return "fps"
    return None


def _bounds(name: str, value: Any) -> tuple[float | None, float | None, float | None]:
    if name in _NAMED_ENUMS:
        return None, None, None
    if name in _NAMED_BOUNDS:
        return _NAMED_BOUNDS[name]
    if name.endswith("_hz"):
        if name in {"start_hz", "stop_hz"}:
            return 100e6, 6.2e9, 1e6
        if "sample_rate" in name:
            return 1e6, 61.44e6, 1e4
        if "line_tol" in name:
            return 1, 2000, 1
        return 0, 80e6, 1e3
    if name.endswith("_db"):
        return -20, 80, 0.1
    if name.endswith("_ms"):
        return 0, 500, 1
    if name.endswith("_us"):
        return 0, 10_000, 10
    if name.endswith("_mhz"):
        return -40, 40, 0.05
    if isinstance(value, bool) or isinstance(value, str) or isinstance(value, list):
        return None, None, None
    if isinstance(value, int) and not isinstance(value, bool):
        return 0, 1_000_000, 1
    if isinstance(value, float):
        return 0, 1e12, 0.01
    return None, None, None


def _enum_token(name: str, value: Any) -> Any:
    allowed = _NAMED_ENUMS.get(name)
    if not allowed or value is None:
        return value
    if isinstance(value, bool):
        text = str(value)
    elif isinstance(value, float) and value.is_integer():
        text = str(int(value))
    elif isinstance(value, int):
        text = str(value)
    else:
        text = str(value).strip()
    return text if text in allowed else value


def _typ(name: str, value: Any) -> str:
    if name in _NAMED_ENUMS:
        return "enum"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, list):
        return "string"
    return "string"


def _iter_keys(cfg: dict[str, Any]) -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    for section in _SECTIONS:
        block = cfg.get(section)
        if not isinstance(block, dict):
            continue
        for name, value in block.items():
            key = f"{section}.{name}"
            if key in _SKIP:
                continue
            out.append((key, value))
    return out


def build_catalog(
    current_cfg: dict[str, Any],
    default_cfg: dict[str, Any],
    yaml_text: str = "",
) -> list[ParamSpec]:
    comments = extract_yaml_comments(yaml_text)
    defaults = {k: v for k, v in _iter_keys(default_cfg)}
    specs: list[ParamSpec] = []
    for key, current in _iter_keys(current_cfg):
        name = _leaf(key)
        section = key.split(".", 1)[0]
        group = _GROUP[section]
        applies = ["sweep", "lock"] if group == "shared" else [group]
        lo, hi, step = _bounds(name, current)
        current = _enum_token(name, current)
        default = _enum_token(name, defaults.get(key, current))
        specs.append(ParamSpec(
            key=key,
            label=_NAMED_LABELS.get(name, _humanize(name)),
            group=group,
            type=_typ(name, current),
            unit=_unit(name),
            min=lo,
            max=hi,
            step=step,
            default=default,
            current=current,
            enum_values=list(_NAMED_ENUMS[name]) if name in _NAMED_ENUMS else None,
            description=comments.get(key, ""),
            applies_to=applies,
            modes=catalog_modes(key, applies),
            task=catalog_task(key),
            affects=catalog_affects(key),
            enum_labels=dict(_ENUM_LABELS[name]) if name in _ENUM_LABELS else None,
        ))
    return specs


def catalog_values(cfg: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in _iter_keys(cfg)}


def coerce_param(spec: ParamSpec, raw: Any) -> Any:
    if raw is None:
        raise ValidationError(f"{spec.key}: value is required", {spec.key: "required"})
    if spec.type == "bool":
        if isinstance(raw, bool):
            return raw
        if raw in (0, 1, "0", "1", "true", "false", "True", "False"):
            return raw in (1, "1", "true", "True")
        raise ValidationError(f"{spec.key}: expected bool", {spec.key: "type"})
    if spec.type == "enum":
        allowed = spec.enum_values or []
        if isinstance(raw, bool):
            text = str(raw)
        elif isinstance(raw, float) and raw.is_integer():
            text = str(int(raw))
        elif isinstance(raw, int):
            text = str(raw)
        else:
            text = str(raw).strip()
        if text not in allowed:
            raise ValidationError(f"{spec.key}: not in {allowed}", {spec.key: "enum"})
        return text
    if spec.type == "string":
        if isinstance(raw, list):
            return raw
        if isinstance(raw, str):
            text = raw.strip()
            if text.startswith("["):
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValidationError(
                        f"{spec.key}: invalid JSON array", {spec.key: "type"}
                    ) from exc
                if not isinstance(parsed, list):
                    raise ValidationError(f"{spec.key}: expected array", {spec.key: "type"})
                return parsed
            return text
        raise ValidationError(f"{spec.key}: expected string", {spec.key: "type"})
    if spec.type == "int":
        if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
            raise ValidationError(f"{spec.key}: expected int", {spec.key: "type"})
        if isinstance(raw, float) and not raw.is_integer():
            raise ValidationError(f"{spec.key}: expected int", {spec.key: "type"})
        try:
            value = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"{spec.key}: expected int", {spec.key: "type"}) from exc
        _check_range(spec, value)
        return value
    if spec.type == "float":
        if isinstance(raw, bool):
            raise ValidationError(f"{spec.key}: expected float", {spec.key: "type"})
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"{spec.key}: expected float", {spec.key: "type"}) from exc
        _check_range(spec, value)
        return value
    raise ValidationError(f"{spec.key}: unsupported type", {spec.key: "type"})


def _check_range(spec: ParamSpec, value: float) -> None:
    if spec.min is not None and value < spec.min:
        raise ValidationError(
            f"{spec.key}: {value} < min {spec.min}", {spec.key: "min"}
        )
    if spec.max is not None and value > spec.max:
        raise ValidationError(
            f"{spec.key}: {value} > max {spec.max}", {spec.key: "max"}
        )
