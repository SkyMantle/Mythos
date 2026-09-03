"""YAML 1.1 leaves `400.0e6` as a string; scan math then silently misbehaves."""
from __future__ import annotations

from pathlib import Path

from fpvscan.config import load


def test_load_coerces_exponent_without_sign(tmp_path: Path):
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "scan:\n"
        "  start_hz: 400.0e6\n"
        "  stop_hz: 6.0e9\n"
        "  nested:\n"
        "    - 20.0e6\n"
        "    - keep\n"
        "video:\n"
        "  afc_limit_hz: 8.0e+6\n",
        encoding="utf-8",
    )
    cfg = load(p)
    assert cfg["scan"]["start_hz"] == 400e6
    assert isinstance(cfg["scan"]["start_hz"], float)
    assert cfg["scan"]["stop_hz"] == 6e9
    assert cfg["scan"]["nested"][0] == 20e6
    assert cfg["scan"]["nested"][1] == "keep"
    assert cfg["video"]["afc_limit_hz"] == 8e6


def test_shipped_config_yaml_is_numeric():
    cfg = load(Path(__file__).resolve().parents[1] / "config.yaml")
    for key in ("start_hz", "stop_hz", "sample_rate", "channel_bw_hz",
                "threshold_db", "min_bw_hz", "dc_notch_hz"):
        assert isinstance(cfg["scan"][key], (int, float)), key
        assert cfg["scan"][key] != ""
    assert cfg["scan"]["start_hz"] < cfg["scan"]["stop_hz"]
    assert isinstance(cfg["video"]["sample_rate"], (int, float))
    assert isinstance(cfg["video"]["afc_limit_hz"], (int, float))
