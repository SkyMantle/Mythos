"""YAML coerce must not eat operator strings; shipped rec_fps is not the code default."""
from __future__ import annotations

from pathlib import Path

from fpvscan import config


def test_coerce_keeps_driver_and_preset_strings():
    raw = {
        "sdr": {"driver": "bladerf", "gain_db": "35"},
        "video": {"rec_preset": "veryfast", "ffmpeg_path": "", "width": "640"},
    }
    out = config._coerce(raw)
    assert out["sdr"]["driver"] == "bladerf"
    assert out["sdr"]["gain_db"] == 35.0
    assert out["video"]["rec_preset"] == "veryfast"
    assert out["video"]["ffmpeg_path"] == ""
    assert out["video"]["width"] == 640.0


def test_coerce_numeric_strings_in_lists():
    out = config._coerce({"scan": {"pts": ["400.0e6", "not-a-number"]}})
    assert out["scan"]["pts"][0] == 400.0e6
    assert out["scan"]["pts"][1] == "not-a-number"


def test_shipped_rec_fps_is_24_not_engine_default_5():
    """VideoRecorder fps=5 if the yaml key is dropped — files play five times too slow."""
    cfg = config.load(Path(__file__).resolve().parents[1] / "config.yaml")
    assert int(cfg["video"]["rec_fps"]) == 24
    assert int(cfg["video"]["rec_height"]) == 288
    assert cfg["video"]["rec_preset"] == "veryfast"
    assert int(cfg["video"]["stream_quality"]) == 60
    assert isinstance(cfg["video"]["rec_preset"], str)
