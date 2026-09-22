"""bringup.py is the stand script: sync diagnostic + occupancy-shape loop."""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import numpy as np

from fpvscan.sdr.sim import _cvbs


ROOT = Path(__file__).resolve().parents[1]


def _bringup():
    p = ROOT / "scripts" / "bringup.py"
    spec = importlib.util.spec_from_file_location("bringup_mod", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_sync_report_finds_pal_line_rate(capsys):
    """When decode() fails, this is what the operator reads. A flat
    signal must stay silent; a clean PAL train must print ~15 625 Hz.
    """
    br = _bringup()
    fs = 8e6
    t = np.arange(int(fs * 0.04)) / fs
    br._sync_report(_cvbs(t, 15625.0), fs)
    out = capsys.readouterr().out
    assert "фронтів" in out
    assert "рівних інтервалів" in out
    hz = [int(x) for x in re.findall(r"= (\d+) Гц", out)]
    # printed as int(fs/med); sim blanking pulls the median a bin off 15625
    assert any(14_000 < h < 17_500 for h in hz)

    br._sync_report(np.full(int(fs * 0.04), 0.3, dtype=np.float32), fs)
    flat = capsys.readouterr().out
    assert flat == ""


def test_sync_report_says_few_edges_on_noise(capsys):
    br = _bringup()
    fs = 2e6
    rng = np.random.default_rng(0)
    # Very quiet noise: few threshold crossings after normalize
    v = rng.normal(0.5, 0.01, int(fs * 0.02)).astype(np.float32)
    br._sync_report(v, fs)
    out = capsys.readouterr().out
    assert "мало" in out or "фронтів" in out


def test_shape_loop_uses_five_thresholds():
    """Analog video is a plateau: width barely grows as the threshold
    drops. A single 8 dB occupancy (engine default) cannot tell CW from
    FM. The 15 MHz neighbor window keeps Band-F from contaminating F4.
    """
    src = (ROOT / "scripts" / "bringup.py").read_text(encoding="utf-8")
    assert "for th in (3, 6, 10, 20, 30):" in src
    assert "min_bw_hz=0.3e6" in src
    assert "abs(z.center_hz - top.center_hz) < 15e6" in src
