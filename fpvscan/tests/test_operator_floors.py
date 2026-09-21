"""Operator scripts: weak-signal floors, bench overlap, bringup PNG."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_record_warns_when_peak_is_below_the_noise_floor():
    """clip_frac catches saturation. The other failure mode is an empty
    .cf32 of band noise: peak below -35 dBFS, replay looks like 'no VTx'
    and the operator debugs the detector. record.py must say so.
    """
    src = (ROOT / "scripts" / "record.py").read_text(encoding="utf-8")
    assert "peak < -35" in src
    assert "сигнал надто слабкий" in src
    assert "--agc" in src


def test_bench_retune_step_overlaps_like_a_sweep():
    """`step = fs` parks a 20 MHz VTx on the stitch of two hops, so the
    full-pass time looks fine while the real sweep (0.9 fs − ch/2) is
    slower. 0.8 fs is the overlap used to count hops in 400 MHz–6 GHz.
    """
    src = (ROOT / "scripts" / "bench_retune.py").read_text(encoding="utf-8")
    assert "step = fs * 0.8" in src
    assert "arange(400e6 + fs / 2, 6000e6, step)" in src


def test_bringup_png_is_vga_and_save_writes_a_sidecar():
    """Native PAL-field PNG is ~640×232; operators compare it to a 480-line
    screenshot from the console. --save must go through write_capture so
    FileSource can replay the same grab.
    """
    src = (ROOT / "scripts" / "bringup.py").read_text(encoding="utf-8")
    assert 'default="bringup.png"' in src
    assert 'Image.fromarray(fr.luma, "L").resize((640, 480)).save(a.out)' in src
    assert "write_capture(a.save, x, fc, fs" in src
