"""Shipped YAML / startup contracts that starve LOCK or miss a VTx."""
from __future__ import annotations

import inspect
from pathlib import Path

from fpvscan.config import load
from fpvscan.engine import Engine
from fpvscan.sdr.factory import make_source

ROOT = Path(__file__).resolve().parents[1]
CFG = load(ROOT / "config.yaml")


def test_capture_fits_inside_the_iq_ring_and_covers_a_field():
    """One PAL/NTSC field is ~16–20 ms. capture_ms below that, or a ring
    shorter than one capture, makes every LOCK snapshot short-circuit
    and the console never gets a frame.
    """
    cap_s = float(CFG["video"]["capture_ms"]) / 1000
    ring_s = float(CFG["video"]["ring_seconds"])
    assert CFG["video"]["capture_ms"] >= 40
    assert ring_s > cap_s


def test_threshold_stays_low_enough_for_a_weak_vtx():
    """#5 only checks the key is numeric. Raising threshold_db to a
    'cleaner' 12 dB drops the bench VTx that sits ~6 dB over the floor.
    """
    assert 0 < CFG["scan"]["threshold_db"] <= 6
    assert CFG["scan"]["dc_notch_hz"] == 200e3


def test_factory_defaults_to_bladerf_not_sim():
    """`cfg.get("driver", ...)` falling through to sim would boot a Pi
    into a fake scene with no error. Instantiating BladeRF needs the
    library — pin the default in source.
    """
    src = inspect.getsource(make_source)
    assert 'cfg.get("driver", "bladerf")' in src
    assert CFG["sdr"]["driver"] == "bladerf"


def test_run_uses_a_bounded_event_queue():
    """_emit evicts old spectra to keep a frame. An unbounded Queue
    grows without a browser; maxsize=1 fights frames against spectra.
    """
    text = (ROOT / "run.py").read_text(encoding="utf-8")
    assert "Queue(maxsize=64)" in text


def test_run_finally_releases_radio_and_recorder():
    """Leaving the reader/ffmpeg/USB handle open after a crash keeps
    the bladeRF busy and the next start fails with a mystery 'device
    busy' — the finally on _run is the only closer.
    """
    src = inspect.getsource(Engine._run)
    assert "self._stop_reader()" in src
    assert "self._rec_stop()" in src
    assert "self.src.close()" in src
    assert "finally:" in src
