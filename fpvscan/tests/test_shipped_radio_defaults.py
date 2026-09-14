"""Operator-tuned radio / web defaults that are not the code fallbacks."""
from __future__ import annotations

import inspect
from pathlib import Path

from fpvscan.config import load
from fpvscan.web.server import create_app

ROOT = Path(__file__).resolve().parents[1]
CFG = load(ROOT / "config.yaml")


def test_settle_and_gain_are_the_bench_values_not_code_defaults():
    """BladeRF.__init__ defaults settle_us=400 and gain_db=30. The
    bench YAML is 500 µs / 35 dB — cutting settle drops 5.8 GHz
    snapshots on the PLL transient; cutting gain hides a weak VTx.
    """
    assert CFG["sdr"]["settle_us"] == 500
    assert CFG["sdr"]["gain_db"] == 35
    assert CFG["sdr"]["settle_us"] > 400


def test_priority_bands_are_off_so_the_sweep_is_linear():
    """priority_bands=True interleaves 433/900/1G2/2G4/3G3/5G8 into
    every step and doubles USB retunes. Shipped false is a full
    400 MHz–6 GHz pass; flipping it silently halves dwell per step.
    """
    assert CFG["scan"]["priority_bands"] is False


def test_console_binds_all_interfaces():
    """host 127.0.0.1 on a headless Pi is only reachable from the
    board itself. 0.0.0.0 is what run.py then prints as 127.0.0.1
    for the operator's browser on the LAN.
    """
    assert CFG["web"]["host"] in ("0.0.0.0", "::")
    assert int(CFG["web"]["port"]) == 8080


def test_record_route_only_start_begins_a_clip():
    """Console posts /api/record/start|stop. Treating any truthy
    token as start (or 'on' like bias-tee) leaves ffmpeg running
    after Стоп, or never starts after Запис.
    """
    src = inspect.getsource(create_app)
    assert 'engine.command("rec_start" if on == "start" else "rec_stop")' in src
