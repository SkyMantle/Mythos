"""headless.py --driver must override YAML so sim/file work without a board."""
from __future__ import annotations

from pathlib import Path


SRC = (Path(__file__).resolve().parents[1]
       / "scripts" / "headless.py").read_text(encoding="utf-8")


def test_headless_driver_flag_overrides_yaml():
    """Shipped config.yaml is `driver: bladerf`. `py scripts/headless.py
    --driver sim` is the documented no-hardware path; without the
    override the script still opens libbladeRF and dies before the engine
    thread starts, which looks like 'headless is broken'.
    """
    assert 'ap.add_argument("--driver", help="bladerf | sim | file")' in SRC
    assert 'if a.driver:' in SRC
    assert 'cfg["sdr"]["driver"] = a.driver' in SRC


def test_headless_defaults_to_shipped_config_yaml():
    """A relative default other than config.yaml loads an empty dict
    (or a laptop copy) and the scan range / USB knobs are not the ones
    the operator just edited.
    """
    assert 'ap.add_argument("-c", "--config", default="config.yaml")' in SRC
