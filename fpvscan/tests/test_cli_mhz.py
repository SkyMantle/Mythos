"""Operator-facing contracts: MHz on the CLI, Hz on the radio."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_headless_lock_flag_is_mhz():
    """`py scripts/headless.py --lock 5800` must tune 5800e6 Hz.

    Treating the flag as Hz would LOCK onto 5.8 kHz and the operator
    would see snow on a frequency they just typed from the hit list.
    """
    src = (ROOT / "scripts" / "headless.py").read_text(encoding="utf-8")
    assert 'eng.command("lock", freq_hz=a.lock * 1e6)' in src
    assert 'help="одразу стати на частоту, МГц"' in src


def test_headless_debug_drops_confirm_hits_to_one():
    """Debug mode must emit every inspect decision. Leaving confirm_hits
    at 2 (shipped) means the console stays empty while candidates print.
    """
    src = (ROOT / "scripts" / "headless.py").read_text(encoding="utf-8")
    assert 'cfg["scan"]["debug_candidates"] = True' in src
    assert 'cfg["scan"]["confirm_hits"] = 1' in src


def test_record_script_freq_and_rate_are_mhz():
    """`scripts/record.py -f 5800 -s 40` is MHz / Msps, same as the
    operator notebook. Hz here would ask the bladeRF for 5.8 kHz.
    """
    src = (ROOT / "scripts" / "record.py").read_text(encoding="utf-8")
    assert "fc, fs = a.freq * 1e6, a.rate * 1e6" in src
    assert 'help="МГц"' in src
    assert "Мвідл/с" in src
