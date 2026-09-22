"""Headless is the 'radio vs web' isolator; a dead engine must not hang."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_headless_breaks_when_the_engine_thread_dies():
    """Queue maxsize=2000 is already covered. Without the liveness
    check, a USB exception that kills _run() leaves the operator
    staring at an empty console until --secs elapses.
    """
    src = (ROOT / "scripts" / "headless.py").read_text(encoding="utf-8")
    assert "eng._thread.is_alive()" in src
    assert "МЕРТВА" in src
    assert "if not alive:" in src
    body = src[src.index("if not alive:"):]
    assert "break" in body.split("\n", 3)[1] or "break" in body.split("\n", 4)[2]
    assert "Робоча нитка померла" in src


def test_doctor_lib_flag_skips_the_search_path():
    """A broken PATH libbladeRF.so.2 must not win over an explicit --lib.
    doctor also stops (return 1) when nothing is on disk — it must not
    proceed to bladerf_open and print a confusing FPGA failure.
    """
    src = (ROOT / "scripts" / "doctor.py").read_text(encoding="utf-8")
    assert 'ap.add_argument("--lib"' in src
    assert "files = [a.lib] if a.lib else B.find_lib_files()" in src
    assert "return 1" in src
    assert "не знайдено жодного файлу" in src
