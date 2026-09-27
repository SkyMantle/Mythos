"""doctor.py step 2 must still name bladeRF-cli when no library is on disk."""
from __future__ import annotations

from pathlib import Path


SRC = (Path(__file__).resolve().parents[1]
       / "scripts" / "doctor.py").read_text(encoding="utf-8")


def test_lib_miss_names_bladerf_cli_info():
    """An empty search is 'the Nuand installer was never run' or
    BLADERF_LIB_DIR points at the wrong folder. If the hint drops
    `bladeRF-cli -e info`, the next hour is spent reinstalling
    Python / ctypes instead of checking whether the CLI itself
    can see the DLL. Step 5 (board not seen) is a different
    failure — the library already loaded.
    """
    start = SRC.index('print("\\n2. пошук бібліотеки на диску")')
    end = SRC.index('print("\\n3. залежності поруч")')
    body = SRC[start:end]
    assert "не знайдено жодного файлу" in body
    assert "bladeRF-cli -e info" in body
    assert "return 1" in body
    assert "--lib" in body
