"""bench_retune.py must release the exclusive bladeRF handle."""
from __future__ import annotations

from pathlib import Path


SRC = (Path(__file__).resolve().parents[1]
       / "scripts" / "bench_retune.py").read_text(encoding="utf-8")


def test_bench_closes_after_the_three_sections():
    """`BladeRF.open` takes the board exclusively. A bench run
    that returns without `close()` leaves doctor / bringup /
    `run.py` failing with 'device busy' until the process exits
    — and the script's `main()` is what systemd-less operators
    run from an ssh session they keep open. Close must come
    after section 3 (the capture timing loop), not only on the
    BladeRFError path.
    """
    assert "src.close()" in SRC
    assert SRC.rfind("src.close()") > SRC.find("зривів потоку")
    assert SRC.rfind("src.close()") > SRC.find("prime_quick_tune")
    # Error path still names the radio, but close is in the happy path.
    assert "except BladeRFError" in SRC
