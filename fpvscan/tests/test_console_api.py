"""Console contracts that send the wrong unit or HTTP verb silently fail LOCK/record."""
from __future__ import annotations

from pathlib import Path

HTML = (
    Path(__file__).resolve().parents[1] / "fpvscan" / "web" / "static" / "index.html"
).read_text(encoding="utf-8")


def test_detection_click_locks_hz_not_mhz():
    """Hit list stores freq_hz. Multiplying by 1e6 (as the MHz box does) would lock THz."""
    assert "n.onclick = () => lock(parseFloat(n.dataset.f))" in HTML
    assert "lock(parseFloat(n.dataset.f) * 1e6)" not in HTML
    assert 'data-f="${d.freq_hz}"' in HTML


def test_record_uses_start_stop_bias_uses_on_off():
    """Server maps record/{on} with on=='start'; bias_tee/{action} with action=='on'.
    Swapping the vocabularies means Запис never starts and bias-tee never enables."""
    assert "'/api/record/' + (recording ? 'stop' : 'start')" in HTML
    assert "'/api/bias_tee/' + (on ? 'on' : 'off')" in HTML
    assert "/api/record/' + (recording ? 'off' : 'on')" not in HTML
    assert "/api/bias_tee/' + (on ? 'start' : 'stop')" not in HTML


def test_sweep_clears_current_so_nudge_cannot_retune():
    """After Сканувати, ±0.5 MHz must be a no-op until the operator picks a hit."""
    sweep = HTML[HTML.index("b-sweep').onclick") : HTML.index("b-clear').onclick")]
    assert "current = null" in sweep
    assert "fetch('/api/sweep'" in sweep


def test_hits_are_listed_strongest_first():
    """Operator clicks the first row. Sorting by frequency would hide the loud VTx."""
    assert "sort((a, b) => b.snr_db - a.snr_db)" in HTML


def test_error_notices_use_level_error():
    """Engine emits level='error'. Checking 'err' leaves USB failures looking OK."""
    assert "m.data.level === 'error'" in HTML
