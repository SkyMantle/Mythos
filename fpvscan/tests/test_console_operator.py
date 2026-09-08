"""Console contracts operators hit every session: MHz entry, clip, reconnect."""
from __future__ import annotations

from pathlib import Path

HTML = (Path(__file__).resolve().parents[1]
        / "fpvscan" / "web" / "static" / "index.html").read_text(encoding="utf-8")


def test_manual_lock_converts_mhz_and_ignores_empty():
    """Typing 5800 must lock 5.8 GHz, not 5.8 kHz. Empty Enter must not fire."""
    assert "lock(v * 1e6)" in HTML
    assert "if (!isNaN(v) && v > 0)" in HTML
    assert "parseFloat(document.getElementById('manual-freq').value)" in HTML


def test_clip_frac_marks_bad_above_one_tenth_percent():
    """>0.1% clipped ADC samples are IM products, not extra VTx."""
    assert "cf > 0.1 ? 'bad'" in HTML
    assert "(s.clip_frac || 0) * 100" in HTML


def test_stalled_lock_closes_socket_after_six_seconds():
    """Without this, a hung WS leaves a frozen picture and no reconnect."""
    assert "Date.now() - lastFrame < 6000" in HTML
    assert "curMode !== 'LOCK'" in HTML
    assert "sock.close()" in HTML
    assert "потік кадрів мовчить" in HTML


def test_websocket_uses_wss_on_https():
    assert "location.protocol === 'https:' ? 'wss' : 'ws'" in HTML


def test_lock_mode_hides_waterfall_via_watching_class():
    """LOCK must give the picture the full pane; waterfall stays for SWEEP."""
    assert "classList.toggle('watching', s.mode === 'LOCK')" in HTML
    assert "main.watching #sec-ribbon" in HTML
    assert "main.watching #sec-fall" in HTML


def test_nudge_is_a_noop_until_a_channel_is_current():
    assert HTML.count("if (!current) return") >= 2
    assert "lock(current - 0.5e6)" in HTML
    assert "lock(current + 0.5e6)" in HTML
