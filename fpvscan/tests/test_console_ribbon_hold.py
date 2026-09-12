"""Ribbon frequency mapping and the LOCK stall watchdog."""
from __future__ import annotations

from pathlib import Path

HTML = (Path(__file__).resolve().parents[1]
        / "fpvscan" / "web" / "static" / "index.html").read_text(encoding="utf-8")


def _fn(name: str) -> str:
    start = HTML.index(f"function {name}")
    nxt = HTML.find("\n    function ", start + 1)
    return HTML[start:nxt if nxt != -1 else None]


def test_push_spectrum_maps_bins_by_hz_and_max_holds():
    """Painting by bin index (not center±span/2) parks a 5.8 GHz peak
    on the 433 MHz tick. Overwriting instead of max-hold makes a weak
    VTx vanish on the next sweep step that only sees its skirt.
    """
    body = _fn("pushSpectrum")
    assert "d.center_hz - d.span_hz / 2" in body
    assert "(f - F0) / (F1 - F0)" in body
    # Max-hold with a 0.6 dB leak: a one-step skirt must not erase a
    # real peak, but a gone VTx must fade instead of staining the ribbon.
    assert "Math.max(bins[i], ribData[x] - 0.6)" in body


def test_show_frame_stamps_last_frame_so_live_video_does_not_reconnect():
    """The 6 s LOCK watchdog closes the socket when lastFrame is stale.
    If showFrame never writes lastFrame, a healthy stream reconnects
    every few seconds and the picture blinks.
    """
    body = _fn("showFrame")
    assert "lastFrame = Date.now()" in body
    assert "Date.now() - lastFrame < 6000" in HTML


def test_hit_list_and_state_use_the_same_mhz_bucket():
    """applyState uses Math.round(freq_hz/1e6); the detection WS path
    must too. Floor vs round on 5800.6 MHz yields two cards for one VTx.
    """
    assert HTML.count("hits.set(Math.round(") == 2
    assert "hits.set(Math.round(m.data.freq_hz / 1e6)" in HTML
    assert "hits.set(Math.round(d.freq_hz / 1e6)" in HTML
