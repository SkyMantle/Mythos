from __future__ import annotations

from fpvscan.scan_hits import filter_published, keeps


def _det(freq_mhz: float, snr: float, **kw) -> dict:
    row = {
        "freq_hz": freq_mhz * 1e6,
        "snr_db": snr,
        "band": kw.get("band", "3G3"),
        "pic_score": kw.get("pic_score", 0.0),
        "pic_locked": kw.get("pic_locked", False),
        "confidence": kw.get("confidence", 0.6),
    }
    return row


def test_hide_near_dup_drops_2mhz_beside_stronger() -> None:
    peak = _det(3489, 22.0, pic_score=0.4)
    image = _det(3491, 11.0, pic_score=0.05)
    shown = filter_published([peak, image], "hide_near_dup")
    freqs = [round(d["freq_hz"] / 1e6) for d in shown]
    assert freqs == [3489]
    assert keeps(image, [peak, image], "hide_near_dup") is False
    assert keeps(peak, [peak, image], "hide_near_dup") is True


def test_hide_weak_keeps_3489_class_drops_inband_birdie() -> None:
    vtx = _det(3489, 20.0, pic_score=0.22)
    bird = _det(3520, 6.0, pic_score=0.0)
    shown = filter_published([vtx, bird], "hide_weak")
    freqs = {round(d["freq_hz"] / 1e6) for d in shown}
    assert 3489 in freqs
    assert 3520 not in freqs
    assert filter_published([vtx, bird], "all") == [vtx, bird]


def test_hide_no_video_drops_never_locked() -> None:
    snow = _det(2412, 18.0, pic_score=0.02, pic_locked=False, band="2G4")
    live = _det(4988, 19.0, pic_score=0.35, pic_locked=True, band="5G8")
    shown = filter_published([snow, live], "hide_no_video")
    assert [round(d["freq_hz"] / 1e6) for d in shown] == [4988]
