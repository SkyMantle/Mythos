from __future__ import annotations

from fpvscan.scan_hits import (
    HIT_KEY_HZ,
    HIT_SELECT_HZ,
    LOCK_SPURIOUS_HZ,
    MERGE_CHANNEL_HZ,
    MERGE_LOCK_HZ,
    PRUNE_SETTLE_S,
    filter_published,
    hit_key,
    is_video_green,
    keeps,
    lock_retune,
    same_channel,
    should_prune_empty_lock,
    sticky_published_hz,
)


def _det(freq_mhz: float, snr: float, **kw) -> dict:
    row = {
        "freq_hz": freq_mhz * 1e6,
        "snr_db": snr,
        "band": kw.get("band", "3G3"),
        "pic_score": kw.get("pic_score", 0.0),
        "pic_locked": kw.get("pic_locked", False),
        "confidence": kw.get("confidence", 0.6),
    }
    for key in ("standard", "row_corr", "pic_lines", "lines", "last_picture_at"):
        if key in kw:
            row[key] = kw[key]
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


def test_video_green_needs_analog_and_flowing_frames() -> None:
    flowing = _det(
        4988, 19.0, standard="PAL", pic_locked=True, pic_score=0.4, row_corr=0.3,
    )
    assert is_video_green(flowing, streaming=True) is True
    scored = _det(
        5740, 12.0, standard="NTSC", pic_locked=False, pic_score=0.22, row_corr=0.05,
        last_picture_at=10.0,
    )
    assert is_video_green(scored, now=11.0) is True
    stale = dict(scored)
    assert is_video_green(stale, now=20.0) is False
    snr_only = _det(3489, 40.0, standard="?", pic_locked=False, pic_score=0.0)
    assert is_video_green(snr_only, streaming=True) is False
    published_no_stream = _det(
        4988, 18.0, standard="PAL", pic_locked=True, pic_score=0.5,
    )
    assert is_video_green(published_no_stream) is False


def test_prune_empty_lock_waits_and_only_selected() -> None:
    empty = _det(2412, 16.0, pic_locked=False, pic_score=0.04, row_corr=0.01, pic_lines=0)
    live = _det(
        4988, 18.0, standard="PAL", pic_locked=True, pic_score=0.4,
        row_corr=0.3, pic_lines=250,
    )
    assert PRUNE_SETTLE_S == 5.0
    assert should_prune_empty_lock(
        operator_selected=False, auto_lock=False, elapsed_s=9.0, sample=empty,
    ) is False
    assert should_prune_empty_lock(
        operator_selected=True, auto_lock=True, elapsed_s=9.0, sample=empty,
    ) is False
    assert should_prune_empty_lock(
        operator_selected=True, auto_lock=False, elapsed_s=2.0, sample=empty,
    ) is False
    assert should_prune_empty_lock(
        operator_selected=True, auto_lock=False, elapsed_s=5.0, sample=empty,
    ) is True
    assert should_prune_empty_lock(
        operator_selected=True, auto_lock=False, elapsed_s=9.0, sample=live,
    ) is False
    raster = _det(
        3333, 10.0, pic_locked=False, pic_score=0.08, row_corr=0.14, pic_lines=120,
    )
    assert should_prune_empty_lock(
        operator_selected=True, auto_lock=False, elapsed_s=9.0, sample=raster,
    ) is False


def test_same_channel_covers_hz_and_10khz_wander() -> None:
    base = 3597e6
    assert same_channel(base, base + 1) is True
    assert same_channel(base, base + 10e3) is True
    assert same_channel(base, base + MERGE_CHANNEL_HZ) is True
    assert same_channel(base, base + MERGE_CHANNEL_HZ + 1) is False
    assert hit_key(base) == hit_key(base + 1)
    assert hit_key(base) == hit_key(base + 10e3)
    assert hit_key(base) != hit_key(base + 100e3)
    assert HIT_KEY_HZ == 50e3


def test_lock_retune_nudge_01_mhz_is_not_noop() -> None:
    base = 3597e6
    target, skip = lock_retune(base + 1, base, locked=True)
    assert skip is True
    assert target == base
    target, skip = lock_retune(base + 10e3, base, locked=True)
    assert skip is True
    target, skip = lock_retune(base + 100e3, base, locked=True)
    assert skip is False
    assert target == base + 100e3
    target, skip = lock_retune(base + 100e3, base, force=True, locked=True)
    assert skip is False
    assert target == base + 100e3
    target, skip = lock_retune(base, base, force=True, locked=True)
    assert skip is False
    assert LOCK_SPURIOUS_HZ < 100e3


def test_sticky_published_hz_freezes_and_pins_lock() -> None:
    base = 3597e6
    assert sticky_published_hz(base, base + 1) == base
    assert sticky_published_hz(base, base + 10e3) == base
    assert sticky_published_hz(
        base, base + 1500, lock_target=base, locked=True,
    ) == base
    assert sticky_published_hz(
        base, base + 1, lock_target=base + 100e3, locked=True,
    ) == base + 100e3
    assert MERGE_LOCK_HZ == HIT_SELECT_HZ == 2e6
    assert sticky_published_hz(
        4989e6, 4989e6, lock_target=4990.5e6, locked=True,
    ) == 4990.5e6
    assert sticky_published_hz(
        4990.5e6, 4989e6, lock_target=4990.5e6, locked=True,
    ) == 4990.5e6
    assert sticky_published_hz(
        4990.5e6, 5000.5e6, lock_target=4990.5e6, locked=True,
    ) == 5000.5e6
