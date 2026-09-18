"""Shipped YAML flags whose wrong value empties the console or kills ffmpeg."""
from __future__ import annotations

import inspect
from pathlib import Path

from fpvscan.config import load
from fpvscan.engine import Engine
from fpvscan.recorder import ffmpeg_path


CFG = load(Path(__file__).resolve().parents[1] / "config.yaml")


def test_shipped_auto_peek_is_on():
    """auto_peek false means Sweep never shows a picture until the operator
    clicks. The look-then-scan workflow (and auto_peek_secs=5) is what the
    shipped yaml was tuned for.
    """
    assert CFG["scan"]["auto_peek"] is True


def test_shipped_spectrum_every_emits_on_the_first_lock_frame():
    """LOCK emits spectrum when `_lock_n % every == 1`. every=1 never
    matches (n%1 is always 0), so the ribbon stays black for the whole
    hold. Shipped 8 is the first-frame tick the console paints.
    """
    every = int(CFG["video"]["spectrum_every"])
    assert every == 8
    assert every > 1
    assert 1 % every == 1
    src = inspect.getsource(Engine._do_lock)
    assert "_lock_n % every == 1" in src


def test_shipped_debug_candidates_is_off():
    """run.py's event queue is 64. debug_candidates=True emits one candidate
    per occupancy blob per step and those events push frames out — LOCK
    video stalls on a busy 5.8 GHz pass. Headless can take 2000; the
    console cannot.
    """
    assert CFG["scan"]["debug_candidates"] is False


def test_empty_ffmpeg_path_is_unset_not_a_missing_file(monkeypatch, tmp_path):
    """Shipped `ffmpeg_path: ''` must not be treated as an explicit path.
    `if explicit:` used to skip it; `if explicit is not None` would raise
    FfmpegMissing on every Запис because YAML loaded an empty string.
    """
    assert CFG["video"]["ffmpeg_path"] == ""
    rec_src = inspect.getsource(Engine._rec_start)
    assert 'v.get("ffmpeg_path") or None' in rec_src
    fake = tmp_path / "ffmpeg"
    fake.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr("fpvscan.recorder.shutil.which", lambda _: str(fake))
    assert ffmpeg_path("") == str(fake)
    assert ffmpeg_path(None) == str(fake)
