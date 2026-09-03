"""Output paths must stay under out/, not the caller's cwd."""
from __future__ import annotations

import re
from pathlib import Path

from fpvscan import paths


def test_stamped_includes_prefix_and_mhz():
    name = paths.stamped("rec", "mp4", 5800e6)
    assert re.fullmatch(r"rec_\d{8}-\d{6}_5800M\.mp4", name)


def test_stamped_without_freq_or_prefix():
    assert paths.stamped("shot", "webp").endswith(".webp")
    assert paths.stamped("shot", "webp").startswith("shot_")
    bare = paths.stamped("", "bin", None)
    assert re.fullmatch(r"\d{8}-\d{6}\.bin", bare)


def test_resolve_bare_name_goes_to_default_dir(tmp_path: Path):
    p = paths.resolve("clip.mp4", tmp_path / "video", "unused.mp4")
    assert p == tmp_path / "video" / "clip.mp4"
    assert p.parent.is_dir()


def test_resolve_absolute_path_is_kept(tmp_path: Path):
    dest = tmp_path / "elsewhere" / "a.mp4"
    p = paths.resolve(str(dest), tmp_path / "video", "unused.mp4")
    assert p == dest
    assert dest.parent.is_dir()


def test_resolve_none_uses_default(tmp_path: Path):
    p = paths.resolve(None, tmp_path / "caps", "cap.cf32")
    assert p == tmp_path / "caps" / "cap.cf32"
