"""paths.ensure must create a parent for files and the directory itself when bare."""
from __future__ import annotations

from pathlib import Path

from fpvscan.paths import CAPS, FRAMES, PHOTOS, VIDEO, ensure


def test_ensure_file_creates_parent_not_the_file(tmp_path: Path):
    dst = tmp_path / "photos" / "shot.webp"
    out = ensure(dst)
    assert out == dst
    assert dst.parent.is_dir()
    assert not dst.exists()


def test_ensure_suffixless_creates_the_directory(tmp_path: Path):
    d = tmp_path / "caps"
    out = ensure(d)
    assert out == d
    assert d.is_dir()


def test_frames_alias_is_photos():
    """Old call sites use FRAMES. If the alias drifts from PHOTOS,
    snapshots land in a second tree and the operator cannot find them.
    """
    assert FRAMES == PHOTOS
    assert PHOTOS.name == "photos"
    assert VIDEO.name == "video"
    assert CAPS.name == "caps"
