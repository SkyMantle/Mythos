"""An explicit ffmpeg_path must not silently fall through to PATH."""
from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from fpvscan.recorder import FfmpegMissing, ffmpeg_path


def test_explicit_missing_path_does_not_search_path():
    """YAML `video.ffmpeg_path` is what the operator intended.
    Falling through to `shutil.which` after a typo records with a
    different binary (wrong codecs / an old WinGet copy) while the
    console still says the configured path.
    """
    with patch("fpvscan.recorder.shutil.which") as which:
        with pytest.raises(FfmpegMissing, match="не існує") as ei:
            ffmpeg_path("/definitely/missing/fpvscan-ffmpeg")
        which.assert_not_called()
    assert "fpvscan-ffmpeg" in str(ei.value)


def test_explicit_existing_path_is_used_as_is():
    assert ffmpeg_path(sys.executable) == sys.executable
